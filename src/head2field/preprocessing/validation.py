"""Read-only prepared-subject v1 loader independent of FEM labels."""

from __future__ import annotations

from head2field.metadata import is_digest_key

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib

from .conductivity import (
    CONDUCTIVITY_TABLE_ID,
    CONDUCTIVITY_UNIT,
    CONDUCTIVITY_VALUES,
    MODEL_SIGMA_DIVISOR_S_PER_M,
    MODEL_SIGMA_TRANSFORM_ID,
    conductivity_table_metadata,
    labels_to_conductivity,
    normalize_conductivity_for_model,
    validate_conductivity_arrays,
    validate_conductivity_table,
)
from .electrodes import STANDARD_76_ELECTRODE_NAMES

PREPARED_SUBJECT_SCHEMA_ID = "prepared_subject_v1"
PREPARED_SUBJECT_MANIFEST_NAME = "prepared_subject_v1.json"
PREPARED_SUBJECT_SHAPE = (256, 256, 256)
PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0, -128.0),
    (0.0, 1.0, 0.0, -128.0),
    (0.0, 0.0, 1.0, -128.0),
    (0.0, 0.0, 0.0, 1.0),
)
PREPARED_SUBJECT_VOXEL_SIZE_MM = (1.0, 1.0, 1.0)
REFERENCE_ELECTRODE = "Oz"
ELECTRODE_COUNT = 76
SOURCE_ELECTRODE_COUNT = 75

REQUIRED_PREPARED_SUBJECT_FILES: tuple[str, ...] = (
    "images/T1_1mm.nii.gz",
    "images/final_tissues.nii.gz",
    "images/sigma_1mm.nii.gz",
    "voxels/gmwm_ijk.npy",
    "electrodes/electrode_names_all.json",
    "electrodes/electrode_names_no_oz.json",
    "electrodes/electrode_coords_mm.npy",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PreparedSubjectV1:
    """Fully validated read-only data ready for v1 model-input assembly."""

    subject_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    t1_1mm: np.ndarray
    tissue_labels_1mm: np.ndarray
    sigma_s_per_m_1mm: np.ndarray
    sigma_model_1mm: np.ndarray
    affine: np.ndarray
    inverse_affine: np.ndarray
    full_shape: tuple[int, int, int]
    gmwm_ijk: np.ndarray
    electrode_names_all: tuple[str, ...]
    source_names: tuple[str, ...]
    electrode_coords_mm: np.ndarray
    source_coords_mm: np.ndarray
    reference_name: str
    reference_index: int
    reference_coord_mm: np.ndarray
    head_model_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedAssets:
    """Internal read-only assets shared by manifest construction and loading."""

    t1: np.ndarray
    tissues: np.ndarray
    sigma: np.ndarray
    sigma_model: np.ndarray
    affine: np.ndarray
    shape: tuple[int, int, int]
    gmwm_ijk: np.ndarray
    names_all: tuple[str, ...]
    source_names: tuple[str, ...]
    coords: np.ndarray
    reference_index: int


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Generate canonical manifest bytes independent of formatting and key order."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def prepared_subject_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Compute the canonical SHA256 of a prepared-subject manifest."""

    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def _readonly(array: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """Return a contiguous array owning its memory and protected against accidental writes."""

    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    """Read UTF-8 JSON whose top-level value must be an object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{role} 不是有效 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{role} 顶层必须是 JSON 对象: {path}")
    return payload


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    role: str,
) -> None:
    """Match runtime field checks to JSON Schema additionalProperties=false."""

    actual = {key for key in payload if not is_digest_key(key)}
    expected = {key for key in expected if not is_digest_key(key)}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{role} 字段集合不一致：missing={missing}, extra={extra}")


def _load_name_list(path: Path, role: str) -> tuple[str, ...]:
    """Strictly read a nonempty JSON electrode-name list without duplicates."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{role} 不是有效 JSON: {path}") from error
    if not isinstance(payload, list) or any(
        not isinstance(name, str) or not name.strip() for name in payload
    ):
        raise TypeError(f"{role} 必须是非空字符串组成的 JSON 列表")
    names = tuple(payload)
    if len(set(names)) != len(names):
        raise ValueError(f"{role} 包含重复电极名")
    return names


def _load_nifti_3d(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Load a 3D NIfTI read-only, accepting trailing singleton dimensions."""

    import nibabel as nib

    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: NIfTI 必须为三维，实际 shape={data.shape}")
    return image, data


def _validate_grid(
    images: Sequence[nib.spatialimages.SpatialImage],
    arrays: Sequence[np.ndarray],
    *,
    expected_shape: tuple[int, int, int],
    expected_affine: Sequence[Sequence[float]],
    affine_atol: float,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Require all three volumes to align on the canonical 1 mm RAS grid."""

    import nibabel as nib

    shapes = [tuple(int(value) for value in array.shape) for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"T1/tissue/sigma shape 不一致: {shapes}")
    shape = shapes[0]
    if shape != tuple(expected_shape):
        raise ValueError(
            f"prepared-subject shape={shape}，期望 {tuple(expected_shape)}"
        )

    affines = [np.asarray(image.affine, dtype=np.float64) for image in images]
    if any(not np.all(np.isfinite(affine)) for affine in affines):
        raise ValueError("T1/tissue/sigma affine 含 NaN 或 Inf")
    base_affine = affines[0]
    if any(
        not np.allclose(base_affine, affine, rtol=0.0, atol=affine_atol)
        for affine in affines[1:]
    ):
        raise ValueError("T1/tissue/sigma affine 不一致")

    frozen_affine = np.asarray(expected_affine, dtype=np.float64)
    if frozen_affine.shape != (4, 4) or not np.all(np.isfinite(frozen_affine)):
        raise ValueError("expected_affine 必须是有限 4x4 矩阵")
    if not np.allclose(
        base_affine,
        frozen_affine,
        rtol=0.0,
        atol=affine_atol,
    ):
        raise ValueError("prepared-subject affine 与冻结 canonical affine 不一致")

    for role, affine in zip(("T1", "tissue", "sigma"), affines, strict=True):
        orientation = "".join(nib.aff2axcodes(affine))
        if orientation != "RAS":
            raise ValueError(f"{role} orientation 必须为 RAS，实际为 {orientation}")
        voxel_sizes = np.asarray(nib.affines.voxel_sizes(affine), dtype=np.float64)
        if not np.allclose(
            voxel_sizes,
            PREPARED_SUBJECT_VOXEL_SIZE_MM,
            rtol=0.0,
            atol=affine_atol,
        ):
            raise ValueError(f"{role} voxel size 必须为 1 mm，实际为 {voxel_sizes}")
    return shape, base_affine


def _load_and_validate_assets(
    subject_dir: Path,
    *,
    expected_shape: tuple[int, int, int],
    expected_affine: Sequence[Sequence[float]],
    affine_atol: float,
) -> _PreparedAssets:
    """Read only minimal inference files and validate values and query order."""

    root = Path(subject_dir)
    paths = {relative: root / relative for relative in REQUIRED_PREPARED_SUBJECT_FILES}
    for relative, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"prepared-subject 缺少 {relative}: {path}")

    t1_image, raw_t1 = _load_nifti_3d(paths["images/T1_1mm.nii.gz"])
    tissue_image, raw_tissues = _load_nifti_3d(paths["images/final_tissues.nii.gz"])
    sigma_image, raw_sigma = _load_nifti_3d(paths["images/sigma_1mm.nii.gz"])
    shape, affine = _validate_grid(
        (t1_image, tissue_image, sigma_image),
        (raw_t1, raw_tissues, raw_sigma),
        expected_shape=expected_shape,
        expected_affine=expected_affine,
        affine_atol=affine_atol,
    )
    if np.dtype(t1_image.get_data_dtype()) != np.dtype(np.float32):
        raise ValueError(
            f"T1_1mm NIfTI dtype 必须为 float32，实际为 {t1_image.get_data_dtype()}"
        )
    if np.dtype(sigma_image.get_data_dtype()) != np.dtype(np.float32):
        raise ValueError(
            "sigma_1mm NIfTI dtype 必须为 float32，"
            f"实际为 {sigma_image.get_data_dtype()}"
        )

    t1 = np.asarray(raw_t1, dtype=np.float32)
    if not np.all(np.isfinite(t1)):
        raise ValueError("T1_1mm 含 NaN 或 Inf")
    # labels_to_conductivity checks exact integers and the 0--10 range together.
    expected_sigma = labels_to_conductivity(raw_tissues)
    tissues = np.asarray(raw_tissues, dtype=np.int16)
    sigma = np.asarray(raw_sigma)
    validate_conductivity_arrays(
        tissues,
        sigma,
        context=str(root),
    )
    if not np.array_equal(expected_sigma, sigma):
        raise RuntimeError("内部错误：共享逐体素电导率门禁返回后仍不一致")
    sigma_model = normalize_conductivity_for_model(sigma)

    gmwm_path = paths["voxels/gmwm_ijk.npy"]
    raw_gmwm = np.load(gmwm_path, allow_pickle=False)
    if (
        raw_gmwm.dtype == np.bool_
        or not np.issubdtype(raw_gmwm.dtype, np.integer)
        or raw_gmwm.ndim != 2
        or raw_gmwm.shape[1:] != (3,)
        or len(raw_gmwm) == 0
    ):
        raise TypeError("gmwm_ijk 必须是非空 [Q,3] 整数数组")
    gmwm = np.asarray(raw_gmwm, dtype=np.int64)
    if np.any(gmwm < 0) or np.any(gmwm >= np.asarray(shape, dtype=np.int64)[None]):
        raise IndexError("gmwm_ijk 含越界坐标")
    expected_gmwm = np.argwhere(np.isin(tissues, (1, 2)))
    if not np.array_equal(gmwm, expected_gmwm):
        raise ValueError(
            "gmwm_ijk 必须严格等于 np.argwhere(tissue in {1,2}) 的 C-order 点序"
        )

    names_all = _load_name_list(
        paths["electrodes/electrode_names_all.json"],
        "electrode_names_all",
    )
    source_names = _load_name_list(
        paths["electrodes/electrode_names_no_oz.json"],
        "electrode_names_no_oz",
    )
    if len(names_all) != ELECTRODE_COUNT:
        raise ValueError(
            f"electrode_names_all 必须含 {ELECTRODE_COUNT} 名电极，实际 {len(names_all)}"
        )
    if names_all.count(REFERENCE_ELECTRODE) != 1:
        raise ValueError("electrode_names_all 必须且只能包含一个 Oz reference")
    if names_all != STANDARD_76_ELECTRODE_NAMES:
        raise ValueError("electrode_names_all 必须严格等于 v1 固定 76 电极顺序")
    expected_sources = tuple(name for name in names_all if name != REFERENCE_ELECTRODE)
    if len(source_names) != SOURCE_ELECTRODE_COUNT or source_names != expected_sources:
        raise ValueError(
            "electrode_names_no_oz 必须按 electrode_names_all 原顺序列出其余 75 个 source"
        )

    raw_coords = np.load(
        paths["electrodes/electrode_coords_mm.npy"],
        allow_pickle=False,
    )
    if (
        raw_coords.dtype == np.bool_
        or not np.issubdtype(raw_coords.dtype, np.number)
        or np.iscomplexobj(raw_coords)
        or raw_coords.shape != (ELECTRODE_COUNT, 3)
    ):
        raise TypeError(f"electrode_coords_mm 必须是实数 [{ELECTRODE_COUNT},3] 数组")
    coords = np.asarray(raw_coords, dtype=np.float32)
    if not np.all(np.isfinite(coords)):
        raise ValueError("electrode_coords_mm 含 NaN 或 Inf")
    reference_index = names_all.index(REFERENCE_ELECTRODE)
    return _PreparedAssets(
        t1=_readonly(t1, dtype=np.float32),
        tissues=_readonly(tissues, dtype=np.int16),
        sigma=_readonly(sigma, dtype=np.float32),
        sigma_model=_readonly(sigma_model, dtype=np.float32),
        affine=_readonly(affine, dtype=np.float64),
        shape=shape,
        gmwm_ijk=_readonly(gmwm, dtype=np.int64),
        names_all=names_all,
        source_names=source_names,
        coords=_readonly(coords, dtype=np.float32),
        reference_index=reference_index,
    )


def _manifest_file_entries(subject_dir: Path) -> dict[str, dict[str, Any]]:
    """Build relative-path, size, and hash records for minimal inference files."""

    root = Path(subject_dir)
    entries: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_PREPARED_SUBJECT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"prepared-subject 缺少 {relative}: {path}")
        stat = path.stat()
        entries[relative] = {
            "size": int(stat.st_size),
        }
    return entries


def build_prepared_subject_manifest(
    subject_dir: Path,
    *,
    head_model_provenance: Mapping[str, Any],
    expected_shape: tuple[int, int, int] = PREPARED_SUBJECT_SHAPE,
    expected_affine: Sequence[Sequence[float]] = PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    affine_atol: float = 1e-5,
) -> dict[str, Any]:
    """Inspect staging assets read-only and build a manifest without writing files."""

    assets = _load_and_validate_assets(
        Path(subject_dir),
        expected_shape=expected_shape,
        expected_affine=expected_affine,
        affine_atol=affine_atol,
    )
    files = _manifest_file_entries(Path(subject_dir))
    return {
        "schema_id": PREPARED_SUBJECT_SCHEMA_ID,
        "shape": list(assets.shape),
        "voxel_size_mm": list(PREPARED_SUBJECT_VOXEL_SIZE_MM),
        "orientation": "RAS",
        "affine_ijk_to_ras_mm": assets.affine.tolist(),
        "reference_electrode": REFERENCE_ELECTRODE,
        "electrode_count": ELECTRODE_COUNT,
        "source_count": SOURCE_ELECTRODE_COUNT,
        "query_count": len(assets.gmwm_ijk),
        "conductivity_table": conductivity_table_metadata(),
        "model_input_transform": {
            "id": MODEL_SIGMA_TRANSFORM_ID,
            "divisor_s_per_m": MODEL_SIGMA_DIVISOR_S_PER_M,
            "output_dtype": "float32",
        },
        "head_model_provenance": dict(head_model_provenance),
        "files": files,
    }


def _validate_file_manifest(
    subject_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Validate the minimal file set, sizes, and optional per-file SHA256 hashes."""

    raw_files = manifest.get("files")
    if not isinstance(raw_files, Mapping):
        raise TypeError("prepared-subject manifest files 必须是 JSON 对象")
    actual_keys = set(raw_files)
    expected_keys = set(REQUIRED_PREPARED_SUBJECT_FILES)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"prepared-subject manifest 文件集合不一致，missing={missing}, extra={extra}"
        )

    root = Path(subject_dir)
    for relative in REQUIRED_PREPARED_SUBJECT_FILES:
        entry = raw_files[relative]
        if not isinstance(entry, Mapping):
            raise TypeError(f"manifest files[{relative!r}] 必须是对象")
        _require_exact_keys(
            entry,
            {"sha256", "size"},
            f"manifest files[{relative!r}]",
        )
        entry.get("sha256")
        declared_size = entry.get("size")
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or declared_size < 0
        ):
            raise TypeError(f"manifest files[{relative!r}].size 必须是非负整数")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"prepared-subject 缺少 {relative}: {path}")
        actual_size = int(path.stat().st_size)
        if actual_size != declared_size:
            raise ValueError(
                f"{relative}: size 与 manifest 不一致：{actual_size} != {declared_size}"
            )


def _validate_manifest_against_assets(
    manifest: Mapping[str, Any],
    assets: _PreparedAssets,
) -> None:
    """Cross-check manifest declarations against numerically validated assets."""

    _require_exact_keys(
        manifest,
        {
            "schema_id",
            "shape",
            "voxel_size_mm",
            "orientation",
            "affine_ijk_to_ras_mm",
            "reference_electrode",
            "electrode_count",
            "source_count",
            "query_count",
            "conductivity_table",
            "sigma_source_final_tissues_sha256",
            "model_input_transform",
            "head_model_provenance",
            "files",
        },
        "prepared-subject manifest",
    )
    if manifest.get("schema_id") != PREPARED_SUBJECT_SCHEMA_ID:
        raise ValueError(
            f"prepared-subject schema_id 必须为 {PREPARED_SUBJECT_SCHEMA_ID!r}"
        )
    validate_conductivity_table(manifest.get("conductivity_table", {}))
    if manifest.get("reference_electrode") != REFERENCE_ELECTRODE:
        raise ValueError("prepared-subject reference_electrode 必须为 'Oz'")
    if manifest.get("electrode_count") != ELECTRODE_COUNT:
        raise ValueError(f"prepared-subject electrode_count 必须为 {ELECTRODE_COUNT}")
    if manifest.get("source_count") != SOURCE_ELECTRODE_COUNT:
        raise ValueError(
            f"prepared-subject source_count 必须为 {SOURCE_ELECTRODE_COUNT}"
        )
    if manifest.get("query_count") != len(assets.gmwm_ijk):
        raise ValueError("prepared-subject query_count 与 gmwm_ijk 行数不一致")
    if manifest.get("shape") != list(assets.shape):
        raise ValueError("prepared-subject manifest shape 与 NIfTI 不一致")
    if manifest.get("voxel_size_mm") != list(PREPARED_SUBJECT_VOXEL_SIZE_MM):
        raise ValueError("prepared-subject voxel_size_mm 必须为 [1.0,1.0,1.0]")
    if manifest.get("orientation") != "RAS":
        raise ValueError("prepared-subject orientation 必须为 'RAS'")
    try:
        declared_affine = np.asarray(
            manifest["affine_ijk_to_ras_mm"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prepared-subject manifest affine 无效") from error
    if declared_affine.shape != (4, 4) or not np.array_equal(
        declared_affine,
        assets.affine,
    ):
        raise ValueError("prepared-subject manifest affine 与 NIfTI 不一致")

    transform = manifest.get("model_input_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("prepared-subject manifest 缺少 model_input_transform")
    _require_exact_keys(
        transform,
        {"id", "divisor_s_per_m", "output_dtype"},
        "model_input_transform",
    )
    if transform.get("id") != MODEL_SIGMA_TRANSFORM_ID:
        raise ValueError("prepared-subject sigma 模型变换 ID 不一致")
    if transform.get("divisor_s_per_m") != MODEL_SIGMA_DIVISOR_S_PER_M:
        raise ValueError("prepared-subject sigma 模型变换除数不一致")
    if transform.get("output_dtype") != "float32":
        raise ValueError("prepared-subject sigma 模型变换 dtype 必须为 float32")

    provenance = manifest.get("head_model_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("prepared-subject manifest 缺少 head_model_provenance")
    _require_exact_keys(
        provenance,
        {
            "schema_id",
            "runtime_manifest_sha256",
            "simnibs_version",
            "canonical_t1_sha256",
            "charm_t1_sha256",
            "final_tissues_sha256",
            "head_mesh_sha256",
        },
        "head_model_provenance",
    )
    if provenance.get("schema_id") != "charm_runtime_binding_v1":
        raise ValueError("head_model_provenance schema_id 不匹配")
    if provenance.get("simnibs_version") != "4.6.0":
        raise ValueError("head_model_provenance SimNIBS 版本必须为 4.6.0")


def validate_prepared_subject_manifest(
    subject_dir: Path,
    manifest: Mapping[str, Any] | None = None,
    *,
    expected_shape: tuple[int, int, int] = PREPARED_SUBJECT_SHAPE,
    expected_affine: Sequence[Sequence[float]] = PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    affine_atol: float = 1e-5,
) -> str:
    """Validate the manifest, all file hashes, and prepared inputs read-only; return a canonical hash."""

    root = Path(subject_dir)
    payload = (
        _load_json_object(
            root / PREPARED_SUBJECT_MANIFEST_NAME,
            "prepared-subject manifest",
        )
        if manifest is None
        else dict(manifest)
    )
    _validate_file_manifest(
        root,
        payload,
    )
    assets = _load_and_validate_assets(
        root,
        expected_shape=expected_shape,
        expected_affine=expected_affine,
        affine_atol=affine_atol,
    )
    _validate_manifest_against_assets(payload, assets)
    return prepared_subject_manifest_sha256(payload)


def load_prepared_subject_v1(
    subject_dir: Path,
    *,
    expected_shape: tuple[int, int, int] = PREPARED_SUBJECT_SHAPE,
    expected_affine: Sequence[Sequence[float]] = PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    affine_atol: float = 1e-5,
) -> PreparedSubjectV1:
    """Read and validate only v1 inference inputs without accessing ``labels/`` or writing files."""

    root = Path(subject_dir)
    manifest_path = root / PREPARED_SUBJECT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 prepared-subject manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path, "prepared-subject manifest")
    _validate_file_manifest(
        root,
        manifest,
    )
    assets = _load_and_validate_assets(
        root,
        expected_shape=expected_shape,
        expected_affine=expected_affine,
        affine_atol=affine_atol,
    )
    _validate_manifest_against_assets(manifest, assets)

    source_indices = [
        index
        for index, name in enumerate(assets.names_all)
        if name != REFERENCE_ELECTRODE
    ]
    source_coords = _readonly(assets.coords[source_indices], dtype=np.float32)
    reference_coord = _readonly(
        assets.coords[assets.reference_index],
        dtype=np.float32,
    )
    inverse_affine = _readonly(np.linalg.inv(assets.affine), dtype=np.float64)
    return PreparedSubjectV1(
        subject_dir=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=prepared_subject_manifest_sha256(manifest),
        t1_1mm=assets.t1,
        tissue_labels_1mm=assets.tissues,
        sigma_s_per_m_1mm=assets.sigma,
        sigma_model_1mm=assets.sigma_model,
        affine=assets.affine,
        inverse_affine=inverse_affine,
        full_shape=assets.shape,
        gmwm_ijk=assets.gmwm_ijk,
        electrode_names_all=assets.names_all,
        source_names=assets.source_names,
        electrode_coords_mm=assets.coords,
        source_coords_mm=source_coords,
        reference_name=REFERENCE_ELECTRODE,
        reference_index=assets.reference_index,
        reference_coord_mm=reference_coord,
        head_model_provenance=dict(manifest["head_model_provenance"]),
    )


__all__ = [
    "CONDUCTIVITY_TABLE_ID",
    "CONDUCTIVITY_UNIT",
    "CONDUCTIVITY_VALUES",
    "ELECTRODE_COUNT",
    "PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM",
    "PREPARED_SUBJECT_MANIFEST_NAME",
    "PREPARED_SUBJECT_SCHEMA_ID",
    "PREPARED_SUBJECT_SHAPE",
    "PREPARED_SUBJECT_VOXEL_SIZE_MM",
    "REFERENCE_ELECTRODE",
    "REQUIRED_PREPARED_SUBJECT_FILES",
    "SOURCE_ELECTRODE_COUNT",
    "PreparedSubjectV1",
    "build_prepared_subject_manifest",
    "load_prepared_subject_v1",
    "prepared_subject_manifest_sha256",
    "validate_prepared_subject_manifest",
]
