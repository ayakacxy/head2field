"""Fixed v1 scalar-conductivity table, model transform, and read-only consistency checks."""

from __future__ import annotations

from head2field.metadata import scientific_metadata

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib


CONDUCTIVITY_TABLE_ID = "simnibs-4.6-isotropic-v1"
CONDUCTIVITY_UNIT = "S/m"
CONDUCTIVITY_VALUES: tuple[float, ...] = (
    0.0,
    0.126,
    0.275,
    1.654,
    0.010,
    0.465,
    0.500,
    0.008,
    0.025,
    0.600,
    0.160,
)
CONDUCTIVITY_METADATA_SCHEMA_ID = "subject_conductivity_v1"
MODEL_SIGMA_DIVISOR_S_PER_M = 1.654
MODEL_SIGMA_TRANSFORM_ID = "sigma_s_per_m_divide_1.654_float32_v1"

# Preserve the public mapping name so callers need not copy or reorder the frozen table.
SIMNIBS_SCALAR_CONDUCTIVITY: dict[int, float] = dict(enumerate(CONDUCTIVITY_VALUES))

TISSUE_NAMES: dict[int, str] = {
    0: "background",
    1: "white_matter",
    2: "gray_matter",
    3: "csf",
    4: "bone",
    5: "scalp",
    6: "eyeballs",
    7: "compact_bone",
    8: "spongy_bone",
    9: "blood",
    10: "muscle",
}


@dataclass(frozen=True)
class ConductivityValidation:
    """Auditable result of a read-only conductivity validation."""

    subject_dir: Path
    tissue_path: Path
    sigma_path: Path
    metadata_path: Path
    shape: tuple[int, int, int]
    affine: np.ndarray
    source_tissues_sha256: str
    sigma_sha256: str
    table_id: str = CONDUCTIVITY_TABLE_ID


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash files in chunks without reading an entire compressed NIfTI into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Generate canonical JSON bytes using the project's frozen rules."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def conductivity_table_metadata() -> dict[str, Any]:
    """Return an independent copy of the full 0--10 table and its canonical hash."""

    return {
        "id": CONDUCTIVITY_TABLE_ID,
        "unit": CONDUCTIVITY_UNIT,
        "values": list(CONDUCTIVITY_VALUES),
    }


def validate_conductivity_table(table: Mapping[str, Any]) -> None:
    """Strictly check table ID, units, complete ordered values, and declared/recomputed hashes."""

    if not isinstance(table, Mapping):
        raise TypeError("conductivity_table 必须是 JSON 对象")
    expected_keys = {"id", "unit", "values"}
    table = scientific_metadata(table)
    if set(table) != expected_keys:
        missing = sorted(expected_keys - set(table))
        extra = sorted(set(table) - expected_keys)
        raise ValueError(
            f"conductivity_table 字段集合不一致：missing={missing}, extra={extra}"
        )
    if table.get("id") != CONDUCTIVITY_TABLE_ID:
        raise ValueError(
            "conductivity table ID 不一致："
            f"{table.get('id')!r} != {CONDUCTIVITY_TABLE_ID!r}"
        )
    if table.get("unit") != CONDUCTIVITY_UNIT:
        raise ValueError(f"conductivity table 单位必须为 {CONDUCTIVITY_UNIT!r}")
    raw_values = table.get("values")
    if not isinstance(raw_values, list) or len(raw_values) != len(CONDUCTIVITY_VALUES):
        raise ValueError("conductivity table 必须完整包含 label 0--10 的 11 个值")
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as error:
        raise TypeError("conductivity table values 必须全部是数值") from error
    if values != CONDUCTIVITY_VALUES:
        raise ValueError("conductivity table values 与冻结的 0--10 顺序不一致")


def _strict_tissue_labels(labels: np.ndarray) -> np.ndarray:
    """Accept integer dtypes or exact integer values; never round tissue labels."""

    raw = np.asarray(labels)
    if raw.dtype == np.bool_ or not (
        np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)
    ):
        raise TypeError("组织标签必须使用整数或精确整数值的实数 dtype")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(np.isfinite(raw)):
            raise ValueError("组织标签含 NaN 或 Inf")
        if not np.array_equal(raw, np.trunc(raw)):
            raise ValueError("组织标签包含非整数值；v1 禁止四舍五入")

    unique = np.unique(raw)
    allowed = np.arange(len(CONDUCTIVITY_VALUES), dtype=np.int16)
    unknown = unique[~np.isin(unique, allowed)]
    if unknown.size:
        raise ValueError(f"发现没有电导率定义的组织标签: {unknown.tolist()}")
    return raw.astype(np.int16, copy=False)


def _load_3d_data(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Read a 3D NIfTI, accepting trailing singleton dimensions in legacy tissue files."""

    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"体数据必须为三维，实际 shape={data.shape}")
    return data


def labels_to_conductivity(labels: np.ndarray) -> np.ndarray:
    """Map exact integer SimNIBS labels to float32 raw scalar conductivity."""

    validated = _strict_tissue_labels(labels)
    lookup = np.asarray(CONDUCTIVITY_VALUES, dtype=np.float32)
    return np.asarray(lookup[validated], dtype=np.float32)


def normalize_conductivity_for_model(sigma_s_per_m: np.ndarray) -> np.ndarray:
    """Apply the shared v1 training/inference float32 ``sigma / 1.654`` transform."""

    raw = np.asarray(sigma_s_per_m)
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("sigma_s_per_m 必须是实数数组")
    if np.iscomplexobj(raw):
        raise TypeError("sigma_s_per_m 不能是复数数组")
    sigma = np.asarray(raw, dtype=np.float32)
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma_s_per_m 含 NaN 或 Inf")
    if np.any(sigma < 0.0):
        raise ValueError("sigma_s_per_m 不能为负数")
    return np.asarray(
        sigma / np.float32(MODEL_SIGMA_DIVISOR_S_PER_M),
        dtype=np.float32,
    )


def _validate_sigma_array(
    labels: np.ndarray,
    sigma: np.ndarray,
    *,
    context: str,
) -> None:
    """Check stored S/m values against the frozen lookup table voxel by voxel."""

    if sigma.dtype != np.dtype(np.float32):
        raise ValueError(f"{context}: sigma dtype 必须为 float32，实际为 {sigma.dtype}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError(f"{context}: sigma 含 NaN 或 Inf")
    expected = labels_to_conductivity(labels)
    if expected.shape != sigma.shape:
        raise ValueError(
            f"{context}: tissue/sigma shape 不一致：{expected.shape} vs {sigma.shape}"
        )
    mismatch = expected != sigma
    if np.any(mismatch):
        count = int(np.count_nonzero(mismatch))
        max_abs = float(
            np.max(
                np.abs(
                    expected[mismatch].astype(np.float64)
                    - sigma[mismatch].astype(np.float64)
                )
            )
        )
        raise ValueError(
            f"{context}: sigma 与冻结电导率表逐体素不一致，"
            f"mismatch={count}, max_abs={max_abs}"
        )


def validate_conductivity_arrays(
    labels: np.ndarray,
    sigma: np.ndarray,
    *,
    context: str = "conductivity",
) -> None:
    """Expose strict voxel-wise validation to the prepared-subject loader."""

    _validate_sigma_array(labels, np.asarray(sigma), context=context)


def _metadata_array(
    payload: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Read a finite numeric array with a fixed shape from metadata."""

    try:
        result = np.asarray(payload[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"metadata 缺少有效字段 {key!r}") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"metadata 字段 {key!r} 必须是有限 shape={shape} 数组")
    return result


def _metadata_shape(payload: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    """Read a 3D shape from metadata."""

    raw = payload.get(key)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"metadata 字段 {key!r} 必须是三维 shape")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise TypeError(f"metadata 字段 {key!r} 必须只含整数")
    return tuple(raw)


def validate_subject_conductivity(
    subject_dir: Path,
    *,
    metadata_path: Path | None = None,
    affine_atol: float = 1e-5,
) -> ConductivityValidation:
    """Validate tissues, sigma, spatial metadata, and the frozen table read-only."""

    import nibabel as nib

    root = Path(subject_dir)
    tissue_path = root / "images" / "final_tissues.nii.gz"
    sigma_path = root / "images" / "sigma_1mm.nii.gz"
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else root / "images" / "sigma_1mm.json"
    )
    for role, path in (
        ("final_tissues", tissue_path),
        ("sigma", sigma_path),
        ("sigma metadata", resolved_metadata_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {role}: {path}")

    try:
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"sigma metadata 不是有效 JSON: {resolved_metadata_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise TypeError("sigma metadata 顶层必须是 JSON 对象")
    if metadata.get("schema_id") != CONDUCTIVITY_METADATA_SCHEMA_ID:
        raise ValueError(
            f"sigma metadata schema_id 必须为 {CONDUCTIVITY_METADATA_SCHEMA_ID!r}"
        )
    validate_conductivity_table(metadata.get("conductivity_table", {}))

    tissue_image = nib.load(tissue_path)
    sigma_image = nib.load(sigma_path)
    labels = _load_3d_data(tissue_image)
    sigma = _load_3d_data(sigma_image)
    shape = tuple(int(value) for value in labels.shape)
    sigma_shape = tuple(int(value) for value in sigma.shape)
    if sigma_shape != shape:
        raise ValueError(f"tissue/sigma shape 不一致：{shape} vs {sigma_shape}")
    if not np.allclose(
        tissue_image.affine,
        sigma_image.affine,
        rtol=0.0,
        atol=affine_atol,
    ):
        raise ValueError("tissue/sigma affine 不一致")
    if np.dtype(sigma_image.get_data_dtype()) != np.dtype(np.float32):
        raise ValueError(
            "sigma NIfTI header dtype 必须为 float32，"
            f"实际为 {sigma_image.get_data_dtype()}"
        )

    source_metadata = metadata.get("source_final_tissues")
    output_metadata = metadata.get("output_sigma")
    if not isinstance(source_metadata, Mapping) or not isinstance(
        output_metadata, Mapping
    ):
        raise ValueError("sigma metadata 缺少 source_final_tissues/output_sigma 对象")
    source_hash = str(source_metadata.get("sha256", ""))
    sigma_hash = str(output_metadata.get("sha256", ""))
    if _metadata_shape(source_metadata, "shape") != shape:
        raise ValueError("final_tissues 当前 shape 与 sigma metadata 不一致")
    if _metadata_shape(output_metadata, "shape") != shape:
        raise ValueError("sigma 当前 shape 与 sigma metadata 不一致")
    source_affine = _metadata_array(
        source_metadata,
        "affine_ijk_to_ras_mm",
        shape=(4, 4),
    )
    output_affine = _metadata_array(
        output_metadata,
        "affine_ijk_to_ras_mm",
        shape=(4, 4),
    )
    if not np.allclose(
        source_affine,
        tissue_image.affine,
        rtol=0.0,
        atol=affine_atol,
    ):
        raise ValueError("final_tissues 当前 affine 与 sigma metadata 不一致")
    if not np.allclose(
        output_affine,
        sigma_image.affine,
        rtol=0.0,
        atol=affine_atol,
    ):
        raise ValueError("sigma 当前 affine 与 sigma metadata 不一致")
    if output_metadata.get("dtype") != "float32":
        raise ValueError("sigma metadata dtype 必须为 'float32'")
    if output_metadata.get("unit") != CONDUCTIVITY_UNIT:
        raise ValueError(f"sigma metadata unit 必须为 {CONDUCTIVITY_UNIT!r}")

    transform = metadata.get("model_input_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("sigma metadata 缺少 model_input_transform")
    if transform.get("id") != MODEL_SIGMA_TRANSFORM_ID:
        raise ValueError("sigma metadata 模型变换 ID 不一致")
    if transform.get("divisor_s_per_m") != MODEL_SIGMA_DIVISOR_S_PER_M:
        raise ValueError("sigma metadata 模型变换除数不一致")
    if transform.get("output_dtype") != "float32":
        raise ValueError("sigma metadata 模型变换输出 dtype 必须为 float32")

    _validate_sigma_array(labels, sigma, context=str(root))
    # Also apply the shared model transform to reject values that cannot be normalized.
    normalized = normalize_conductivity_for_model(sigma)
    if normalized.dtype != np.float32 or not np.all(np.isfinite(normalized)):
        raise RuntimeError("内部错误：v1 sigma 模型变换未产生有限 float32")
    return ConductivityValidation(
        subject_dir=root,
        tissue_path=tissue_path,
        sigma_path=sigma_path,
        metadata_path=resolved_metadata_path,
        shape=shape,
        affine=np.asarray(tissue_image.affine, dtype=np.float64),
        source_tissues_sha256=source_hash,
        sigma_sha256=sigma_hash,
    )


def _is_within(path: Path, parent: Path) -> bool:
    """Check whether a resolved path lies within the specified directory."""

    return path == parent or parent in path.parents


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a same-directory temporary file and no-replace publication."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_subject_conductivity(
    source_subject_dir: Path,
    staging_subject_dir: Path,
) -> Path:
    """Generate sigma and provenance metadata in independent staging from read-only sources.

    ``staging_subject_dir`` must not equal or lie within the source-subject
    directory, and destination files must not already exist. This function
    neither copies nor modifies ``final_tissues.nii.gz``.
    """

    import nibabel as nib

    source_root = Path(source_subject_dir)
    staging_root = Path(staging_subject_dir)
    resolved_source = source_root.resolve(strict=True)
    resolved_staging = staging_root.resolve(strict=False)
    if _is_within(resolved_staging, resolved_source):
        raise ValueError("staging_subject_dir 必须位于来源被试目录之外")

    tissue_path = source_root / "images" / "final_tissues.nii.gz"
    if not tissue_path.is_file():
        raise FileNotFoundError(f"缺少 final_tissues: {tissue_path}")

    images_dir = staging_root / "images"
    # Reject existing symlinks that could redirect independent staging writes into the source.
    resolved_images = images_dir.resolve(strict=False)
    if _is_within(resolved_images, resolved_source):
        raise ValueError("staging images 解析后位于来源被试目录内")
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = images_dir / "sigma_1mm.nii.gz"
    metadata_path = images_dir / "sigma_1mm.json"
    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "拒绝覆盖 staging 中已有电导率资产: "
            + ", ".join(str(path) for path in existing)
        )

    source_stat_before = tissue_path.stat()
    tissue_image = nib.load(tissue_path)
    labels = _load_3d_data(tissue_image)
    sigma = labels_to_conductivity(labels)
    affine = np.asarray(tissue_image.affine, dtype=np.float64)
    header = tissue_image.header.copy()
    header.set_data_dtype(np.float32)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".sigma_1mm.",
        suffix=".nii.gz",
        dir=images_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        nib.save(nib.Nifti1Image(sigma, affine, header), temporary_path)
        descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    shape = [int(value) for value in sigma.shape]
    metadata = {
        "schema_id": CONDUCTIVITY_METADATA_SCHEMA_ID,
        "source_final_tissues": {
            "relative_path": "images/final_tissues.nii.gz",
            "shape": shape,
            "affine_ijk_to_ras_mm": affine.tolist(),
        },
        "output_sigma": {
            "relative_path": "images/sigma_1mm.nii.gz",
            "shape": shape,
            "affine_ijk_to_ras_mm": affine.tolist(),
            "dtype": "float32",
            "unit": CONDUCTIVITY_UNIT,
        },
        "conductivity_table": conductivity_table_metadata(),
        "model_input_transform": {
            "id": MODEL_SIGMA_TRANSFORM_ID,
            "divisor_s_per_m": MODEL_SIGMA_DIVISOR_S_PER_M,
            "output_dtype": "float32",
        },
        "tissue_labels_present": [
            int(value) for value in np.unique(_strict_tissue_labels(labels))
        ],
    }
    _write_json_atomic(metadata_path, metadata)
    source_stat_after = tissue_path.stat()
    if (
        source_stat_after.st_size != source_stat_before.st_size
        or source_stat_after.st_mtime_ns != source_stat_before.st_mtime_ns
    ):
        raise RuntimeError("生成电导率期间 final_tissues 来源发生变化")
    return output_path


# 导入时核对完整电导率数值与单位。


__all__ = [
    "CONDUCTIVITY_METADATA_SCHEMA_ID",
    "CONDUCTIVITY_TABLE_ID",
    "CONDUCTIVITY_UNIT",
    "CONDUCTIVITY_VALUES",
    "MODEL_SIGMA_DIVISOR_S_PER_M",
    "MODEL_SIGMA_TRANSFORM_ID",
    "SIMNIBS_SCALAR_CONDUCTIVITY",
    "TISSUE_NAMES",
    "ConductivityValidation",
    "conductivity_table_metadata",
    "labels_to_conductivity",
    "normalize_conductivity_for_model",
    "sha256_file",
    "validate_conductivity_arrays",
    "validate_conductivity_table",
    "validate_subject_conductivity",
    "write_subject_conductivity",
]
