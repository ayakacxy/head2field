"""Assemble prepared-subject v1 from canonical T1, external segmentation, and a fixed cap."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .charm_adapter import (
    SegmentationAdapter,
    SimNIBSCharmAdapter,
    validate_charm_runtime,
)
from .conductivity import (
    labels_to_conductivity,
)
from .electrodes import (
    load_standard_cap_from_head_model,
    load_standard_electrode_inputs,
)
from .pipeline import preprocess_t1
from .transaction import _path_entry_exists, staged_output_directory
from .validation import (
    PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    PREPARED_SUBJECT_MANIFEST_NAME,
    PREPARED_SUBJECT_SHAPE,
    PreparedSubjectV1,
    build_prepared_subject_manifest,
    load_prepared_subject_v1,
    validate_prepared_subject_manifest,
)


@dataclass(frozen=True)
class PreparationResult:
    """Identities of the three retained outputs after top-level preparation."""

    canonical_output_dir: Path
    head_model_output_dir: Path
    prepared_output_dir: Path
    prepared_subject_manifest_sha256: str


def _fingerprint(path: Path, role: str) -> tuple[Path, int, int]:
    """记录只读输入的路径、大小和修改时间。"""
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} 不是普通文件：{path}")
    stat = resolved.stat()
    return resolved, int(stat.st_size), int(stat.st_mtime_ns)


def _assert_unchanged(fingerprint: tuple[Path, int, int], role: str) -> None:
    """确认组装过程中输入文件的大小和修改时间未改变。"""
    path, size, mtime_ns = fingerprint
    if _fingerprint(path, role)[1:] != (size, mtime_ns):
        raise RuntimeError(f"{role} 在 prepared-subject assembly 期间发生变化")


def _load_3d(image: nib.spatialimages.SpatialImage, role: str) -> np.ndarray:
    """Read a 3D NIfTI, accepting trailing singleton dimensions from SimNIBS."""

    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{role} 必须为三维，实际 shape={data.shape}")
    return data


def _validate_canonical_pair(
    canonical_t1: Path,
    final_tissues: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Validate aligned T1/final_tissues images on the fixed 1 mm RAS grid."""

    t1_image = nib.load(str(canonical_t1))
    tissue_image = nib.load(str(final_tissues))
    t1 = _load_3d(t1_image, "canonical T1")
    tissues = _load_3d(tissue_image, "final_tissues")
    expected_shape = tuple(PREPARED_SUBJECT_SHAPE)
    if t1.shape != expected_shape or tissues.shape != expected_shape:
        raise ValueError(
            f"T1/final_tissues 必须为 {expected_shape}，实际为 {t1.shape}/{tissues.shape}"
        )
    expected_affine = np.asarray(
        PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
        dtype=np.float64,
    )
    for role, image in (("T1", t1_image), ("final_tissues", tissue_image)):
        affine = np.asarray(image.affine, dtype=np.float64)
        if not np.allclose(affine, expected_affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"{role} affine 与固定 canonical 网格不一致")
        if tuple(nib.aff2axcodes(affine)) != ("R", "A", "S"):
            raise ValueError(f"{role} orientation 必须为 RAS")
    if not np.isfinite(np.asarray(t1, dtype=np.float32)).all():
        raise ValueError("canonical T1 含 NaN 或 Inf")
    return tissue_image, tissues


def _write_json_new(path: Path, payload: Any) -> None:
    """Write a new JSON file, rejecting any existing directory entry."""

    if os.path.lexists(path):
        raise FileExistsError(f"输出已经存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_sigma(
    *,
    tissue_image: nib.spatialimages.SpatialImage,
    tissues: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate float32 sigma using the frozen 0--10 table for manifest provenance."""

    sigma = labels_to_conductivity(tissues)
    images_dir = output_dir / "images"
    sigma_path = images_dir / "sigma_1mm.nii.gz"
    header = tissue_image.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(sigma, tissue_image.affine, header=header)
    image.set_qform(tissue_image.affine, code=1)
    image.set_sform(tissue_image.affine, code=1)
    nib.save(image, str(sigma_path))


def assemble_prepared_subject(
    *,
    canonical_t1: Path,
    final_tissues: Path,
    output_dir: Path,
    head_model_dir: Path | None = None,
    electrode_names_json: Path | None = None,
    electrode_coords_mm_npy: Path | None = None,
) -> PreparedSubjectV1:
    """Transactionally assemble and fully validate a new prepared-subject v1 directory."""

    if head_model_dir is None:
        raise ValueError("prepared-subject assembly 必须绑定本次 CHARM head_model_dir")
    if (electrode_names_json is None) != (electrode_coords_mm_npy is None):
        raise ValueError("显式电极 override 必须同时提供 names JSON 与 coords NPY")
    inputs: dict[str, tuple[Path, int, int, str]] = {
        "canonical T1": _fingerprint(canonical_t1, "canonical T1"),
        "final_tissues": _fingerprint(final_tissues, "final_tissues"),
    }
    t1_path = inputs["canonical T1"][0]
    tissue_path = inputs["final_tissues"][0]
    tissue_image, tissues = _validate_canonical_pair(t1_path, tissue_path)
    charm_identity = validate_charm_runtime(
        head_model_dir,
        canonical_t1=t1_path,
        final_tissues=tissue_path,
    )
    inputs["CHARM runtime manifest"] = _fingerprint(
        charm_identity.manifest_path,
        "CHARM runtime manifest",
    )
    inputs["CHARM T1"] = _fingerprint(
        charm_identity.charm_t1_path,
        "CHARM T1",
    )
    inputs["CHARM head mesh"] = _fingerprint(
        charm_identity.head_mesh_path,
        "CHARM head mesh",
    )
    if electrode_names_json is not None and electrode_coords_mm_npy is not None:
        inputs["electrode names"] = _fingerprint(
            electrode_names_json,
            "electrode names",
        )
        inputs["electrode coordinates"] = _fingerprint(
            electrode_coords_mm_npy,
            "electrode coordinates",
        )
        names, coordinates = load_standard_electrode_inputs(
            inputs["electrode names"][0],
            inputs["electrode coordinates"][0],
        )
    else:
        names, coordinates, cap_path = load_standard_cap_from_head_model(head_model_dir)
        inputs["CHARM standard cap"] = _fingerprint(
            cap_path,
            "CHARM standard cap",
        )
    query_ijk = np.argwhere(np.isin(tissues, (1, 2))).astype(np.int64, copy=False)
    if len(query_ijk) == 0:
        raise ValueError("final_tissues 中没有 GM/WM 查询点")

    destination = Path(output_dir).expanduser()
    with staged_output_directory(destination) as staging:
        images = staging / "images"
        voxels = staging / "voxels"
        electrodes = staging / "electrodes"
        images.mkdir()
        voxels.mkdir()
        electrodes.mkdir()
        shutil.copyfile(t1_path, images / "T1_1mm.nii.gz")
        shutil.copyfile(tissue_path, images / "final_tissues.nii.gz")
        _write_sigma(
            tissue_image=tissue_image,
            tissues=tissues,
            output_dir=staging,
        )
        np.save(voxels / "gmwm_ijk.npy", query_ijk, allow_pickle=False)
        _write_json_new(
            electrodes / "electrode_names_all.json",
            list(names),
        )
        _write_json_new(
            electrodes / "electrode_names_no_oz.json",
            list(names[1:]),
        )
        np.save(
            electrodes / "electrode_coords_mm.npy",
            coordinates,
            allow_pickle=False,
        )
        manifest = build_prepared_subject_manifest(
            staging,
            head_model_provenance=charm_identity.to_prepared_provenance(),
        )
        _write_json_new(staging / PREPARED_SUBJECT_MANIFEST_NAME, manifest)
        validate_prepared_subject_manifest(
            staging,
            manifest=manifest,
        )
        for role, fingerprint in inputs.items():
            _assert_unchanged(fingerprint, role)
    return load_prepared_subject_v1(
        destination,
    )


def _reject_overlapping_outputs(paths: tuple[Path, ...]) -> None:
    """Reject identical or nested stage outputs so promotion cannot affect another stage."""

    resolved = [path.expanduser().resolve(strict=False) for path in paths]
    for index, left in enumerate(resolved):
        if _path_entry_exists(left):
            raise FileExistsError(f"prepare 输出已经存在，拒绝覆盖：{left}")
        for right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "canonical、head-model 和 prepared 输出不得相同或互相嵌套"
                )


def prepare_subject(
    *,
    input_t1: Path,
    profile: str,
    canonical_output_dir: Path,
    head_model_output_dir: Path,
    prepared_output_dir: Path,
    electrode_names_json: Path | None = None,
    electrode_coords_mm_npy: Path | None = None,
    template: Path | None = None,
    segmentation_adapter: SegmentationAdapter | None = None,
    max_samples: int = 60_000,
    minimum_margin_voxels: int = 12,
) -> PreparationResult:
    """Orchestrate canonicalization, external segmentation, conductivity, and prepared assembly."""

    canonical_dir = Path(canonical_output_dir)
    head_model_dir = Path(head_model_output_dir)
    prepared_dir = Path(prepared_output_dir)
    _reject_overlapping_outputs((canonical_dir, head_model_dir, prepared_dir))
    if (electrode_names_json is None) != (electrode_coords_mm_npy is None):
        raise ValueError("显式电极 override 必须同时提供 names JSON 与 coords NPY")
    if electrode_names_json is not None and electrode_coords_mm_npy is not None:
        load_standard_electrode_inputs(
            electrode_names_json,
            electrode_coords_mm_npy,
        )
    preprocess_t1(
        input_t1=input_t1,
        output_dir=canonical_dir,
        profile=profile,
        template=template,
        max_samples=max_samples,
        minimum_margin_voxels=minimum_margin_voxels,
    )
    canonical_t1 = canonical_dir / "images" / "T1_1mm.nii.gz"
    adapter = segmentation_adapter or SimNIBSCharmAdapter()
    final_tissues = adapter.run(
        canonical_t1=canonical_t1,
        output_dir=head_model_dir,
    )
    prepared = assemble_prepared_subject(
        canonical_t1=canonical_t1,
        final_tissues=final_tissues,
        output_dir=prepared_dir,
        head_model_dir=head_model_dir,
        electrode_names_json=electrode_names_json,
        electrode_coords_mm_npy=electrode_coords_mm_npy,
    )
    return PreparationResult(
        canonical_output_dir=canonical_dir,
        head_model_output_dir=head_model_dir,
        prepared_output_dir=prepared_dir,
        prepared_subject_manifest_sha256=prepared.manifest_sha256,
    )


__all__ = [
    "PreparationResult",
    "assemble_prepared_subject",
    "prepare_subject",
]
