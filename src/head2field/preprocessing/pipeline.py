"""Explicit T1 v1 preprocessing profiles and transactional publication."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from . import alignment as rigid_alignment
from .profiles import (
    HCP_ACPC_PROFILE,
    NON_HCP_RAW_PROFILE,
    T1_PREPROCESSING_PROFILES,
)

MANIFEST_NAME = "preprocessing_v1.json"
CANONICAL_T1_RELATIVE_PATH = Path("images/T1_1mm.nii.gz")
RIGID_TRANSFORM_RELATIVE_PATH = Path("transforms/rigid_alignment_transform.json")
HCP_QC_RELATIVE_PATH = Path("qc/canonical_t1_qc.png")
RIGID_QC_RELATIVE_PATH = Path("qc/rigid_alignment_qc.png")


@dataclass(frozen=True)
class _FileFingerprint:
    """Store pre-processing identities of read-only inputs for checks before publication."""

    path: Path
    size_bytes: int
    mtime_ns: int


def _fingerprint(path: Path, role: str) -> _FileFingerprint:
    """Read file identities without storing absolute paths in artifacts."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} 不存在或不是普通文件：{path}")
    stat = resolved.stat()
    return _FileFingerprint(
        path=resolved,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def _assert_unchanged(fingerprint: _FileFingerprint, role: str) -> None:
    """Confirm processing did not modify or replace read-only inputs."""

    if not fingerprint.path.is_file():
        raise RuntimeError(f"{role} 在处理期间消失。")
    stat = fingerprint.path.stat()
    if (
        int(stat.st_size) != fingerprint.size_bytes
        or int(stat.st_mtime_ns) != fingerprint.mtime_ns
    ):
        raise RuntimeError(f"{role} 在处理期间发生变化，拒绝发布输出。")


def _source_record(
    fingerprint: _FileFingerprint,
    description: str,
) -> dict[str, Any]:
    """建立脱敏输入记录，保留用途和文件大小。"""

    return {
        "description": description,
        "path_policy": "absolute_path_omitted",
        "size_bytes": fingerprint.size_bytes,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON in stable UTF-8 format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _existing_profile(output_dir: Path) -> str | None:
    """Read the existing profile without trusting other contents of an old directory."""

    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    profile = payload.get("profile")
    return profile if isinstance(profile, str) else None


def _reject_existing_output(output_dir: Path, requested_profile: str) -> None:
    """Refuse overwrites, repeated processing, and profile changes at one destination."""

    if not os.path.lexists(output_dir):
        return
    previous_profile = _existing_profile(output_dir) if output_dir.is_dir() else None
    if previous_profile == requested_profile:
        raise FileExistsError(
            f"输出目录已由 profile={requested_profile} 完成，拒绝重复处理：{output_dir}"
        )
    if previous_profile is not None:
        raise FileExistsError(
            "输出目录已有冲突 profile，拒绝覆盖："
            f"existing={previous_profile}, requested={requested_profile}, path={output_dir}"
        )
    raise FileExistsError(f"输出目标已经存在，拒绝覆盖：{output_dir}")


def _validate_profile(profile: str, template: Path | None) -> None:
    """Require explicit profiles and mutually consistent parameters."""

    if profile not in T1_PREPROCESSING_PROFILES:
        raise ValueError(
            f"未知或缺失的 T1 profile={profile!r}；必须显式选择 {T1_PREPROCESSING_PROFILES}"
        )
    if profile == HCP_ACPC_PROFILE and template is not None:
        raise ValueError("hcp_acpc 已完成 ACPC 对齐，不得提供刚体姿态模板。")
    if profile == NON_HCP_RAW_PROFILE and template is None:
        raise ValueError("non_hcp_raw 必须提供 --template 以执行一次 6-DOF 刚体摆正。")


def _canonical_grid_center() -> np.ndarray:
    """Compute world coordinates of the fixed output-grid center."""

    center_ijk = (
        np.asarray(rigid_alignment.TARGET_SHAPE, dtype=np.float64) - 1.0
    ) / 2.0
    return nib.affines.apply_affine(rigid_alignment.TARGET_AFFINE, center_ijk)


def _hcp_centering_transform(
    source_image: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate full-head centering translation without invoking rigid optimization."""

    source_mask = rigid_alignment.head_mask(source_image)
    source_center = rigid_alignment.world_center(source_mask, source_image.affine)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = _canonical_grid_center() - source_center
    return rigid_alignment.validate_rigid_transform(transform), source_center


def _reject_already_canonical_input(
    source_image: nib.spatialimages.SpatialImage,
) -> None:
    """Reject v1-canonicalized T1 inputs submitted again through the raw-image entry point."""

    expected_shape = tuple(int(value) for value in rigid_alignment.TARGET_SHAPE)
    if source_image.shape != expected_shape:
        return
    if np.array_equal(
        np.asarray(source_image.affine, dtype=np.float64),
        rigid_alignment.TARGET_AFFINE,
    ):
        raise ValueError("输入已经满足 v1 canonical T1 网格，拒绝重复预处理。")


def _save_canonical_t1(
    path: Path,
    source_image: nib.Nifti1Image,
    canonical_data: np.ndarray,
) -> nib.Nifti1Image:
    """Save fixed float32 canonical T1 and return the reloaded image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = source_image.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(
        np.asarray(canonical_data, dtype=np.float32),
        rigid_alignment.TARGET_AFFINE,
        header=header,
    )
    output_image.header.set_zooms((1.0, 1.0, 1.0))
    output_image.set_qform(rigid_alignment.TARGET_AFFINE, code=1)
    output_image.set_sform(rigid_alignment.TARGET_AFFINE, code=1)
    nib.save(output_image, str(path))
    return nib.load(str(path))


def _validate_canonical_t1(
    path: Path,
    minimum_margin_voxels: int,
) -> dict[str, Any]:
    """Validate the canonical grid, numeric values, and full foreground bounds."""

    if minimum_margin_voxels < 0:
        raise ValueError("minimum_margin_voxels 不能为负数。")
    image = nib.load(str(path))
    expected_shape = tuple(int(value) for value in rigid_alignment.TARGET_SHAPE)
    if image.shape != expected_shape:
        raise RuntimeError(
            f"canonical T1 shape 错误：{image.shape} != {expected_shape}"
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.array_equal(affine, rigid_alignment.TARGET_AFFINE):
        raise RuntimeError("canonical T1 affine 与固定 v1 affine 不一致。")
    if tuple(nib.aff2axcodes(affine)) != ("R", "A", "S"):
        raise RuntimeError("canonical T1 orientation 必须为 RAS。")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if not np.allclose(zooms, (1.0, 1.0, 1.0), rtol=0.0, atol=1e-7):
        raise RuntimeError(f"canonical T1 spacing 必须为 1 mm，实际为 {zooms}。")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise RuntimeError("canonical T1 含有非有限值。")
    foreground = rigid_alignment.head_mask(image)
    indices = np.argwhere(foreground)
    if indices.size == 0:
        raise RuntimeError("canonical T1 没有可识别的头部前景。")
    lower = indices.min(axis=0)
    upper = indices.max(axis=0)
    margins = np.stack([lower, np.asarray(expected_shape) - 1 - upper])
    if np.any(margins < minimum_margin_voxels):
        raise RuntimeError(
            "canonical T1 前景触及安全边界："
            f"minimum={minimum_margin_voxels}, margins={margins.tolist()}"
        )
    stat = path.stat()
    return {
        "path": CANONICAL_T1_RELATIVE_PATH.as_posix(),
        "size_bytes": int(stat.st_size),
        "shape": list(expected_shape),
        "dtype": str(image.get_data_dtype()),
        "spacing_mm": list(zooms),
        "orientation": "RAS",
        "affine": affine.tolist(),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "foreground_bbox_ijk": [lower.tolist(), upper.tolist()],
        "foreground_margins_voxels": margins.tolist(),
    }


def _qc_record(path: Path, relative_path: Path) -> dict[str, Any]:
    """Record QC artifact identities and retain the production manual-review requirement."""

    stat = path.stat()
    return {
        "path": relative_path.as_posix(),
        "size_bytes": int(stat.st_size),
        "panel_order_left_to_right": ["sagittal", "coronal", "axial"],
        "manual_review_required": True,
    }


def _transform_payload(
    transform: np.ndarray,
    source_center: np.ndarray,
    result: Any,
    raw_fingerprint: _FileFingerprint,
    template_fingerprint: _FileFingerprint,
    max_samples: int,
) -> dict[str, Any]:
    """Build a rigid-transform record without machine-specific absolute paths."""

    transform = rigid_alignment.validate_rigid_transform(transform)
    parameters = rigid_alignment.validate_optimizer_result(result, level_step=0)
    rotation = transform[:3, :3]
    return {
        "schema_id": "rigid_t1_alignment_v1",
        "degrees_of_freedom": 6,
        "scaling_allowed": False,
        "input_t1": _source_record(raw_fingerprint, "显式提供的非 HCP 原始 T1"),
        "orientation_template": _source_record(
            template_fingerprint,
            "显式提供的刚体姿态参考 T1",
        ),
        "source_world_to_canonical_world": transform.tolist(),
        "optimizer_parameters_xyz_deg_then_mm": parameters.tolist(),
        "optimizer_success": True,
        "optimizer_message": str(result.message),
        "final_negative_mutual_information": float(result.fun),
        "maximum_samples_per_level": int(max_samples),
        "source_head_bbox_center_mm": np.asarray(
            source_center, dtype=np.float64
        ).tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "rotation_orthogonality_max_abs_error": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        ),
    }


def _rename_noreplace(source: Path, target: Path) -> None:
    """Use atomic no-replace rename on Linux and a conservative fallback elsewhere."""

    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(target),
                1,
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"发布目标已出现，拒绝覆盖：{target}")
            if error_number not in (errno.ENOSYS, errno.EINVAL):
                raise OSError(error_number, os.strerror(error_number), str(target))

    # Recheck immediately before fallback rename; Linux with renameat2 does not use this path.
    if os.path.lexists(target):
        raise FileExistsError(f"发布目标已出现，拒绝覆盖：{target}")
    os.rename(source, target)


def preprocess_t1(
    *,
    input_t1: Path,
    output_dir: Path,
    profile: str,
    template: Path | None = None,
    max_samples: int = 60_000,
    minimum_margin_voxels: int = 12,
) -> dict[str, Any]:
    """Preprocess one T1 with an explicit profile and publish the complete directory on success."""

    input_t1 = Path(input_t1)
    requested_output_dir = Path(output_dir).expanduser()
    template = None if template is None else Path(template)
    _validate_profile(profile, template)
    if max_samples <= 0:
        raise ValueError("max_samples 必须为正数。")
    _reject_existing_output(requested_output_dir, profile)
    output_dir = requested_output_dir.parent.resolve() / requested_output_dir.name
    _reject_existing_output(output_dir, profile)

    raw_fingerprint = _fingerprint(input_t1, "原始 T1")
    template_fingerprint = (
        _fingerprint(template, "刚体姿态模板") if template is not None else None
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.preprocessing-v1-staging-",
            dir=output_dir.parent,
        )
    )
    published = False
    try:
        source_image = nib.load(str(raw_fingerprint.path))
        _reject_already_canonical_input(source_image)
        source = rigid_alignment.load_3d(source_image)
        rigid_performed = profile == NON_HCP_RAW_PROFILE
        transform_relative_path: Path | None = None

        if profile == HCP_ACPC_PROFILE:
            transform, source_center = _hcp_centering_transform(source_image)
            optimizer_result = None
            template_image = None
            canonical_data = rigid_alignment.resample_to_canonical(
                source_image,
                source,
                transform,
            )
            qc_relative_path = HCP_QC_RELATIVE_PATH
        else:
            assert template_fingerprint is not None
            template_image = nib.load(str(template_fingerprint.path))
            template_data = rigid_alignment.load_3d(template_image)
            # This is the only rigid-estimation call for the non_hcp_raw profile.
            transform, optimizer_result, source_center = rigid_alignment.estimate_rigid(
                source_image,
                source,
                template_image,
                template_data,
                max_samples,
            )
            transform = rigid_alignment.validate_rigid_transform(transform)
            rigid_alignment.validate_optimizer_result(optimizer_result, level_step=0)
            canonical_data = rigid_alignment.resample_to_canonical(
                source_image,
                source,
                transform,
            )
            transform_relative_path = RIGID_TRANSFORM_RELATIVE_PATH
            qc_relative_path = RIGID_QC_RELATIVE_PATH

        canonical_path = staging / CANONICAL_T1_RELATIVE_PATH
        _save_canonical_t1(canonical_path, source_image, canonical_data)
        canonical_record = _validate_canonical_t1(
            canonical_path,
            minimum_margin_voxels=minimum_margin_voxels,
        )

        qc_path = staging / qc_relative_path
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        rigid_alignment.save_qc(
            qc_path,
            canonical_data,
            "T1 v1 canonical QC",
        )
        qc_record = _qc_record(qc_path, qc_relative_path)

        transform_record: dict[str, Any] | None = None
        transform_artifact: dict[str, Any] | None = None
        if rigid_performed:
            assert optimizer_result is not None
            assert template_fingerprint is not None
            assert transform_relative_path is not None
            transform_record = _transform_payload(
                transform,
                source_center,
                optimizer_result,
                raw_fingerprint,
                template_fingerprint,
                max_samples,
            )
            transform_path = staging / transform_relative_path
            _write_json(transform_path, transform_record)
            transform_stat = transform_path.stat()
            transform_artifact = {
                "path": transform_relative_path.as_posix(),
                "size_bytes": int(transform_stat.st_size),
            }

        _assert_unchanged(raw_fingerprint, "原始 T1")
        if template_fingerprint is not None:
            _assert_unchanged(template_fingerprint, "刚体姿态模板")

        manifest: dict[str, Any] = {
            "schema_id": "t1_preprocessing_v1",
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "profile": profile,
            "input_t1": _source_record(
                raw_fingerprint,
                "显式提供的 HCP ACPC/畸变校正 T1"
                if profile == HCP_ACPC_PROFILE
                else "显式提供的非 HCP 原始 T1",
            ),
            "profile_contract": {
                "rigid_estimation_performed": rigid_performed,
                "rigid_estimation_call_count": 1 if rigid_performed else 0,
                "hcp_acpc_bypass": profile == HCP_ACPC_PROFILE,
            },
            "rigid_alignment": {
                "performed": rigid_performed,
                "degrees_of_freedom": 6 if rigid_performed else None,
                "source_world_to_canonical_world": transform.tolist(),
                "transform_path": (
                    transform_relative_path.as_posix()
                    if transform_relative_path is not None
                    else None
                ),
                "transform_artifact": transform_artifact,
                "bypass_reason": (
                    "HCP 输入已完成 ACPC 对齐和畸变校正，仅执行 canonical 居中"
                    if not rigid_performed
                    else None
                ),
            },
            "orientation_template": (
                _source_record(template_fingerprint, "显式提供的刚体姿态参考 T1")
                if template_fingerprint is not None
                else None
            ),
            "canonical_t1": canonical_record,
            "qc": qc_record,
            "publication": {
                "policy": "new_output_directory_only",
                "method": "sibling_staging_then_atomic_noreplace_rename",
                "partial_output_published_on_failure": False,
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        _reject_existing_output(output_dir, profile)
        _rename_noreplace(staging, output_dir)
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            # Clean only the unique unpublished staging directory created by this invocation.
            shutil.rmtree(staging)


__all__ = [
    "CANONICAL_T1_RELATIVE_PATH",
    "HCP_ACPC_PROFILE",
    "MANIFEST_NAME",
    "NON_HCP_RAW_PROFILE",
    "T1_PREPROCESSING_PROFILES",
    "preprocess_t1",
]
