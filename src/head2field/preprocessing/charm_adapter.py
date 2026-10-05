"""Explicit preprocessing adapter for the user's external SimNIBS/CHARM runtime."""

from __future__ import annotations

from head2field.metadata import scientific_metadata, is_digest_key

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .conductivity import sha256_file
from .transaction import _path_entry_exists, _rename_noreplace

REQUIRED_SIMNIBS_VERSION = "4.6.0"
CHARM_RUNTIME_SCHEMA_ID = "charm_runtime_v1"
CHARM_RUNTIME_MANIFEST_NAME = "charm_runtime_v1.json"


class SegmentationAdapterError(RuntimeError):
    """Indicate missing segmentation runtime or outputs that violate the contract."""


@dataclass(frozen=True)
class CharmRuntimeIdentity:
    """已验证路径、组织空间信息和运行版本的 CHARM 记录。"""

    head_model_dir: Path
    manifest_path: Path
    manifest_sha256: str
    canonical_t1_sha256: str
    charm_t1_path: Path
    charm_t1_sha256: str
    final_tissues_path: Path
    final_tissues_sha256: str
    head_mesh_path: Path
    head_mesh_sha256: str
    simnibs_version: str

    def to_prepared_provenance(self) -> dict[str, Any]:
        """Return a machine-path-free binding record embeddable in prepared-subject metadata."""

        return {
            "schema_id": "charm_runtime_binding_v1",
            "simnibs_version": self.simnibs_version,
        }


@runtime_checkable
class SegmentationAdapter(Protocol):
    """Explicit injectable segmentation-adapter contract for top-level preparation."""

    def run(self, *, canonical_t1: Path, output_dir: Path) -> Path:
        """Generate a new directory and return its ``final_tissues.nii.gz``."""


def installed_simnibs_version() -> str:
    """Return the user environment's SimNIBS version, failing explicitly if unavailable."""

    try:
        return metadata.version("simnibs")
    except metadata.PackageNotFoundError as exc:
        raise SegmentationAdapterError(
            "preprocessing 需要用户在当前环境中安装外部 SimNIBS 4.6.0。"
        ) from exc


def _fingerprint(path: Path) -> tuple[int, int]:
    """记录只读输入的大小和修改时间。"""

    if not path.is_file():
        raise FileNotFoundError(f"输入不是普通文件：{path}")
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _exact_keys(payload: Mapping[str, Any], expected: set[str], role: str) -> None:
    """Require JSON object fields to match the frozen set exactly."""

    actual = {key for key in payload if not is_digest_key(key)}
    expected = {key for key in expected if not is_digest_key(key)}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SegmentationAdapterError(
            f"{role} 字段集合不一致：missing={missing}, extra={extra}"
        )


def _internal_file_record(path: Path, *, root: Path) -> dict[str, Any]:
    """Build safe relative provenance for regular files inside the head model."""

    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if path.is_symlink() or not resolved_path.is_file():
        raise SegmentationAdapterError(f"CHARM 产物必须是普通文件：{path}")
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SegmentationAdapterError(f"CHARM 产物逃逸 head-model：{path}") from exc
    stat = resolved_path.stat()
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": int(stat.st_size),
    }


def _external_file_record(path: Path) -> dict[str, Any]:
    """Build deidentified provenance for canonical T1 outside the head model."""

    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise SegmentationAdapterError(f"canonical T1 必须是普通文件：{path}")
    stat = resolved.stat()
    return {
        "path_policy": "absolute_path_omitted",
        "size_bytes": int(stat.st_size),
    }


def _single_head_mesh(head_model_dir: Path) -> Path:
    """Require exactly one official head mesh in the CHARM output root."""

    meshes = sorted(path for path in head_model_dir.glob("*.msh") if path.is_file())
    if len(meshes) != 1:
        raise SegmentationAdapterError(
            f"CHARM head-model 根目录必须恰有一个 .msh，实际为 {len(meshes)} 个"
        )
    return meshes[0]


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Exclusively write a new run manifest in a stable format."""

    if os.path.lexists(path):
        raise FileExistsError(f"CHARM runtime manifest 已存在：{path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_charm_runtime_manifest(
    *,
    canonical_t1: Path,
    head_model_dir: Path,
    simnibs_version: str,
) -> Path:
    """Write a complete deidentified run identity for finished CHARM outputs."""

    if simnibs_version != REQUIRED_SIMNIBS_VERSION:
        raise SegmentationAdapterError(
            f"CHARM runtime 要求 SimNIBS {REQUIRED_SIMNIBS_VERSION}，"
            f"实际为 {simnibs_version}"
        )
    root = Path(head_model_dir).resolve(strict=True)
    charm_t1 = root / "T1.nii.gz"
    final_tissues = root / "final_tissues.nii.gz"
    head_mesh = _single_head_mesh(root)
    payload = {
        "schema_id": CHARM_RUNTIME_SCHEMA_ID,
        "schema_version": 1,
        "adapter": "external-simnibs-charm",
        "simnibs_version": simnibs_version,
        "atlas_policy": "use_external_installation_resources",
        "redistributed_runtime_or_atlas": False,
        "canonical_t1": _external_file_record(Path(canonical_t1)),
        "charm_t1": _internal_file_record(charm_t1, root=root),
        "final_tissues": _internal_file_record(final_tissues, root=root),
        "head_mesh": _internal_file_record(head_mesh, root=root),
        "publication": {
            "policy": "new_output_directory_only",
            "method": "sibling_staging_then_atomic_noreplace_rename",
            "partial_output_published_on_failure": False,
        },
    }
    manifest_path = root / CHARM_RUNTIME_MANIFEST_NAME
    _write_json_new(manifest_path, payload)
    return manifest_path


def _validated_internal_record(
    *,
    root: Path,
    record: Mapping[str, Any],
    role: str,
) -> Path:
    """Strictly reload and validate a file record inside the head model."""

    _exact_keys(record, {"relative_path", "size_bytes", "sha256"}, role)
    relative = Path(str(record["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise SegmentationAdapterError(f"{role} relative_path 不安全")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SegmentationAdapterError(f"{role} 文件不存在或不是普通文件")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise SegmentationAdapterError(f"{role} 路径逃逸 head-model") from exc
    size = record["size_bytes"]
    record.get("sha256", "")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SegmentationAdapterError(f"{role} size/SHA256 格式无效")
    if path.stat().st_size != size:
        raise SegmentationAdapterError(f"{role} size/SHA256 与运行 manifest 不一致")
    return path


def validate_charm_runtime(
    head_model_dir: Path,
    *,
    canonical_t1: Path | None = None,
    final_tissues: Path | None = None,
    head_mesh: Path | None = None,
) -> CharmRuntimeIdentity:
    """Audit the CHARM manifest and its T1, tissue, and mesh identity chain read-only."""

    root = Path(head_model_dir).expanduser().resolve(strict=True)
    manifest_path = root / CHARM_RUNTIME_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SegmentationAdapterError("CHARM runtime manifest 缺失或不是普通文件")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SegmentationAdapterError("CHARM runtime manifest 不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise SegmentationAdapterError("CHARM runtime manifest 顶层必须是 JSON 对象")
    _exact_keys(
        payload,
        {
            "schema_id",
            "schema_version",
            "adapter",
            "simnibs_version",
            "atlas_policy",
            "redistributed_runtime_or_atlas",
            "canonical_t1",
            "charm_t1",
            "final_tissues",
            "head_mesh",
            "publication",
        },
        "CHARM runtime manifest",
    )
    exact = {
        "schema_id": CHARM_RUNTIME_SCHEMA_ID,
        "schema_version": 1,
        "adapter": "external-simnibs-charm",
        "simnibs_version": REQUIRED_SIMNIBS_VERSION,
        "atlas_policy": "use_external_installation_resources",
        "redistributed_runtime_or_atlas": False,
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in exact.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise SegmentationAdapterError(f"CHARM runtime 固定合同不匹配：{mismatches}")

    raw_canonical = payload["canonical_t1"]
    if not isinstance(raw_canonical, Mapping):
        raise SegmentationAdapterError("canonical_t1 provenance 必须是对象")
    _exact_keys(
        raw_canonical,
        {"path_policy", "size_bytes", "sha256"},
        "canonical_t1 provenance",
    )
    if raw_canonical["path_policy"] != "absolute_path_omitted":
        raise SegmentationAdapterError("canonical_t1 path_policy 不匹配")
    canonical_size = raw_canonical["size_bytes"]
    canonical_hash = raw_canonical.get("sha256", "")
    if (
        isinstance(canonical_size, bool)
        or not isinstance(canonical_size, int)
        or canonical_size <= 0
    ):
        raise SegmentationAdapterError("canonical_t1 size/SHA256 格式无效")
    if canonical_t1 is not None:
        canonical_record = _external_file_record(Path(canonical_t1))
        if scientific_metadata(canonical_record) != scientific_metadata(raw_canonical):
            raise SegmentationAdapterError("CHARM runtime 未绑定当前 canonical T1")

    raw_charm_t1 = payload["charm_t1"]
    raw_tissues = payload["final_tissues"]
    raw_mesh = payload["head_mesh"]
    if not all(
        isinstance(record, Mapping) for record in (raw_charm_t1, raw_tissues, raw_mesh)
    ):
        raise SegmentationAdapterError(
            "CHARM T1/final_tissues/head_mesh provenance 必须是对象"
        )
    charm_t1_path = _validated_internal_record(
        root=root,
        record=raw_charm_t1,
        role="CHARM T1",
    )
    tissues_path = _validated_internal_record(
        root=root,
        record=raw_tissues,
        role="CHARM final_tissues",
    )
    mesh_path = _validated_internal_record(
        root=root,
        record=raw_mesh,
        role="CHARM head_mesh",
    )
    if _single_head_mesh(root).resolve(strict=True) != mesh_path.resolve(strict=True):
        raise SegmentationAdapterError("CHARM runtime 未绑定根目录唯一 head mesh")
    if final_tissues is not None and tissues_path != Path(final_tissues).resolve(
        strict=True
    ):
        raise SegmentationAdapterError(
            "显式 final_tissues 与 CHARM runtime manifest 不一致"
        )
    if head_mesh is not None and mesh_path != Path(head_mesh).resolve(strict=True):
        raise SegmentationAdapterError(
            "显式 head_mesh 与 CHARM runtime manifest 不一致"
        )

    publication = payload["publication"]
    if not isinstance(publication, Mapping):
        raise SegmentationAdapterError("CHARM publication 必须是对象")
    expected_publication = {
        "policy": "new_output_directory_only",
        "method": "sibling_staging_then_atomic_noreplace_rename",
        "partial_output_published_on_failure": False,
    }
    _exact_keys(publication, set(expected_publication), "CHARM publication")
    if dict(publication) != expected_publication:
        raise SegmentationAdapterError("CHARM publication 固定合同不匹配")
    return CharmRuntimeIdentity(
        head_model_dir=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        canonical_t1_sha256=str(canonical_hash),
        charm_t1_path=charm_t1_path,
        charm_t1_sha256=str(raw_charm_t1.get("sha256", "")),
        final_tissues_path=tissues_path,
        final_tissues_sha256=str(raw_tissues.get("sha256", "")),
        head_mesh_path=mesh_path,
        head_mesh_sha256=str(raw_mesh.get("sha256", "")),
        simnibs_version=str(payload["simnibs_version"]),
    )


def validate_prepared_head_mesh_binding(
    *,
    prepared_subject_dir: Path,
    prepared_manifest: Mapping[str, Any],
    head_mesh: Path,
) -> CharmRuntimeIdentity:
    """Cross-check prepared T1/tissues, the CHARM manifest, and the supplied head mesh."""

    mesh = Path(head_mesh).expanduser().resolve(strict=True)
    identity = validate_charm_runtime(
        mesh.parent,
        canonical_t1=Path(prepared_subject_dir) / "images" / "T1_1mm.nii.gz",
        head_mesh=mesh,
    )
    prepared_tissues = (
        Path(prepared_subject_dir) / "images" / "final_tissues.nii.gz"
    ).resolve(strict=True)
    # 直接比较组织内容和空间信息，确保 prepared 文件确实来自本次头模型。
    import nibabel as nib

    source_image = nib.load(identity.final_tissues_path)
    prepared_image = nib.load(prepared_tissues)
    if not np.array_equal(
        source_image.affine, prepared_image.affine
    ) or not np.array_equal(
        np.asarray(source_image.dataobj), np.asarray(prepared_image.dataobj)
    ):
        raise SegmentationAdapterError("prepared final_tissues 与 CHARM 组织结果不一致")
    recorded = prepared_manifest.get("head_model_provenance")
    if not isinstance(recorded, Mapping):
        raise SegmentationAdapterError("prepared manifest 缺少 head_model_provenance")
    if scientific_metadata(recorded) != scientific_metadata(
        identity.to_prepared_provenance()
    ):
        raise SegmentationAdapterError("prepared manifest 与 CHARM runtime 身份不一致")
    return identity


class SimNIBSCharmAdapter:
    """Run complete CHARM segmentation through the external SimNIBS Python API."""

    def __init__(self, *, required_version: str = REQUIRED_SIMNIBS_VERSION) -> None:
        self.required_version = str(required_version)

    def run(self, *, canonical_t1: Path, output_dir: Path) -> Path:
        """Run CHARM in sibling staging and publish the head model without overwriting."""

        version = installed_simnibs_version()
        if version != self.required_version:
            raise SegmentationAdapterError(
                f"CHARM profile 要求 SimNIBS {self.required_version}，实际为 {version}。"
            )
        try:
            from simnibs.segmentation import charm_main
        except ImportError as exc:
            raise SegmentationAdapterError(
                "无法导入外部 simnibs.segmentation.charm_main。"
            ) from exc

        source = Path(canonical_t1).expanduser().resolve(strict=True)
        destination = Path(output_dir).expanduser()
        if not destination.name:
            raise ValueError("CHARM 输出目录不能是文件系统根目录")
        if _path_entry_exists(destination):
            raise FileExistsError(f"CHARM 输出目录已存在，拒绝覆盖：{destination}")
        before = _fingerprint(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.charm-staging-",
                dir=destination.parent,
            )
        )
        staging_output = staging_root / destination.name
        published = False
        try:
            charm_main.run(
                subject_dir=str(staging_output),
                T1=str(source),
                T2=None,
                registerT2=False,
                initatlas=True,
                segment=True,
                create_surfaces=True,
                mesh_image=True,
                usesettings=None,
                noneck=False,
                init_transform=None,
                use_transform=None,
                force_qform=False,
                force_sform=False,
                fs_dir=None,
                options_str="",
                debug=False,
            )
            final_tissues = staging_output / "final_tissues.nii.gz"
            if not final_tissues.is_file():
                raise SegmentationAdapterError(
                    "CHARM 完成后缺少 final_tissues.nii.gz。"
                )
            if _fingerprint(source) != before:
                raise SegmentationAdapterError("canonical T1 在 CHARM 运行期间发生变化")
            write_charm_runtime_manifest(
                canonical_t1=source,
                head_model_dir=staging_output,
                simnibs_version=version,
            )
            validate_charm_runtime(
                staging_output,
                canonical_t1=source,
                final_tissues=final_tissues,
            )
            if _path_entry_exists(destination):
                raise FileExistsError(
                    f"CHARM 输出在发布前出现，拒绝覆盖：{destination}"
                )
            _rename_noreplace(staging_output, destination)
            published = True
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)
        if not published:
            raise SegmentationAdapterError("CHARM 输出未发布")
        identity = validate_charm_runtime(
            destination,
            canonical_t1=source,
        )
        return identity.final_tissues_path


__all__ = [
    "CHARM_RUNTIME_MANIFEST_NAME",
    "CHARM_RUNTIME_SCHEMA_ID",
    "REQUIRED_SIMNIBS_VERSION",
    "CharmRuntimeIdentity",
    "SegmentationAdapter",
    "SegmentationAdapterError",
    "SimNIBSCharmAdapter",
    "installed_simnibs_version",
    "validate_charm_runtime",
    "validate_prepared_head_mesh_binding",
    "write_charm_runtime_manifest",
]
