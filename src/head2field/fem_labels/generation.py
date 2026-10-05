"""Public entry points for planning, generating, loading, and validating FEM labels."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing.conductivity import (
    conductivity_table_metadata,
    validate_conductivity_table,
)
from ..preprocessing.validation import PreparedSubjectV1, load_prepared_subject_v1
from .contracts import (
    PRODUCTION_PROFILE,
    STANDARD_76_ELECTRODE_NAMES,
    FEMV1ContractError,
)
from .leadfield_run import (
    dry_run_tdcs_leadfield,
    run_tdcs_leadfield,
    validate_tdcs_leadfield_run,
)
from .simnibs_adapter import (
    build_base_mesh,
    installed_simnibs_version,
    interpolate_element_field_at_queries,
    read_source_element_field,
)
from .transaction import staged_output_directory

FEM_LABELS_SCHEMA_ID = "fem_labels_v1"
FEM_LABELS_MANIFEST_NAME = "fem_labels_v1.json"
FEM_LABELS_PLAN_SCHEMA_ID = "fem_labels_plan_v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_SOURCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    """Generate canonical JSON bytes independent of key order."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    """Compute the SHA256 of canonical JSON."""

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validated_sha256(value: str, role: str) -> str:
    """Validate and return a lowercase hexadecimal SHA256."""

    text = str(value)
    return text


def _source_label_path(source_name: str) -> str:
    """Map frozen electrode names to safe relative label paths."""

    if _SAFE_SOURCE_PATTERN.fullmatch(source_name) is None:
        raise FEMV1ContractError(f"电极名不能安全映射为文件名：{source_name!r}")
    return f"labels/{source_name}_to_Oz_float32.npy"


@dataclass(frozen=True)
class FEMLabelGenerationPlan:
    """Side-effect-free dry-run plan without file writes or SimNIBS imports."""

    prepared_subject_manifest_sha256: str
    query_order_sha256: str
    query_count: int
    source_names: tuple[str, ...]
    relative_label_paths: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        """Return a plan without absolute paths or runtime-specific identifiers."""

        return {
            "schema_id": FEM_LABELS_PLAN_SCHEMA_ID,
            "status": "dry_run",
            "profile_id": PRODUCTION_PROFILE.profile_id,
            "query_count": self.query_count,
            "reference_electrode": PRODUCTION_PROFILE.reference,
            "reference_mode": PRODUCTION_PROFILE.reference_mode,
            "electrode_count": PRODUCTION_PROFILE.electrode_count,
            "source_count": len(self.source_names),
            "source_names": list(self.source_names),
            "vector_frame": PRODUCTION_PROFILE.vector_frame,
            "component_order": list(PRODUCTION_PROFILE.component_order),
            "label_axis_order": ["query", "component"],
            "output_dtype": PRODUCTION_PROFILE.output_dtype,
            "output_unit": PRODUCTION_PROFILE.output_units,
            "source_current_mA": PRODUCTION_PROFILE.output_source_current_ma,
            "reference_current_mA": PRODUCTION_PROFILE.output_reference_current_ma,
            "labels": [
                {
                    "source_name": source_name,
                    "relative_path": relative_path,
                    "shape": [self.query_count, 3],
                    "dtype": PRODUCTION_PROFILE.output_dtype,
                }
                for source_name, relative_path in zip(
                    self.source_names,
                    self.relative_label_paths,
                    strict=True,
                )
            ],
            "side_effects": "none",
        }


def build_fem_label_plan(
    *,
    query_count: int,
    prepared_subject_manifest_sha256: str,
    query_order_sha256: str,
    source_names: Sequence[str] = STANDARD_76_ELECTRODE_NAMES[1:],
) -> FEMLabelGenerationPlan:
    """Build a pure dry-run plan for 75 source-to-Oz fields."""

    if isinstance(query_count, bool) or not isinstance(query_count, int):
        raise TypeError("query_count 必须是整数")
    if query_count <= 0:
        raise FEMV1ContractError("FEM 计划至少需要一个查询点")
    normalized_sources = tuple(str(name) for name in source_names)
    if normalized_sources != STANDARD_76_ELECTRODE_NAMES[1:]:
        raise FEMV1ContractError("source_names 必须是标准 76 电极去除 Oz 后的顺序")
    relative_paths = tuple(_source_label_path(name) for name in normalized_sources)
    if len(set(relative_paths)) != len(relative_paths):
        raise FEMV1ContractError("标签相对路径发生冲突")
    return FEMLabelGenerationPlan(
        prepared_subject_manifest_sha256=_validated_sha256(
            prepared_subject_manifest_sha256,
            "prepared_subject_manifest_sha256",
        ),
        query_order_sha256=_validated_sha256(
            query_order_sha256,
            "query_order_sha256",
        ),
        query_count=query_count,
        source_names=normalized_sources,
        relative_label_paths=relative_paths,
    )


def _plan_from_prepared(prepared: PreparedSubjectV1) -> FEMLabelGenerationPlan:
    """Extract immutable planning inputs from a validated prepared subject."""

    query_record = prepared.manifest["files"]["voxels/gmwm_ijk.npy"]
    return build_fem_label_plan(
        query_count=len(prepared.gmwm_ijk),
        prepared_subject_manifest_sha256=prepared.manifest_sha256,
        query_order_sha256=str(query_record.get("sha256", "")),
        source_names=prepared.source_names,
    )


def dry_run_fem_label_generation(
    prepared_subject_dir: Path,
) -> FEMLabelGenerationPlan:
    """Validate a prepared subject read-only and return a side-effect-free label plan."""

    return _plan_from_prepared(
        load_prepared_subject_v1(
            prepared_subject_dir,
        )
    )


def _validate_cross_contract(
    prepared: PreparedSubjectV1,
    hdf5_contract: Mapping[str, Any],
) -> None:
    """Check electrode names and world coordinates between the prepared subject and HDF5."""

    names = tuple(str(name) for name in hdf5_contract["electrode_names_all"])
    if names != STANDARD_76_ELECTRODE_NAMES:
        raise FEMV1ContractError("HDF5 电极顺序与标准 76 电极不一致")
    if tuple(prepared.source_names) != names[1:]:
        raise FEMV1ContractError("prepared-subject source 顺序与 HDF5 不一致")
    coordinates = np.asarray(
        hdf5_contract["electrode_coordinates_mm"], dtype=np.float64
    )
    expected = np.asarray(prepared.electrode_coords_mm, dtype=np.float64)
    if coordinates.shape != expected.shape or not np.allclose(
        coordinates,
        expected,
        rtol=0.0,
        atol=1e-4,
    ):
        raise FEMV1ContractError("prepared-subject 与 HDF5 的 world 电极坐标不一致")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a new JSON file using stable UTF-8 encoding."""

    if os.path.lexists(path):
        raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _label_record(
    path: Path,
    *,
    source_name: str,
    array: np.ndarray,
    query_count: int,
) -> dict[str, Any]:
    """Validate in-memory labels and record the identity of the newly written file."""

    if array.shape != (query_count, 3):
        raise FEMV1ContractError(
            f"{source_name} 标签 shape={array.shape}，期望 {(query_count, 3)}"
        )
    if array.dtype != np.dtype(np.float32):
        raise FEMV1ContractError(f"{source_name} 标签 dtype 必须为 float32")
    if not np.isfinite(array).all():
        raise FEMV1ContractError(f"{source_name} 标签含 NaN 或 Inf")
    return {
        "source_name": source_name,
        "relative_path": f"{path.parent.name}/{path.name}",
        "size_bytes": int(path.stat().st_size),
        "shape": [query_count, 3],
        "dtype": "float32",
    }


def _run_record_as_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a validated run record into label provenance."""

    return {
        "path": str(record["relative_path"]),
        "size_bytes": int(record["size_bytes"]),
    }


def _build_provenance(
    run_manifest: Mapping[str, Any],
    *,
    hdf5_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse a validated run identity without rereading or rehashing large files."""

    return {
        "schema_version": "fem-provenance-v1",
        "profile_id": PRODUCTION_PROFILE.profile_id,
        "path_base": "leadfield_run_dir",
        "simnibs": {
            "adapter": "external-simnibs-cpu-reference",
            "runtime_version": PRODUCTION_PROFILE.simnibs_version,
        },
        "solver_log": _run_record_as_provenance(run_manifest["solver_log"]),
        "solver_runtime_mat": _run_record_as_provenance(
            run_manifest["solver_runtime_mat"]
        ),
        "leadfield_hdf5": _run_record_as_provenance(run_manifest["leadfield_hdf5"]),
        "solver_conductivity": dict(run_manifest["solver_conductivity"]),
        "leadfield_contract": dict(hdf5_contract),
    }


def _base_manifest(
    *,
    plan: FEMLabelGenerationPlan,
    labels: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the FEM labels v1 manifest."""

    table = conductivity_table_metadata()
    validate_conductivity_table(table)
    return {
        "schema_id": FEM_LABELS_SCHEMA_ID,
        "schema_version": 1,
        "profile_id": PRODUCTION_PROFILE.profile_id,
        "query_count": plan.query_count,
        "reference_electrode": PRODUCTION_PROFILE.reference,
        "reference_mode": PRODUCTION_PROFILE.reference_mode,
        "electrode_count": PRODUCTION_PROFILE.electrode_count,
        "source_count": PRODUCTION_PROFILE.source_count,
        "source_names": list(plan.source_names),
        "vector_frame": PRODUCTION_PROFILE.vector_frame,
        "component_order": list(PRODUCTION_PROFILE.component_order),
        "label_axis_order": ["query", "component"],
        "output_dtype": PRODUCTION_PROFILE.output_dtype,
        "output_unit": PRODUCTION_PROFILE.output_units,
        "source_current_mA": PRODUCTION_PROFILE.output_source_current_ma,
        "reference_current_mA": PRODUCTION_PROFILE.output_reference_current_ma,
        "conductivity_table": table,
        "interpolation": {
            "backend": PRODUCTION_PROFILE.interpolation_backend,
            "method": PRODUCTION_PROFILE.interpolation_method,
            "keep_tissues": list(PRODUCTION_PROFILE.keep_tissues),
            "compute_device": PRODUCTION_PROFILE.compute_device,
        },
        "leadfield_provenance": dict(provenance),
        "labels": [dict(record) for record in labels],
        "publication": {
            "policy": "new_output_directory_only",
            "method": "sibling_staging_then_atomic_noreplace_rename",
            "partial_output_published_on_failure": False,
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest whose top-level value must be a JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FEMV1ContractError(f"FEM manifest 不是有效 JSON：{path}") from exc
    if not isinstance(payload, dict):
        raise FEMV1ContractError("FEM manifest 顶层必须是 JSON 对象")
    return payload


def validate_fem_labels_manifest(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    prepared_subject: PreparedSubjectV1 | Path | None = None,
) -> str:
    """Use the independent audit module to validate the complete label manifest."""

    from .audit import validate_fem_labels_manifest as audit

    return audit(
        output_dir,
        manifest=manifest,
        prepared_subject=prepared_subject,
    )


@dataclass(frozen=True)
class FEMLabelsV1:
    """Read-only handle to FEM labels that have passed contract validation."""

    output_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    source_names: tuple[str, ...]
    label_paths: tuple[Path, ...]

    def load_source(
        self,
        source_name: str,
        *,
        mmap_mode: str | None = "r",
    ) -> np.ndarray:
        """Load a [query, component] label array using the frozen source name."""

        try:
            index = self.source_names.index(source_name)
        except ValueError as exc:
            raise KeyError(f"未知 source：{source_name!r}") from exc
        return np.load(self.label_paths[index], mmap_mode=mmap_mode, allow_pickle=False)


def load_fem_labels_v1(
    output_dir: Path,
    *,
    prepared_subject: PreparedSubjectV1 | Path | None = None,
) -> FEMLabelsV1:
    """Validate and return labels without implicitly merging or reordering sources."""

    root = Path(output_dir)
    manifest_path = root / FEM_LABELS_MANIFEST_NAME
    payload = _load_manifest(manifest_path)
    manifest_hash = validate_fem_labels_manifest(
        root,
        manifest=payload,
        prepared_subject=prepared_subject,
    )
    names = tuple(str(name) for name in payload["source_names"])
    paths = tuple(root / str(record["relative_path"]) for record in payload["labels"])
    return FEMLabelsV1(
        output_dir=root,
        manifest_path=manifest_path,
        manifest=payload,
        manifest_sha256=manifest_hash,
        source_names=names,
        label_paths=paths,
    )


def export_fem_labels_from_hdf5(
    *,
    prepared_subject_dir: Path,
    leadfield_hdf5: Path,
    solver_log: Path,
    leadfield_run_manifest: Path,
    output_dir: Path,
) -> Path:
    """Interpolate 75 source-to-Oz labels from validated HDF5 and publish transactionally."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    plan = _plan_from_prepared(prepared)
    hdf5_path = Path(leadfield_hdf5).expanduser().resolve(strict=True)
    log_path = Path(solver_log).expanduser().resolve(strict=True)
    run_manifest_path = Path(leadfield_run_manifest).expanduser().resolve(strict=True)
    if not (hdf5_path.parent == log_path.parent == run_manifest_path.parent):
        raise FEMV1ContractError("solver log、HDF5 与 run manifest 必须来自同一目录")
    validate_tdcs_leadfield_run(
        prepared_subject_dir=prepared_subject_dir,
        leadfield_hdf5=hdf5_path,
        solver_log=log_path,
        run_manifest=run_manifest_path,
        prepared_subject=prepared,
    )
    run_manifest = _load_manifest(run_manifest_path)
    hdf5_contract = run_manifest["hdf5_contract"]
    _validate_cross_contract(prepared, hdf5_contract)
    version = installed_simnibs_version()
    if version != PRODUCTION_PROFILE.simnibs_version:
        raise FEMV1ContractError(
            f"FEM profile 要求 SimNIBS {PRODUCTION_PROFILE.simnibs_version}，"
            f"实际为 {version}"
        )
    provenance = _build_provenance(
        run_manifest,
        hdf5_contract=hdf5_contract,
    )

    destination = Path(output_dir).expanduser()
    with staged_output_directory(destination) as staging:
        labels_dir = staging / "labels"
        labels_dir.mkdir(parents=True, exist_ok=False)
        # 保存查询顺序本身，供加载时逐项比较。
        np.save(staging / "query_ijk.npy", prepared.gmwm_ijk, allow_pickle=False)
        mesh = build_base_mesh(hdf5_path)
        reference_nifti = prepared.subject_dir / "images" / "T1_1mm.nii.gz"
        records: list[dict[str, Any]] = []
        for source_index, source_name in enumerate(plan.source_names):
            element_field = read_source_element_field(
                hdf5_path,
                source_index=source_index,
            )
            labels = interpolate_element_field_at_queries(
                base_mesh=mesh,
                element_field=element_field,
                reference_nifti=reference_nifti,
                query_ijk=prepared.gmwm_ijk,
            )
            array = np.asarray(labels, dtype=np.float32)
            label_path = staging / plan.relative_label_paths[source_index]
            np.save(label_path, array, allow_pickle=False)
            records.append(
                _label_record(
                    label_path,
                    source_name=source_name,
                    array=array,
                    query_count=plan.query_count,
                )
            )
        manifest = _base_manifest(
            plan=plan,
            labels=records,
            provenance=provenance,
        )
        manifest["query_indices"] = {
            "relative_path": "query_ijk.npy",
            "size_bytes": (staging / "query_ijk.npy").stat().st_size,
        }
        _write_json(staging / FEM_LABELS_MANIFEST_NAME, manifest)
    return destination


def dry_run_fem_generation(
    *,
    prepared_subject_dir: Path,
    head_mesh: Path,
    cpus: int = 1,
) -> dict[str, Any]:
    """Build a complete side-effect-free plan for TDCS solving and label export."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    return {
        "schema_id": "fem_generation_plan_v1",
        "status": "dry_run",
        "leadfield": dry_run_tdcs_leadfield(
            prepared_subject_dir=prepared_subject_dir,
            head_mesh=head_mesh,
            cpus=cpus,
        ).to_manifest(),
        "label_export": _plan_from_prepared(prepared).to_manifest(),
        "side_effects": "none",
    }


def generate_fem_labels(
    *,
    prepared_subject_dir: Path,
    head_mesh: Path,
    leadfield_output_dir: Path,
    labels_output_dir: Path,
    cpus: int = 1,
) -> Path:
    """Orchestrate solving and complete label generation from prepared inputs."""

    leadfield_destination = Path(leadfield_output_dir).expanduser().resolve(False)
    labels_destination = Path(labels_output_dir).expanduser().resolve(False)
    if (
        leadfield_destination == labels_destination
        or leadfield_destination in labels_destination.parents
        or labels_destination in leadfield_destination.parents
    ):
        raise ValueError("leadfield_output_dir 与 labels_output_dir 不得相同或互相嵌套")
    run = run_tdcs_leadfield(
        prepared_subject_dir=prepared_subject_dir,
        head_mesh=head_mesh,
        output_dir=leadfield_destination,
        cpus=cpus,
    )
    return export_fem_labels_from_hdf5(
        prepared_subject_dir=prepared_subject_dir,
        leadfield_hdf5=run.hdf5_path,
        solver_log=run.solver_log,
        leadfield_run_manifest=run.manifest_path,
        output_dir=labels_destination,
    )


def validate_fem_artifacts(
    *,
    prepared_subject_dir: Path,
    fem_labels_dir: Path | None = None,
    leadfield_hdf5: Path | None = None,
    solver_log: Path | None = None,
    leadfield_run_manifest: Path | None = None,
) -> dict[str, Any]:
    """Audit FEM artifacts read-only; full mode also recomputes file hashes."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    hdf_inputs = (leadfield_hdf5, solver_log, leadfield_run_manifest)
    if any(value is not None for value in hdf_inputs) and not all(
        value is not None for value in hdf_inputs
    ):
        raise ValueError("HDF5 验收必须同时提供 HDF5、solver log 和 run manifest")
    if fem_labels_dir is None and leadfield_hdf5 is None:
        raise ValueError("至少提供 fem_labels_dir 或完整 HDF5 验收三件套")
    result: dict[str, Any] = {
        "schema_id": "fem_artifact_validation_v1",
        "status": "valid",
        "integrity": "scientific_contract",
    }
    if fem_labels_dir is not None:
        labels = load_fem_labels_v1(
            fem_labels_dir,
            prepared_subject=prepared,
        )
        result["fem_labels"] = {
            "source_count": len(labels.source_names),
            "query_count": int(labels.manifest["query_count"]),
            "reference_electrode": labels.manifest["reference_electrode"],
        }
    if leadfield_hdf5 is not None:
        assert solver_log is not None
        assert leadfield_run_manifest is not None
        result["leadfield_run"] = validate_tdcs_leadfield_run(
            prepared_subject_dir=prepared_subject_dir,
            leadfield_hdf5=leadfield_hdf5,
            solver_log=solver_log,
            run_manifest=leadfield_run_manifest,
        )
    return result


__all__ = [
    "FEM_LABELS_MANIFEST_NAME",
    "FEM_LABELS_PLAN_SCHEMA_ID",
    "FEM_LABELS_SCHEMA_ID",
    "FEMLabelGenerationPlan",
    "FEMLabelsV1",
    "build_fem_label_plan",
    "dry_run_fem_generation",
    "dry_run_fem_label_generation",
    "export_fem_labels_from_hdf5",
    "generate_fem_labels",
    "load_fem_labels_v1",
    "validate_fem_artifacts",
    "validate_fem_labels_manifest",
]
