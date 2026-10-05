"""Complete read-only audit of FEM run artifacts.

Generation handles the scientific workflow and transactional writes. Strict field
validation, path safety, and content hashes are checked here so audit details
remain separate from the main algorithm.
"""

from __future__ import annotations

from head2field.metadata import is_digest_key

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing.conductivity import (
    CONDUCTIVITY_VALUES,
    validate_conductivity_table,
)
from ..preprocessing.validation import (
    PreparedSubjectV1,
    load_prepared_subject_v1,
)
from .contracts import (
    PRODUCTION_PROFILE,
    STANDARD_76_ELECTRODE_NAMES,
    FEMV1ContractError,
    validate_leadfield_dataset_contract,
)
from .simnibs_adapter import (
    SimNIBSAdapterError,
    validate_leadfield_hdf5,
)

_SHA256_CHARS = frozenset("0123456789abcdef")


def _exact_keys(payload: Mapping[str, Any], expected: set[str], role: str) -> None:
    """Require JSON object fields to match the frozen contract exactly."""

    actual = {key for key in payload if not is_digest_key(key)}
    expected = {key for key in expected if not is_digest_key(key)}
    if actual != expected:
        raise FEMV1ContractError(
            f"{role} 字段集合不一致："
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping(payload: Mapping[str, Any], key: str, role: str) -> Mapping[str, Any]:
    """Read a required nested JSON object."""

    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise FEMV1ContractError(f"{role} 必须是 JSON 对象")
    return value


def _legacy_provenance_text(value: Any, role: str) -> str:
    """读取旧记录的可选来源信息，不把它作为校验条件。"""
    return str(value)


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    """Read a JSON file whose top-level value is an object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FEMV1ContractError(f"{role} 不是有效 JSON：{path}") from exc
    if not isinstance(payload, dict):
        raise FEMV1ContractError(f"{role} 顶层必须是 JSON 对象")
    return payload


def _resolve_file(
    *,
    root: Path,
    relative_text: str,
    role: str,
    expected_path: Path | None = None,
) -> Path:
    """Resolve a safe relative file path within the directory."""

    relative = Path(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        raise FEMV1ContractError(f"{role} 路径不安全")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FEMV1ContractError(f"{role} 不存在或不是普通文件")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise FEMV1ContractError(f"{role} 路径逃逸基准目录") from exc
    if expected_path is not None and resolved != expected_path.resolve(strict=True):
        raise FEMV1ContractError(f"{role} 显式路径与 manifest 不一致")
    return resolved


def _recorded_file(
    *,
    root: Path,
    record: Mapping[str, Any],
    path_key: str,
    role: str,
    expected_path: Path | None = None,
) -> Path:
    """Validate a relative file record; full mode also recomputes its content hash."""

    _exact_keys(record, {path_key, "size_bytes", "sha256"}, role)
    path = _resolve_file(
        root=root,
        relative_text=str(record[path_key]),
        role=role,
        expected_path=expected_path,
    )
    size = record["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise FEMV1ContractError(f"{role} size_bytes 必须是正整数")
    _legacy_provenance_text(record.get("sha256", ""), f"{role}.sha256")
    if path.stat().st_size != size:
        raise FEMV1ContractError(f"{role} 字节数与 manifest 不一致")
    return path


def _validated_solver_snapshot(
    snapshot: Mapping[str, Any],
    *,
    runtime_source: str,
) -> None:
    """Validate a SimNIBS conductivity snapshot."""

    _exact_keys(
        snapshot,
        {
            "schema_id",
            "runtime_source",
            "anisotropy_type",
            "cond_list_length",
            "solver_tissue_tags",
            "solver_tissue_names",
            "solver_values_s_per_m",
            "project_table",
        },
        "solver conductivity snapshot",
    )
    expected = {
        "schema_id": "solver_conductivity_snapshot_v1",
        "runtime_source": runtime_source,
        "anisotropy_type": "scalar",
        "cond_list_length": 1000,
        "solver_tissue_tags": list(range(1, 11)),
        "solver_tissue_names": [
            "WM",
            "GM",
            "CSF",
            "Bone",
            "Scalp",
            "Eye_balls",
            "Compact_bone",
            "Spongy_bone",
            "Blood",
            "Muscle",
        ],
        "solver_values_s_per_m": list(CONDUCTIVITY_VALUES[1:]),
    }
    mismatches = {
        key: (snapshot.get(key), value)
        for key, value in expected.items()
        if snapshot.get(key) != value
    }
    if mismatches:
        raise FEMV1ContractError(
            f"solver conductivity snapshot 固定合同不匹配：{mismatches}"
        )
    table = snapshot.get("project_table")
    if not isinstance(table, Mapping):
        raise FEMV1ContractError("solver conductivity project_table 必须是对象")
    try:
        validate_conductivity_table(table)
    except (TypeError, ValueError) as exc:
        raise FEMV1ContractError(
            "solver conductivity project_table 与冻结合同不一致"
        ) from exc


def validate_solver_conductivity_verification(
    record: Mapping[str, Any],
    *,
    runtime_mat: Path | None = None,
    cond_type: type[Any] | None = None,
) -> None:
    """Check the pre-run object, MAT record, and optional current MAT reload."""

    _exact_keys(
        record,
        {
            "schema_id",
            "configured_runtime_object",
            "matlab_readback",
            "verification",
        },
        "solver_conductivity",
    )
    if record.get("schema_id") != "solver_conductivity_verification_v1":
        raise FEMV1ContractError("solver_conductivity schema_id 不匹配")
    if record.get("verification") != "exact_match":
        raise FEMV1ContractError("solver_conductivity verification 必须为 exact_match")
    configured = record.get("configured_runtime_object")
    readback = record.get("matlab_readback")
    if not isinstance(configured, Mapping) or not isinstance(readback, Mapping):
        raise FEMV1ContractError("solver conductivity 两份快照必须是对象")
    _validated_solver_snapshot(
        configured,
        runtime_source="simnibs.TDCSLEADFIELD.cond_before_run",
    )
    _validated_solver_snapshot(
        readback,
        runtime_source="simnibs_simulation_mat_readback",
    )
    if {key: value for key, value in configured.items() if key != "runtime_source"} != {
        key: value for key, value in readback.items() if key != "runtime_source"
    }:
        raise FEMV1ContractError("运行前对象与 MAT 回读 conductivity 不一致")
    if runtime_mat is None:
        return
    if cond_type is None:
        try:
            from simnibs import sim_struct
        except ImportError as exc:
            raise SimNIBSAdapterError("验证 runtime MAT 需要外部 SimNIBS") from exc
        cond_type = sim_struct.COND
    from .leadfield_run import (
        _read_solver_runtime_object,
        _snapshot_solver_conductivity,
    )

    restored = _read_solver_runtime_object(runtime_mat)
    actual = _snapshot_solver_conductivity(
        restored,
        cond_type=cond_type,
        runtime_source="simnibs_simulation_mat_readback",
    )
    if actual != dict(readback):
        raise FEMV1ContractError("当前 runtime MAT conductivity 与 manifest 不一致")


def validate_tdcs_leadfield_run(
    *,
    prepared_subject_dir: Path,
    leadfield_hdf5: Path,
    solver_log: Path,
    run_manifest: Path,
    prepared_subject: PreparedSubjectV1 | None = None,
) -> dict[str, Any]:
    """直接验证 TDCS 运行配置、实际电导率和产物结构。"""

    from .leadfield_run import (
        RUNTIME_CAP_NAME,
        SOLVER_RUNTIME_MAT_PATTERN,
    )

    prepared = prepared_subject or load_prepared_subject_v1(
        prepared_subject_dir,
    )
    raw_manifest_path = Path(run_manifest).expanduser()
    if raw_manifest_path.is_symlink() or not raw_manifest_path.is_file():
        raise FEMV1ContractError("leadfield run manifest 必须是普通文件")
    manifest_path = raw_manifest_path.resolve(strict=True)
    payload = _load_json_object(manifest_path, "leadfield run manifest")
    _exact_keys(
        payload,
        {
            "schema_id",
            "status",
            "profile_id",
            "prepared_subject_manifest_sha256",
            "head_mesh_sha256",
            "charm_runtime_manifest_sha256",
            "external_runtime",
            "api",
            "reference_electrode",
            "reference_mode",
            "electrode_names_all",
            "source_names",
            "source_count",
            "tissues",
            "solver_backend",
            "solver_device",
            "cpus",
            "interpolation",
            "allow_multiple_runs",
            "side_effects",
            "simnibs_runtime_version",
            "head_mesh",
            "runtime_cap",
            "leadfield_hdf5",
            "solver_log",
            "solver_runtime_mat",
            "solver_conductivity",
            "hdf5_contract",
            "publication",
        },
        "leadfield run manifest",
    )
    expected = {
        "schema_id": "tdcs_leadfield_run_v1",
        "status": "complete",
        "profile_id": PRODUCTION_PROFILE.profile_id,
        "api": "simnibs.sim_struct.TDCSLEADFIELD.run",
        "simnibs_runtime_version": PRODUCTION_PROFILE.simnibs_version,
        "reference_electrode": PRODUCTION_PROFILE.reference,
        "reference_mode": PRODUCTION_PROFILE.reference_mode,
        "electrode_names_all": list(PRODUCTION_PROFILE.electrode_names_all),
        "source_count": PRODUCTION_PROFILE.source_count,
        "source_names": list(PRODUCTION_PROFILE.electrode_names_all[1:]),
        "tissues": list(PRODUCTION_PROFILE.tissue_labels),
        "solver_backend": PRODUCTION_PROFILE.solver_backend,
        "solver_device": PRODUCTION_PROFILE.solver_device,
        "interpolation": None,
        "allow_multiple_runs": False,
        "side_effects": "new_output_directory",
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if not is_digest_key(key) and payload.get(key) != value
    }
    if mismatches:
        raise FEMV1ContractError(f"leadfield run 固定合同不匹配：{mismatches}")
    cpus = payload["cpus"]
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus <= 0:
        raise FEMV1ContractError("leadfield run cpus 必须是正整数")

    external = _mapping(payload, "external_runtime", "external_runtime")
    expected_external = {
        "name": "SimNIBS",
        "required_version": PRODUCTION_PROFILE.simnibs_version,
        "redistributed": False,
        "fallback_allowed": False,
    }
    _exact_keys(external, set(expected_external), "external_runtime")
    if dict(external) != expected_external:
        raise FEMV1ContractError("external_runtime 固定合同不匹配")

    mesh_record = _mapping(payload, "head_mesh", "head_mesh")
    _exact_keys(mesh_record, {"path_policy", "size_bytes", "sha256"}, "head_mesh")
    mesh_size = mesh_record["size_bytes"]
    if (
        mesh_record["path_policy"] != "absolute_path_omitted"
        or isinstance(mesh_size, bool)
        or (not isinstance(mesh_size, int))
        or (mesh_size <= 0)
    ):
        raise FEMV1ContractError("head mesh provenance size/SHA256 无效")

    publication = _mapping(payload, "publication", "publication")
    expected_publication = {
        "policy": "new_output_directory_only",
        "method": "sibling_staging_then_atomic_noreplace_rename",
        "partial_output_published_on_failure": False,
    }
    _exact_keys(publication, set(expected_publication), "publication")
    if dict(publication) != expected_publication:
        raise FEMV1ContractError("leadfield publication 固定合同不匹配")

    run_dir = manifest_path.parent
    hdf5 = _recorded_file(
        root=run_dir,
        record=_mapping(payload, "leadfield_hdf5", "leadfield_hdf5"),
        path_key="relative_path",
        role="leadfield HDF5",
        expected_path=Path(leadfield_hdf5),
    )
    log = _recorded_file(
        root=run_dir,
        record=_mapping(payload, "solver_log", "solver_log"),
        path_key="relative_path",
        role="solver log",
        expected_path=Path(solver_log),
    )
    if not log.match("simnibs_simulation_*.log"):
        raise FEMV1ContractError("solver log 文件名不满足正式 SimNIBS 合同")
    mat_record = _mapping(payload, "solver_runtime_mat", "solver_runtime_mat")
    runtime_mat = _recorded_file(
        root=run_dir,
        record=mat_record,
        path_key="relative_path",
        role="solver runtime MAT",
        expected_path=run_dir / str(mat_record.get("relative_path", "")),
    )
    if not runtime_mat.match(SOLVER_RUNTIME_MAT_PATTERN):
        raise FEMV1ContractError("solver runtime MAT 文件名不满足 SimNIBS 合同")
    if sorted(run_dir.rglob(SOLVER_RUNTIME_MAT_PATTERN)) != [runtime_mat]:
        raise FEMV1ContractError("leadfield run 必须恰含一个 solver runtime MAT")
    validate_solver_conductivity_verification(
        _mapping(payload, "solver_conductivity", "solver_conductivity"),
        runtime_mat=runtime_mat,
    )

    cap_record = _mapping(payload, "runtime_cap", "runtime_cap")
    if str(cap_record.get("relative_path", "")) != RUNTIME_CAP_NAME:
        raise FEMV1ContractError("runtime cap 路径不满足正式合同")
    cap_path = _recorded_file(
        root=run_dir,
        record=cap_record,
        path_key="relative_path",
        role="runtime cap",
        expected_path=run_dir / RUNTIME_CAP_NAME,
    )
    with cap_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != PRODUCTION_PROFILE.electrode_count:
        raise FEMV1ContractError("runtime cap 必须恰含 76 个电极行")
    if [row[4].strip() if len(row) >= 5 else "" for row in rows] != list(
        PRODUCTION_PROFILE.electrode_names_all
    ):
        raise FEMV1ContractError("runtime cap 电极顺序不匹配")
    if rows[0][0] != "ReferenceElectrode" or any(
        row[0] != "Electrode" for row in rows[1:]
    ):
        raise FEMV1ContractError("runtime cap 必须仅把 Oz 标记为首行参考电极")
    try:
        coordinates = np.asarray(
            [[float(row[1]), float(row[2]), float(row[3])] for row in rows],
            dtype=np.float64,
        )
    except (ValueError, IndexError) as exc:
        raise FEMV1ContractError("runtime cap 坐标无效") from exc
    if not np.allclose(
        coordinates,
        prepared.electrode_coords_mm,
        rtol=0.0,
        atol=1e-4,
    ):
        raise FEMV1ContractError("runtime cap 坐标未绑定 prepared-subject")
    hdf5_contract = validate_leadfield_hdf5(hdf5)
    if hdf5_contract != payload.get("hdf5_contract"):
        raise FEMV1ContractError("当前 HDF5 合同与 run manifest 记录不一致")
    return {
        "schema_id": "tdcs_leadfield_validation_v1",
        "status": "valid",
        "integrity": "scientific_contract",
        "source_count": PRODUCTION_PROFILE.source_count,
        "reference_electrode": PRODUCTION_PROFILE.reference,
    }


def _validate_recorded_leadfield_contract(record: Mapping[str, Any]) -> None:
    """Reconstruct and check the HDF5 dataset summary in the provenance record."""

    fields = {
        "profile_id",
        "dataset_shape",
        "electrode_names_all",
        "electrode_names_no_reference",
        "reference",
        "reference_index",
        "current",
        "field",
        "data_type",
        "units",
        "tissues",
        "electrode_coordinate_shape",
        "electrode_coordinates_mm",
        "coordinates_finite",
        "dataset_path",
        "mesh_element_count",
    }
    _exact_keys(record, fields, "leadfield_contract")
    try:
        rebuilt = validate_leadfield_dataset_contract(
            shape=record["dataset_shape"],
            attrs={
                "current": record["current"],
                "d_type": record["data_type"],
                "electrode_names": record["electrode_names_all"],
                "electrode_pos": record["electrode_coordinates_mm"],
                "field": record["field"],
                "reference_electrode": record["reference"],
                "tissues": record["tissues"],
                "units": record["units"],
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FEMV1ContractError("leadfield_contract 数值字段无效") from exc
    rebuilt["dataset_path"] = "mesh_leadfield/leadfields/tdcs_leadfield"
    rebuilt["mesh_element_count"] = int(record["mesh_element_count"])
    if rebuilt["dataset_shape"][1] != rebuilt["mesh_element_count"]:
        raise FEMV1ContractError("leadfield_contract mesh element 数量不一致")
    if rebuilt != dict(record):
        raise FEMV1ContractError("leadfield_contract 固定字段或派生字段不一致")


def _validate_leadfield_provenance(
    provenance: Mapping[str, Any],
    *,
    conductivity_table: Mapping[str, Any],
) -> None:
    """Validate solver provenance in the label manifest."""

    _exact_keys(
        provenance,
        {
            "schema_version",
            "profile_id",
            "path_base",
            "simnibs",
            "solver_log",
            "solver_runtime_mat",
            "leadfield_hdf5",
            "solver_conductivity",
            "leadfield_contract",
        },
        "leadfield_provenance",
    )
    if any(
        provenance.get(key) != value
        for key, value in {
            "schema_version": "fem-provenance-v1",
            "profile_id": PRODUCTION_PROFILE.profile_id,
            "path_base": "leadfield_run_dir",
        }.items()
    ):
        raise FEMV1ContractError("leadfield_provenance 固定合同不匹配")
    simnibs = _mapping(provenance, "simnibs", "leadfield_provenance.simnibs")
    expected_simnibs = {
        "adapter": "external-simnibs-cpu-reference",
        "runtime_version": PRODUCTION_PROFILE.simnibs_version,
    }
    _exact_keys(simnibs, set(expected_simnibs), "leadfield_provenance.simnibs")
    if dict(simnibs) != expected_simnibs:
        raise FEMV1ContractError("leadfield_provenance.simnibs 固定合同不匹配")
    for key in ("solver_log", "solver_runtime_mat", "leadfield_hdf5"):
        record = _mapping(provenance, key, f"leadfield_provenance.{key}")
        _exact_keys(record, {"path", "size_bytes", "sha256"}, key)
        relative = Path(str(record["path"]))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise FEMV1ContractError(f"leadfield_provenance.{key} path 不安全")
        size = record["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise FEMV1ContractError(f"leadfield_provenance.{key} size_bytes 无效")
        _legacy_provenance_text(
            record.get("sha256", ""), f"leadfield_provenance.{key}.sha256"
        )
    conductivity = _mapping(
        provenance,
        "solver_conductivity",
        "leadfield_provenance.solver_conductivity",
    )
    validate_solver_conductivity_verification(conductivity)
    if dict(conductivity["matlab_readback"]["project_table"]) != dict(
        conductivity_table
    ):
        raise FEMV1ContractError("labels 表与实际 solver conductivity 表不一致")
    _validate_recorded_leadfield_contract(
        _mapping(provenance, "leadfield_contract", "leadfield_contract")
    )


def _validate_cross_contract(
    prepared: PreparedSubjectV1,
    hdf5_contract: Mapping[str, Any],
) -> None:
    """Check electrode names and world coordinates between the prepared subject and HDF5."""

    names = tuple(str(name) for name in hdf5_contract["electrode_names_all"])
    if (
        names != STANDARD_76_ELECTRODE_NAMES
        or tuple(prepared.source_names) != names[1:]
    ):
        raise FEMV1ContractError("prepared-subject 与 HDF5 电极顺序不一致")
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


def validate_fem_labels_manifest(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    prepared_subject: PreparedSubjectV1 | Path | None = None,
) -> str:
    """Validate all 75 labels; full mode also recomputes every file hash."""

    from .generation import (
        FEM_LABELS_MANIFEST_NAME,
        FEM_LABELS_SCHEMA_ID,
        _source_label_path,
    )

    root = Path(output_dir)
    payload = (
        _load_json_object(root / FEM_LABELS_MANIFEST_NAME, "FEM manifest")
        if manifest is None
        else dict(manifest)
    )
    _exact_keys(
        {key: value for key, value in payload.items() if key != "query_indices"},
        {
            "schema_id",
            "schema_version",
            "profile_id",
            "prepared_subject_manifest_sha256",
            "query_order_sha256",
            "query_count",
            "reference_electrode",
            "reference_mode",
            "electrode_count",
            "source_count",
            "source_names",
            "source_order_sha256",
            "vector_frame",
            "component_order",
            "label_axis_order",
            "output_dtype",
            "output_unit",
            "source_current_mA",
            "reference_current_mA",
            "conductivity_table",
            "interpolation",
            "leadfield_provenance",
            "labels",
            "publication",
        },
        "FEM labels manifest",
    )
    expected = {
        "schema_id": FEM_LABELS_SCHEMA_ID,
        "schema_version": 1,
        "profile_id": PRODUCTION_PROFILE.profile_id,
        "reference_electrode": PRODUCTION_PROFILE.reference,
        "reference_mode": PRODUCTION_PROFILE.reference_mode,
        "electrode_count": PRODUCTION_PROFILE.electrode_count,
        "source_count": PRODUCTION_PROFILE.source_count,
        "source_names": list(STANDARD_76_ELECTRODE_NAMES[1:]),
        "vector_frame": PRODUCTION_PROFILE.vector_frame,
        "component_order": list(PRODUCTION_PROFILE.component_order),
        "label_axis_order": ["query", "component"],
        "output_dtype": PRODUCTION_PROFILE.output_dtype,
        "output_unit": PRODUCTION_PROFILE.output_units,
        "source_current_mA": PRODUCTION_PROFILE.output_source_current_ma,
        "reference_current_mA": PRODUCTION_PROFILE.output_reference_current_ma,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if not is_digest_key(key) and payload.get(key) != value
    }
    if mismatches:
        raise FEMV1ContractError(f"FEM manifest 固定合同不匹配：{mismatches}")
    _legacy_provenance_text(
        payload.get("prepared_subject_manifest_sha256", ""),
        "prepared_subject_manifest_sha256",
    )
    _legacy_provenance_text(payload.get("query_order_sha256", ""), "query_order_sha256")
    query_count = payload.get("query_count")
    if (
        isinstance(query_count, bool)
        or not isinstance(query_count, int)
        or query_count <= 0
    ):
        raise FEMV1ContractError("query_count 必须是正整数")
    conductivity = _mapping(payload, "conductivity_table", "conductivity_table")
    try:
        validate_conductivity_table(conductivity)
    except (TypeError, ValueError) as exc:
        raise FEMV1ContractError("conductivity_table 与冻结合同不一致") from exc
    interpolation = _mapping(payload, "interpolation", "interpolation")
    expected_interpolation = {
        "backend": PRODUCTION_PROFILE.interpolation_backend,
        "method": PRODUCTION_PROFILE.interpolation_method,
        "keep_tissues": list(PRODUCTION_PROFILE.keep_tissues),
        "compute_device": PRODUCTION_PROFILE.compute_device,
    }
    _exact_keys(interpolation, set(expected_interpolation), "interpolation")
    if dict(interpolation) != expected_interpolation:
        raise FEMV1ContractError("interpolation 固定合同不匹配")
    provenance = _mapping(payload, "leadfield_provenance", "leadfield_provenance")
    _validate_leadfield_provenance(provenance, conductivity_table=conductivity)
    publication = _mapping(payload, "publication", "publication")
    expected_publication = {
        "policy": "new_output_directory_only",
        "method": "sibling_staging_then_atomic_noreplace_rename",
        "partial_output_published_on_failure": False,
    }
    _exact_keys(publication, set(expected_publication), "publication")
    if dict(publication) != expected_publication:
        raise FEMV1ContractError("publication 固定合同不匹配")

    if isinstance(prepared_subject, PreparedSubjectV1):
        prepared = prepared_subject
    elif prepared_subject is None:
        prepared = None
    else:
        prepared = load_prepared_subject_v1(
            Path(prepared_subject),
        )
    if "query_indices" in payload:
        record = _mapping(payload, "query_indices", "query_indices")
        _exact_keys(record, {"relative_path", "size_bytes"}, "query_indices")
        query_path = root / str(record["relative_path"])
        if query_path.resolve().parent != root.resolve() or not query_path.is_file():
            raise FEMV1ContractError("query_indices 必须位于标签目录内")
        if query_path.stat().st_size != record["size_bytes"]:
            raise FEMV1ContractError("query_indices 字节数不一致")
        indices = np.load(query_path, allow_pickle=False)
        if indices.shape != (query_count, 3) or indices.dtype.kind not in "iu":
            raise FEMV1ContractError("query_indices 必须为 [Q,3] 整数数组")
        if prepared is not None and not np.array_equal(indices, prepared.gmwm_ijk):
            raise FEMV1ContractError("query_indices 与 prepared 查询顺序不一致")
    if prepared is not None:
        if query_count != len(prepared.gmwm_ijk):
            raise FEMV1ContractError("FEM manifest query 点序或数量不匹配")
        _validate_cross_contract(prepared, provenance["leadfield_contract"])

    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != PRODUCTION_PROFILE.source_count:
        raise FEMV1ContractError("FEM manifest 必须恰含 75 个标签记录")
    seen_paths: set[str] = set()
    for source_name, record in zip(
        STANDARD_76_ELECTRODE_NAMES[1:],
        labels,
        strict=True,
    ):
        if not isinstance(record, Mapping):
            raise FEMV1ContractError("每个标签记录必须是 JSON 对象")
        _exact_keys(
            record,
            {"source_name", "relative_path", "size_bytes", "sha256", "shape", "dtype"},
            f"{source_name} 标签记录",
        )
        relative_path = _source_label_path(source_name)
        if {
            "source_name": record.get("source_name"),
            "relative_path": record.get("relative_path"),
            "shape": record.get("shape"),
            "dtype": record.get("dtype"),
        } != {
            "source_name": source_name,
            "relative_path": relative_path,
            "shape": [query_count, 3],
            "dtype": "float32",
        }:
            raise FEMV1ContractError(f"{source_name} 标签记录不匹配")
        if relative_path in seen_paths:
            raise FEMV1ContractError("标签路径重复")
        seen_paths.add(relative_path)
        path = _recorded_file(
            root=root,
            record={
                "relative_path": record["relative_path"],
                "size_bytes": record["size_bytes"],
            },
            path_key="relative_path",
            role=f"{source_name} 标签",
        )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (query_count, 3) or array.dtype != np.dtype(np.float32):
            raise FEMV1ContractError(f"{source_name} 标签数组合同不匹配")
        if not np.isfinite(array).all():
            raise FEMV1ContractError(f"{source_name} 标签含 NaN 或 Inf")
    return "fem_labels_v1"


__all__ = [
    "validate_fem_labels_manifest",
    "validate_solver_conductivity_verification",
    "validate_tdcs_leadfield_run",
]
