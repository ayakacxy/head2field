"""Generate Oz-reference HDF5 with an external SimNIBS TDCSLEADFIELD runtime."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing.charm_adapter import validate_prepared_head_mesh_binding
from ..preprocessing.conductivity import (
    CONDUCTIVITY_TABLE_ID,
    CONDUCTIVITY_UNIT,
    CONDUCTIVITY_VALUES,
    validate_conductivity_table,
)
from ..preprocessing.validation import PreparedSubjectV1, load_prepared_subject_v1
from .contracts import PRODUCTION_PROFILE, FEMV1ContractError
from .simnibs_adapter import (
    SimNIBSAdapterError,
    installed_simnibs_version,
    validate_leadfield_hdf5,
)
from .transaction import _path_entry_exists, _rename_noreplace

LEADFIELD_RUN_MANIFEST_NAME = "leadfield_run_v1.json"
RUNTIME_CAP_NAME = "standard_76_Oz_reference.csv"
SOLVER_RUNTIME_MAT_PATTERN = "simnibs_simulation_*.mat"
SOLVER_TISSUE_TAGS = tuple(range(1, 11))
SOLVER_TISSUE_NAMES = (
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
)
SIMNIBS_COND_LIST_LENGTH = 1000


@dataclass(frozen=True)
class TDCSLeadfieldPlan:
    """TDCSLEADFIELD plan without launching SimNIBS or writing files."""

    prepared_subject_manifest_sha256: str
    head_mesh_sha256: str
    charm_runtime_manifest_sha256: str
    source_names: tuple[str, ...]
    cpus: int

    def to_manifest(self) -> dict[str, Any]:
        """Return a dry-run plan without machine-specific paths."""

        return {
            "schema_id": "tdcs_leadfield_plan_v1",
            "status": "dry_run",
            "profile_id": PRODUCTION_PROFILE.profile_id,
            "external_runtime": {
                "name": "SimNIBS",
                "required_version": PRODUCTION_PROFILE.simnibs_version,
                "redistributed": False,
                "fallback_allowed": False,
            },
            "api": "simnibs.sim_struct.TDCSLEADFIELD.run",
            "reference_electrode": PRODUCTION_PROFILE.reference,
            "reference_mode": PRODUCTION_PROFILE.reference_mode,
            "electrode_names_all": list(PRODUCTION_PROFILE.electrode_names_all),
            "source_names": list(self.source_names),
            "source_count": len(self.source_names),
            "tissues": list(PRODUCTION_PROFILE.tissue_labels),
            "solver_backend": PRODUCTION_PROFILE.solver_backend,
            "solver_device": PRODUCTION_PROFILE.solver_device,
            "cpus": self.cpus,
            "interpolation": None,
            "allow_multiple_runs": False,
            "side_effects": "none",
        }


@dataclass(frozen=True)
class LeadfieldRun:
    """External SimNIBS run result that has passed scientific contract validation."""

    output_dir: Path
    hdf5_path: Path
    solver_log: Path
    solver_runtime_mat: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _validate_cpus(cpus: int) -> int:
    """Validate the CPU count for the external solver."""

    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus <= 0:
        raise ValueError("cpus 必须是正整数")
    return cpus


def _actual_project_table(values: list[float]) -> dict[str, Any]:
    """Reconstruct the project conductivity table from run-object values."""

    payload: dict[str, Any] = {
        "id": CONDUCTIVITY_TABLE_ID,
        "unit": CONDUCTIVITY_UNIT,
        "values": values,
    }
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    validate_conductivity_table(payload)
    return payload


def _snapshot_solver_conductivity(
    runtime_object: Any,
    *,
    cond_type: type[Any],
    runtime_source: str,
) -> dict[str, Any]:
    """Extract tags 1--10 from TDCSLEADFIELD or a reloaded MAT object."""

    if getattr(runtime_object, "anisotropy_type", None) != "scalar":
        raise FEMV1ContractError("FEM solver 必须使用 scalar conductivity")
    cond = getattr(runtime_object, "cond", None)
    if not isinstance(cond, list) or len(cond) != SIMNIBS_COND_LIST_LENGTH:
        raise FEMV1ContractError("SimNIBS cond 必须保留官方 1000 项列表")
    names: list[str] = []
    values: list[float] = []
    for tag, expected_name in zip(
        SOLVER_TISSUE_TAGS,
        SOLVER_TISSUE_NAMES,
        strict=True,
    ):
        item = cond[tag - 1]
        if not isinstance(item, cond_type):
            raise FEMV1ContractError(f"SimNIBS cond[{tag - 1}] 不是官方 COND 对象")
        name = str(getattr(item, "name", "")).strip()
        try:
            value = float(item.value)
        except (TypeError, ValueError) as exc:
            raise FEMV1ContractError(f"tissue tag {tag} conductivity 不是实数") from exc
        if name != expected_name or not np.isfinite(value):
            raise FEMV1ContractError(
                f"tissue tag {tag} 名称/电导率无效：{name!r}/{value!r}"
            )
        names.append(name)
        values.append(value)
    return {
        "schema_id": "solver_conductivity_snapshot_v1",
        "runtime_source": runtime_source,
        "anisotropy_type": "scalar",
        "cond_list_length": len(cond),
        "solver_tissue_tags": list(SOLVER_TISSUE_TAGS),
        "solver_tissue_names": names,
        "solver_values_s_per_m": values,
        "project_table": _actual_project_table(
            [float(CONDUCTIVITY_VALUES[0]), *values]
        ),
    }


def _configure_solver_conductivity(tdcs: Any, *, sim_struct: Any) -> dict[str, Any]:
    """Set tags 1--10 using official COND objects and retain the complete cond list."""

    cond = getattr(tdcs, "cond", None)
    if not isinstance(cond, list) or len(cond) != SIMNIBS_COND_LIST_LENGTH:
        raise FEMV1ContractError("TDCSLEADFIELD 没有官方 1000 项 cond 列表")
    tdcs.anisotropy_type = "scalar"
    for tag, name, value in zip(
        SOLVER_TISSUE_TAGS,
        SOLVER_TISSUE_NAMES,
        CONDUCTIVITY_VALUES[1:],
        strict=True,
    ):
        item = sim_struct.COND()
        item.name = name
        item.value = float(value)
        item.distribution_type = None
        item.distribution_parameters = []
        tdcs.cond[tag - 1] = item
    return _snapshot_solver_conductivity(
        tdcs,
        cond_type=sim_struct.COND,
        runtime_source="simnibs.TDCSLEADFIELD.cond_before_run",
    )


def _read_solver_runtime_object(path: Path) -> Any:
    """Reload the pre-solver run object through the official SimNIBS API."""

    try:
        from simnibs.utils.matlab_read import read_mat
    except ImportError as exc:
        raise SimNIBSAdapterError("无法导入 SimNIBS 官方 MATLAB 回读 API") from exc
    try:
        return read_mat(str(path))
    except (OSError, ValueError) as exc:
        raise SimNIBSAdapterError(f"无法回读 SimNIBS runtime MAT：{path}") from exc


def validate_solver_conductivity_verification(
    record: Mapping[str, Any],
    *,
    runtime_mat: Path | None = None,
    cond_type: type[Any] | None = None,
) -> None:
    """Use the independent audit module to check conductivity run evidence."""

    from .audit import validate_solver_conductivity_verification as audit

    audit(record, runtime_mat=runtime_mat, cond_type=cond_type)


def validate_tdcs_leadfield_run(
    *,
    prepared_subject_dir: Path,
    leadfield_hdf5: Path,
    solver_log: Path,
    run_manifest: Path,
    prepared_subject: PreparedSubjectV1 | None = None,
) -> dict[str, Any]:
    """Use the independent audit module to validate the TDCS run directory."""

    from .audit import validate_tdcs_leadfield_run as audit

    return audit(
        prepared_subject_dir=prepared_subject_dir,
        leadfield_hdf5=leadfield_hdf5,
        solver_log=solver_log,
        run_manifest=run_manifest,
        prepared_subject=prepared_subject,
    )


def build_tdcs_leadfield_plan(
    *,
    prepared_subject: PreparedSubjectV1,
    head_mesh_sha256: str,
    charm_runtime_manifest_sha256: str,
    cpus: int = 1,
) -> TDCSLeadfieldPlan:
    """Build a 75-source solve plan from a validated prepared subject."""

    if (
        tuple(prepared_subject.source_names)
        != PRODUCTION_PROFILE.electrode_names_all[1:]
    ):
        raise FEMV1ContractError("prepared-subject source 顺序不是正式 75 电极顺序")
    return TDCSLeadfieldPlan(
        prepared_subject_manifest_sha256=prepared_subject.manifest_sha256,
        head_mesh_sha256=head_mesh_sha256,
        charm_runtime_manifest_sha256=charm_runtime_manifest_sha256,
        source_names=tuple(prepared_subject.source_names),
        cpus=_validate_cpus(cpus),
    )


def dry_run_tdcs_leadfield(
    *,
    prepared_subject_dir: Path,
    head_mesh: Path,
    cpus: int = 1,
) -> TDCSLeadfieldPlan:
    """Validate a prepared subject and mesh read-only and return a side-effect-free plan."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    mesh = Path(head_mesh).expanduser().resolve(strict=True)
    if not mesh.is_file():
        raise FileNotFoundError(f"head mesh 不是普通文件：{mesh}")
    charm_identity = validate_prepared_head_mesh_binding(
        prepared_subject_dir=prepared.subject_dir,
        prepared_manifest=prepared.manifest,
        head_mesh=mesh,
    )
    return build_tdcs_leadfield_plan(
        prepared_subject=prepared,
        head_mesh_sha256=charm_identity.head_mesh_sha256,
        charm_runtime_manifest_sha256=charm_identity.manifest_sha256,
        cpus=cpus,
    )


def _write_runtime_cap(path: Path, prepared: PreparedSubjectV1) -> None:
    """Write a run cap with Oz first using prepared-subject world-RAS coordinates."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for index, (name, coordinate) in enumerate(
            zip(
                prepared.electrode_names_all,
                prepared.electrode_coords_mm,
                strict=True,
            )
        ):
            writer.writerow(
                [
                    "ReferenceElectrode" if index == 0 else "Electrode",
                    format(float(coordinate[0]), ".9g"),
                    format(float(coordinate[1]), ".9g"),
                    format(float(coordinate[2]), ".9g"),
                    name,
                ]
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _single_output(paths: list[Path], role: str) -> Path:
    """Require exactly one generated file of the specified type."""

    if len(paths) != 1:
        raise SimNIBSAdapterError(
            f"TDCSLEADFIELD 要求恰有一个 {role}，实际为 {len(paths)} 个"
        )
    return paths[0]


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    """Record a stable identity for a newly generated run file."""

    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
    }


def run_tdcs_leadfield(
    *,
    prepared_subject_dir: Path,
    head_mesh: Path,
    output_dir: Path,
    cpus: int = 1,
) -> LeadfieldRun:
    """Run SimNIBS and transactionally publish 75-source Oz-reference HDF5."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    mesh = Path(head_mesh).expanduser().resolve(strict=True)
    if not mesh.is_file():
        raise FileNotFoundError(f"head mesh 不是普通文件：{mesh}")
    charm_identity = validate_prepared_head_mesh_binding(
        prepared_subject_dir=prepared.subject_dir,
        prepared_manifest=prepared.manifest,
        head_mesh=mesh,
    )
    mesh_stat = mesh.stat()
    mesh_fingerprint = (int(mesh_stat.st_size), int(mesh_stat.st_mtime_ns))
    plan = build_tdcs_leadfield_plan(
        prepared_subject=prepared,
        head_mesh_sha256=charm_identity.head_mesh_sha256,
        charm_runtime_manifest_sha256=charm_identity.manifest_sha256,
        cpus=cpus,
    )
    version = installed_simnibs_version()
    if version != PRODUCTION_PROFILE.simnibs_version:
        raise SimNIBSAdapterError(
            f"FEM profile 要求 SimNIBS {PRODUCTION_PROFILE.simnibs_version}，"
            f"实际为 {version}。"
        )
    try:
        from simnibs import sim_struct
    except ImportError as exc:
        raise SimNIBSAdapterError(
            "无法导入外部 simnibs.sim_struct.TDCSLEADFIELD。"
        ) from exc

    destination = Path(output_dir).expanduser()
    if not destination.name:
        raise ValueError("leadfield 输出目录不能是文件系统根目录")
    if _path_entry_exists(destination):
        raise FileExistsError(f"leadfield 输出已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.leadfield-staging-",
            dir=destination.parent,
        )
    )
    staging_run = staging_root / destination.name
    temporary_cap = staging_root / RUNTIME_CAP_NAME
    published = False
    try:
        _write_runtime_cap(temporary_cap, prepared)
        tdcs = sim_struct.TDCSLEADFIELD()
        configured = _configure_solver_conductivity(tdcs, sim_struct=sim_struct)
        tdcs.fnamehead = str(mesh)
        tdcs.eeg_cap = str(temporary_cap)
        tdcs.pathfem = str(staging_run)
        tdcs.interpolation = None
        tdcs.tissues = list(PRODUCTION_PROFILE.tissue_labels)
        tdcs.solver_options = PRODUCTION_PROFILE.solver_backend
        tdcs.run(cpus=plan.cpus, allow_multiple_runs=False, save_mat=True)
        if not staging_run.is_dir():
            raise SimNIBSAdapterError("SimNIBS 没有生成 leadfield 运行目录")

        hdf5 = _single_output(sorted(staging_run.rglob("*.hdf5")), "HDF5")
        solver_log = _single_output(
            sorted(staging_run.rglob("simnibs_simulation_*.log")),
            "solver log",
        )
        runtime_mat = _single_output(
            sorted(staging_run.rglob(SOLVER_RUNTIME_MAT_PATTERN)),
            "runtime MAT",
        )
        restored = _read_solver_runtime_object(runtime_mat)
        readback = _snapshot_solver_conductivity(
            restored,
            cond_type=sim_struct.COND,
            runtime_source="simnibs_simulation_mat_readback",
        )
        conductivity = {
            "schema_id": "solver_conductivity_verification_v1",
            "configured_runtime_object": configured,
            "matlab_readback": readback,
            "verification": "exact_match",
        }
        validate_solver_conductivity_verification(conductivity)
        hdf5_contract = validate_leadfield_hdf5(hdf5)
        if tuple(hdf5_contract["electrode_names_all"]) != prepared.electrode_names_all:
            raise FEMV1ContractError("生成的 HDF5 电极顺序与 prepared-subject 不一致")

        cap_path = staging_run / RUNTIME_CAP_NAME
        shutil.copyfile(temporary_cap, cap_path)
        manifest = {
            **plan.to_manifest(),
            "schema_id": "tdcs_leadfield_run_v1",
            "status": "complete",
            "side_effects": "new_output_directory",
            "simnibs_runtime_version": version,
            "head_mesh": {
                "path_policy": "absolute_path_omitted",
                "size_bytes": mesh_fingerprint[0],
            },
            "runtime_cap": _file_record(cap_path, root=staging_run),
            "leadfield_hdf5": _file_record(hdf5, root=staging_run),
            "solver_log": _file_record(solver_log, root=staging_run),
            "solver_runtime_mat": _file_record(runtime_mat, root=staging_run),
            "solver_conductivity": conductivity,
            "hdf5_contract": hdf5_contract,
            "publication": {
                "policy": "new_output_directory_only",
                "method": "sibling_staging_then_atomic_noreplace_rename",
                "partial_output_published_on_failure": False,
            },
        }
        _write_json(staging_run / LEADFIELD_RUN_MANIFEST_NAME, manifest)
        current_stat = mesh.stat()
        if (
            int(current_stat.st_size),
            int(current_stat.st_mtime_ns),
        ) != mesh_fingerprint:
            raise RuntimeError("head mesh 在 SimNIBS 运行期间发生变化")
        if _path_entry_exists(destination):
            raise FileExistsError(
                f"leadfield 输出在发布前出现，拒绝覆盖：{destination}"
            )
        _rename_noreplace(staging_run, destination)
        published = True
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    if not published:
        raise SimNIBSAdapterError("TDCSLEADFIELD 输出未发布")

    manifest_path = destination / LEADFIELD_RUN_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return LeadfieldRun(
        output_dir=destination,
        hdf5_path=destination / manifest["leadfield_hdf5"]["relative_path"],
        solver_log=destination / manifest["solver_log"]["relative_path"],
        solver_runtime_mat=destination
        / manifest["solver_runtime_mat"]["relative_path"],
        manifest_path=manifest_path,
        manifest=manifest,
    )


__all__ = [
    "LEADFIELD_RUN_MANIFEST_NAME",
    "LeadfieldRun",
    "TDCSLeadfieldPlan",
    "build_tdcs_leadfield_plan",
    "dry_run_tdcs_leadfield",
    "run_tdcs_leadfield",
    "validate_tdcs_leadfield_run",
]
