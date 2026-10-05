"""使用纯合成数据验证发布版 FEM 求解、导出、清单与事务。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from head2field.fem_labels import audit as fem_audit
from head2field.fem_labels import generation, leadfield_run
from head2field.fem_labels.cli import build_parser
from head2field.fem_labels.contracts import (
    PRODUCTION_PROFILE,
    STANDARD_76_ELECTRODE_NAMES,
    FEMV1ContractError,
    validate_leadfield_dataset_contract,
)
from head2field.fem_labels.generation import (
    build_fem_label_plan,
    load_fem_labels_v1,
)
from head2field.fem_labels.leadfield_run import LeadfieldRun
from head2field.preprocessing.charm_adapter import (
    CHARM_RUNTIME_MANIFEST_NAME,
    SegmentationAdapterError,
    validate_charm_runtime,
    write_charm_runtime_manifest,
)
from head2field.preprocessing.validation import PreparedSubjectV1


def _prepared(root: Path) -> PreparedSubjectV1:
    """Build a validated typed handle containing only two query points."""

    root.mkdir(exist_ok=True)
    images = root / "images"
    images.mkdir()
    canonical_t1 = images / "T1_1mm.nii.gz"
    prepared_tissues = images / "final_tissues.nii.gz"
    shape = (2, 2, 2)
    affine = np.eye(4, dtype=np.float64)
    tissues = np.zeros(shape, dtype=np.int16)
    tissues[0, 0, 0] = 1
    tissues[1, 1, 1] = 2
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), canonical_t1)
    nib.save(nib.Nifti1Image(tissues, affine), prepared_tissues)
    head_model = root.parent / f"{root.name}_head_model"
    head_model.mkdir()
    (head_model / "T1.nii.gz").write_bytes(canonical_t1.read_bytes())
    (head_model / "final_tissues.nii.gz").write_bytes(prepared_tissues.read_bytes())
    head_mesh = head_model / f"{root.name}.msh"
    head_mesh.write_bytes(b"synthetic head mesh")
    write_charm_runtime_manifest(
        canonical_t1=canonical_t1,
        head_model_dir=head_model,
        simnibs_version="4.6.0",
    )
    charm_identity = validate_charm_runtime(
        head_model,
        canonical_t1=canonical_t1,
        final_tissues=head_model / "final_tissues.nii.gz",
        head_mesh=head_mesh,
    )
    shape = (2, 2, 2)
    affine = np.eye(4, dtype=np.float64)
    coordinates = np.arange(76 * 3, dtype=np.float32).reshape(76, 3)
    tissues = np.zeros(shape, dtype=np.int16)
    tissues[0, 0, 0] = 1
    tissues[1, 1, 1] = 2
    query = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    manifest = {
        "files": {
            "voxels/gmwm_ijk.npy": {
                "sha256": "1" * 64,
                "size": 1,
            }
        },
        "head_model_provenance": charm_identity.to_prepared_provenance(),
    }
    return PreparedSubjectV1(
        subject_dir=root,
        manifest_path=root / "prepared_subject_v1.json",
        manifest=manifest,
        manifest_sha256="0" * 64,
        t1_1mm=np.zeros(shape, dtype=np.float32),
        tissue_labels_1mm=tissues,
        sigma_s_per_m_1mm=np.zeros(shape, dtype=np.float32),
        sigma_model_1mm=np.zeros(shape, dtype=np.float32),
        affine=affine,
        inverse_affine=affine,
        full_shape=shape,
        gmwm_ijk=query,
        electrode_names_all=STANDARD_76_ELECTRODE_NAMES,
        source_names=STANDARD_76_ELECTRODE_NAMES[1:],
        electrode_coords_mm=coordinates,
        source_coords_mm=coordinates[1:],
        reference_name="Oz",
        reference_index=0,
        reference_coord_mm=coordinates[0],
        head_model_provenance=charm_identity.to_prepared_provenance(),
    )


def _bound_mesh(prepared: PreparedSubjectV1) -> Path:
    """Return the unique CHARM head mesh bound to the test prepared subject."""

    return (
        prepared.subject_dir.parent
        / f"{prepared.subject_dir.name}_head_model"
        / f"{prepared.subject_dir.name}.msh"
    )


def _valid_attrs(prepared: PreparedSubjectV1) -> dict[str, object]:
    """Generate synthetic attributes satisfying the official HDF5 contract."""

    return {
        "current": "1A",
        "d_type": "element_data",
        "electrode_names": np.asarray(prepared.electrode_names_all, dtype=object),
        "electrode_pos": prepared.electrode_coords_mm,
        "field": "E",
        "reference_electrode": "Oz",
        "tissues": np.asarray([1, 2], dtype=np.int32),
        "units": "V/m",
    }


def _valid_contract(prepared: PreparedSubjectV1) -> dict[str, object]:
    """Run real pure-contract validation and return the adapter result."""

    contract = validate_leadfield_dataset_contract(
        shape=(75, 1, 3),
        attrs=_valid_attrs(prepared),
    )
    contract["dataset_path"] = "mesh_leadfield/leadfields/tdcs_leadfield"
    contract["mesh_element_count"] = 1
    return contract


def _solver_conductivity_record() -> dict[str, object]:
    """Build conductivity records matching the official pre-run object and MAT reload."""

    table = copy.deepcopy(PRODUCTION_PROFILE.to_manifest()["conductivity_table"])
    common = {
        "schema_id": "solver_conductivity_snapshot_v1",
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
        "solver_values_s_per_m": list(table["values"][1:]),
        "project_table": table,
    }
    configured = copy.deepcopy(common)
    configured["runtime_source"] = "simnibs.TDCSLEADFIELD.cond_before_run"
    readback = copy.deepcopy(common)
    readback["runtime_source"] = "simnibs_simulation_mat_readback"
    return {
        "schema_id": "solver_conductivity_verification_v1",
        "configured_runtime_object": configured,
        "matlab_readback": readback,
        "verification": "exact_match",
    }


def test_simnibs_46_conductivity_mat_round_trip(tmp_path: Path) -> None:
    """Use the official API to check the complete cond list and frozen-table MAT round trip."""

    simnibs = pytest.importorskip("simnibs")
    if simnibs.__version__ != "4.6.0":
        pytest.skip("该集成合同只针对 SimNIBS 4.6.0")
    from simnibs import sim_struct
    from simnibs.utils.matlab_read import read_mat

    tdcs = sim_struct.TDCSLEADFIELD()
    untouched = {
        tag: (tdcs.cond[tag - 1].name, tdcs.cond[tag - 1].value) for tag in (100, 500)
    }
    configured = leadfield_run._configure_solver_conductivity(
        tdcs,
        sim_struct=sim_struct,
    )
    assert len(tdcs.cond) == 1000
    assert {
        tag: (tdcs.cond[tag - 1].name, tdcs.cond[tag - 1].value) for tag in (100, 500)
    } == untouched

    runtime_mat = tmp_path / "simnibs_simulation_roundtrip.mat"
    sim_struct.save_matlab_sim_struct(tdcs, str(runtime_mat))
    restored = read_mat(str(runtime_mat))
    readback = leadfield_run._snapshot_solver_conductivity(
        restored,
        cond_type=sim_struct.COND,
        runtime_source="simnibs_simulation_mat_readback",
    )
    verification = {
        "schema_id": "solver_conductivity_verification_v1",
        "configured_runtime_object": configured,
        "matlab_readback": readback,
        "verification": "exact_match",
    }
    leadfield_run.validate_solver_conductivity_verification(
        verification,
        runtime_mat=runtime_mat,
        cond_type=sim_struct.COND,
    )
    assert len(restored.cond) == 1000
    assert configured["project_table"] == readback["project_table"]


def test_dry_run_plan_is_exactly_75_source_to_oz_and_has_no_side_effects(
    tmp_path: Path,
) -> None:
    """Pure planning must freeze source order and create no outputs."""

    plan = build_fem_label_plan(
        query_count=2,
        prepared_subject_manifest_sha256="0" * 64,
        query_order_sha256="1" * 64,
    )
    payload = plan.to_manifest()
    assert len(payload["labels"]) == 75
    assert payload["source_names"] == list(STANDARD_76_ELECTRODE_NAMES[1:])
    assert payload["labels"][0]["relative_path"] == "labels/Fp1_to_Oz_float32.npy"
    assert payload["labels"][-1]["relative_path"] == "labels/I2_to_Oz_float32.npy"
    assert payload["side_effects"] == "none"
    assert list(tmp_path.iterdir()) == []


def test_hdf5_contract_rejects_source_order_and_reference_drift(tmp_path: Path) -> None:
    """Reference or source-order drift must fail before interpolation."""

    prepared = _prepared(tmp_path / "prepared")
    attrs = _valid_attrs(prepared)
    names = list(prepared.electrode_names_all)
    names[1], names[2] = names[2], names[1]
    attrs["electrode_names"] = names
    with pytest.raises(FEMV1ContractError, match="标准 76"):
        validate_leadfield_dataset_contract(shape=(75, 1, 3), attrs=attrs)
    attrs = _valid_attrs(prepared)
    attrs["reference_electrode"] = "Cz"
    with pytest.raises(FEMV1ContractError, match="reference_electrode"):
        validate_leadfield_dataset_contract(shape=(75, 1, 3), attrs=attrs)


def test_tdcs_runtime_adapter_generates_cap_hdf5_log_and_redacted_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter path must invoke TDCSLEADFIELD and retain the complete run directory."""

    prepared = _prepared(tmp_path / "prepared")
    mesh = _bound_mesh(prepared)
    mesh_content = mesh.read_bytes()
    contract = _valid_contract(prepared)

    class FakeCOND:
        """Simulate official COND fields without performing scientific computation."""

        def __init__(self):
            self.name = None
            self.value = None
            self.descrip = ""
            self.distribution_type = None
            self.distribution_parameters = []

    class FakeTDCSLEADFIELD:
        """Simulate only external-API file boundaries, not FEM numerics."""

        last = None

        def __init__(self):
            self.cond = [FakeCOND() for _ in range(1000)]
            self.cond[99].name = "Electrode_rubber"
            self.cond[99].value = 29.4
            self.cond[499].name = "Saline"
            self.cond[499].value = 1.0
            self.anisotropy_type = "scalar"
            FakeTDCSLEADFIELD.last = self

        def run(
            self,
            *,
            cpus: int,
            allow_multiple_runs: bool,
            save_mat: bool,
        ) -> None:
            assert cpus == 2
            assert allow_multiple_runs is False
            assert save_mat is True
            assert len(self.cond) == 1000
            assert [item.name for item in self.cond[:10]] == [
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
            ]
            assert [item.value for item in self.cond[:10]] == list(
                PRODUCTION_PROFILE.to_manifest()["conductivity_table"]["values"][1:]
            )
            assert self.cond[99].value == 29.4
            assert self.cond[499].value == 1.0
            output = Path(self.pathfem)
            output.mkdir()
            (output / "synthetic.hdf5").write_bytes(b"synthetic hdf5")
            (output / "simnibs_simulation_synthetic.log").write_text(
                "synthetic solver log\n",
                encoding="utf-8",
            )
            (output / "simnibs_simulation_synthetic.mat").write_bytes(
                b"synthetic runtime mat"
            )

    simnibs = types.ModuleType("simnibs")
    simnibs.sim_struct = types.SimpleNamespace(
        COND=FakeCOND,
        TDCSLEADFIELD=FakeTDCSLEADFIELD,
    )
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setattr(
        leadfield_run,
        "load_prepared_subject_v1",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        leadfield_run,
        "installed_simnibs_version",
        lambda: "4.6.0",
    )
    monkeypatch.setattr(
        leadfield_run,
        "validate_leadfield_hdf5",
        lambda path: contract,
    )
    monkeypatch.setattr(
        fem_audit,
        "load_prepared_subject_v1",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        fem_audit,
        "validate_leadfield_hdf5",
        lambda path: contract,
    )
    monkeypatch.setattr(
        leadfield_run,
        "_read_solver_runtime_object",
        lambda path: FakeTDCSLEADFIELD.last,
    )
    output = tmp_path / "leadfield_run"
    run = leadfield_run.run_tdcs_leadfield(
        prepared_subject_dir=prepared.subject_dir,
        head_mesh=mesh,
        output_dir=output,
        cpus=2,
    )
    assert run.hdf5_path.is_file()
    assert run.solver_log.is_file()
    assert run.manifest["head_mesh"]["size_bytes"] == mesh.stat().st_size
    assert str(tmp_path.resolve()) not in json.dumps(run.manifest)
    assert FakeTDCSLEADFIELD.last.solver_options == "pardiso"
    assert FakeTDCSLEADFIELD.last.interpolation is None
    assert run.solver_runtime_mat.is_file()
    assert (
        run.manifest["solver_conductivity"]["matlab_readback"]["project_table"]
        == PRODUCTION_PROFILE.to_manifest()["conductivity_table"]
    )
    with (output / "standard_76_Oz_reference.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][0] == "ReferenceElectrode"
    assert rows[0][4] == "Oz"
    assert [row[4] for row in rows[1:]] == list(STANDARD_76_ELECTRODE_NAMES[1:])
    assert mesh.read_bytes() == mesh_content
    audit = leadfield_run.validate_tdcs_leadfield_run(
        prepared_subject_dir=prepared.subject_dir,
        leadfield_hdf5=run.hdf5_path,
        solver_log=run.solver_log,
        run_manifest=run.manifest_path,
    )
    assert audit["status"] == "valid"
    assert audit["source_count"] == 75

    baseline_manifest = run.manifest_path.read_bytes()
    baseline_payload = json.loads(baseline_manifest.decode("utf-8"))
    invalid_runs: list[dict[str, object]] = []
    missing_mat = copy.deepcopy(baseline_payload)
    del missing_mat["solver_runtime_mat"]
    invalid_runs.append(missing_mat)
    extra_top = copy.deepcopy(baseline_payload)
    extra_top["unexpected"] = True
    invalid_runs.append(extra_top)
    tampered_solver = copy.deepcopy(baseline_payload)
    tampered_solver["solver_conductivity"]["matlab_readback"]["solver_values_s_per_m"][
        0
    ] = 0.127
    invalid_runs.append(tampered_solver)
    extra_publication = copy.deepcopy(baseline_payload)
    extra_publication["publication"]["unexpected"] = True
    invalid_runs.append(extra_publication)
    for invalid in invalid_runs:
        run.manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(FEMV1ContractError):
            leadfield_run.validate_tdcs_leadfield_run(
                prepared_subject_dir=prepared.subject_dir,
                leadfield_hdf5=run.hdf5_path,
                solver_log=run.solver_log,
                run_manifest=run.manifest_path,
            )
    run.manifest_path.write_bytes(baseline_manifest)


def test_tdcs_version_mismatch_has_no_fallback_or_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External-version drift must fail closed without creating the final directory."""

    prepared = _prepared(tmp_path / "prepared")
    mesh = _bound_mesh(prepared)
    monkeypatch.setattr(
        leadfield_run,
        "load_prepared_subject_v1",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        leadfield_run,
        "installed_simnibs_version",
        lambda: "4.5.0",
    )
    output = tmp_path / "must_not_exist"
    with pytest.raises(Exception, match=r"要求 SimNIBS 4\.6\.0"):
        leadfield_run.run_tdcs_leadfield(
            prepared_subject_dir=prepared.subject_dir,
            head_mesh=mesh,
            output_dir=output,
        )
    assert not output.exists()


def test_fem_preflight_rejects_wrong_mesh_manifest_t1_and_tissues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FEM preflight must reject four identity changes along the prepared-to-CHARM hash chain."""

    prepared = _prepared(tmp_path / "prepared")
    mesh = _bound_mesh(prepared)
    monkeypatch.setattr(
        leadfield_run,
        "load_prepared_subject_v1",
        lambda *args, **kwargs: prepared,
    )
    plan = leadfield_run.dry_run_tdcs_leadfield(
        prepared_subject_dir=prepared.subject_dir,
        head_mesh=mesh,
    )
    assert plan.source_names == prepared.source_names

    wrong_mesh = mesh.parent / "wrong.msh"
    wrong_mesh.write_bytes(b"wrong mesh")
    with pytest.raises(SegmentationAdapterError, match=r"一个|唯一|不一致"):
        leadfield_run.dry_run_tdcs_leadfield(
            prepared_subject_dir=prepared.subject_dir,
            head_mesh=wrong_mesh,
        )
    wrong_mesh.unlink()

    runtime_manifest = mesh.parent / CHARM_RUNTIME_MANIFEST_NAME
    baseline_manifest = runtime_manifest.read_bytes()
    payload = json.loads(baseline_manifest.decode("utf-8"))
    payload["unexpected"] = True
    runtime_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SegmentationAdapterError, match="字段集合"):
        leadfield_run.dry_run_tdcs_leadfield(
            prepared_subject_dir=prepared.subject_dir,
            head_mesh=mesh,
        )
    runtime_manifest.write_bytes(baseline_manifest)

    canonical_t1 = prepared.subject_dir / "images" / "T1_1mm.nii.gz"
    canonical_bytes = canonical_t1.read_bytes()
    canonical_t1.write_bytes(canonical_bytes + b"tampered")
    with pytest.raises(SegmentationAdapterError, match="canonical T1"):
        leadfield_run.dry_run_tdcs_leadfield(
            prepared_subject_dir=prepared.subject_dir,
            head_mesh=mesh,
        )
    canonical_t1.write_bytes(canonical_bytes)

    tissues = prepared.subject_dir / "images" / "final_tissues.nii.gz"
    tissue_bytes = tissues.read_bytes()
    tissue_image = nib.load(tissues)
    changed_tissues = np.asarray(tissue_image.dataobj).copy()
    changed_tissues.flat[0] = 2 if changed_tissues.flat[0] == 1 else 1
    nib.save(nib.Nifti1Image(changed_tissues, tissue_image.affine), tissues)
    with pytest.raises(SegmentationAdapterError, match="final_tissues"):
        leadfield_run.dry_run_tdcs_leadfield(
            prepared_subject_dir=prepared.subject_dir,
            head_mesh=mesh,
        )
    tissues.write_bytes(tissue_bytes)


def test_hdf5_export_writes_and_reloads_all_75_labels_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic adapter outputs must pass full manifest, hash, and schema checks."""

    prepared = _prepared(tmp_path / "prepared")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    hdf5 = run_dir / "synthetic.hdf5"
    log = run_dir / "simnibs_simulation_synthetic.log"
    runtime_mat = run_dir / "simnibs_simulation_synthetic.mat"
    run_manifest = run_dir / "leadfield_run_v1.json"
    hdf5.write_bytes(b"synthetic hdf5")
    log.write_text("synthetic solver log\n", encoding="utf-8")
    runtime_mat.write_bytes(b"synthetic runtime mat")
    contract = _valid_contract(prepared)
    run_manifest.write_text(
        json.dumps(
            {
                "leadfield_hdf5": {
                    "relative_path": hdf5.name,
                    "size_bytes": hdf5.stat().st_size,
                    "sha256": hashlib.sha256(hdf5.read_bytes()).hexdigest(),
                },
                "solver_log": {
                    "relative_path": log.name,
                    "size_bytes": log.stat().st_size,
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                },
                "solver_runtime_mat": {
                    "relative_path": runtime_mat.name,
                    "size_bytes": runtime_mat.stat().st_size,
                    "sha256": hashlib.sha256(runtime_mat.read_bytes()).hexdigest(),
                },
                "solver_conductivity": _solver_conductivity_record(),
                "hdf5_contract": contract,
            }
        ),
        encoding="utf-8",
    )
    input_contents = {path: path.read_bytes() for path in (hdf5, log, runtime_mat)}
    monkeypatch.setattr(
        generation,
        "load_prepared_subject_v1",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(generation, "installed_simnibs_version", lambda: "4.6.0")
    monkeypatch.setattr(
        generation,
        "validate_tdcs_leadfield_run",
        lambda **kwargs: {"status": "valid"},
    )
    monkeypatch.setattr(generation, "build_base_mesh", lambda path: object())

    def read_field(path, *, source_index, scale=PRODUCTION_PROFILE.scale):
        del path, scale
        return np.full((1, 3), source_index, dtype=np.float32)

    def interpolate(*, base_mesh, element_field, reference_nifti, query_ijk, **kwargs):
        del base_mesh, reference_nifti, kwargs
        return np.full(
            (len(query_ijk), 3),
            float(element_field[0, 0]),
            dtype=np.float32,
        )

    monkeypatch.setattr(generation, "read_source_element_field", read_field)
    monkeypatch.setattr(
        generation,
        "interpolate_element_field_at_queries",
        interpolate,
    )
    output = tmp_path / "labels"
    result = generation.export_fem_labels_from_hdf5(
        prepared_subject_dir=prepared.subject_dir,
        leadfield_hdf5=hdf5,
        solver_log=log,
        leadfield_run_manifest=run_manifest,
        output_dir=output,
    )
    loaded = load_fem_labels_v1(result, prepared_subject=prepared)
    assert len(loaded.label_paths) == 75
    assert np.array_equal(
        loaded.load_source("Fp1", mmap_mode=None),
        np.zeros((2, 3), dtype=np.float32),
    )
    assert np.array_equal(
        loaded.load_source("I2", mmap_mode=None),
        np.full((2, 3), 74.0, dtype=np.float32),
    )
    for path, content in input_contents.items():
        assert path.read_bytes() == content

    audit = generation.validate_fem_artifacts(
        prepared_subject_dir=prepared.subject_dir,
        fem_labels_dir=output,
    )
    assert audit["status"] == "valid"
    assert audit["fem_labels"]["source_count"] == 75

    first_label = loaded.label_paths[0]
    original = np.load(first_label, allow_pickle=False)
    changed = original.copy()
    changed[0, 0] = 123.0
    np.save(first_label, changed, allow_pickle=False)
    standard_audit = generation.validate_fem_artifacts(
        prepared_subject_dir=prepared.subject_dir,
        fem_labels_dir=output,
    )
    assert standard_audit["integrity"] == "scientific_contract"
    changed[0, 0] = np.nan
    np.save(first_label, changed, allow_pickle=False)
    with pytest.raises(FEMV1ContractError, match="NaN|Inf"):
        generation.validate_fem_artifacts(
            prepared_subject_dir=prepared.subject_dir,
            fem_labels_dir=output,
        )
    np.save(first_label, original, allow_pickle=False)

    base_manifest = copy.deepcopy(dict(loaded.manifest))
    invalid_manifests: list[dict[str, object]] = []
    missing_top = copy.deepcopy(base_manifest)
    del missing_top["interpolation"]
    invalid_manifests.append(missing_top)
    extra_top = copy.deepcopy(base_manifest)
    extra_top["unexpected"] = True
    invalid_manifests.append(extra_top)
    missing_interpolation = copy.deepcopy(base_manifest)
    del missing_interpolation["interpolation"]["method"]
    invalid_manifests.append(missing_interpolation)
    extra_simnibs = copy.deepcopy(base_manifest)
    extra_simnibs["leadfield_provenance"]["simnibs"]["unexpected"] = True
    invalid_manifests.append(extra_simnibs)
    missing_runtime_mat = copy.deepcopy(base_manifest)
    del missing_runtime_mat["leadfield_provenance"]["solver_runtime_mat"]
    invalid_manifests.append(missing_runtime_mat)
    tampered_conductivity = copy.deepcopy(base_manifest)
    tampered_conductivity["leadfield_provenance"]["solver_conductivity"][
        "matlab_readback"
    ]["project_table"]["values"][1] = 0.127
    invalid_manifests.append(tampered_conductivity)
    extra_contract = copy.deepcopy(base_manifest)
    extra_contract["leadfield_provenance"]["leadfield_contract"]["unexpected"] = True
    invalid_manifests.append(extra_contract)
    extra_publication = copy.deepcopy(base_manifest)
    extra_publication["publication"]["unexpected"] = True
    invalid_manifests.append(extra_publication)
    extra_label = copy.deepcopy(base_manifest)
    extra_label["labels"][0]["unexpected"] = True
    invalid_manifests.append(extra_label)
    for invalid in invalid_manifests:
        with pytest.raises(FEMV1ContractError):
            generation.validate_fem_labels_manifest(
                output,
                manifest=invalid,
                prepared_subject=prepared,
            )

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas/fem_labels_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(dict(loaded.manifest))
    for invalid in invalid_manifests:
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid)


def test_complete_generate_composes_solver_and_export_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public generation must not become a renamed interface accepting only preexisting HDF5."""

    leadfield_dir = tmp_path / "leadfield"
    labels_dir = tmp_path / "labels"
    fake_run = LeadfieldRun(
        output_dir=leadfield_dir,
        hdf5_path=leadfield_dir / "run.hdf5",
        solver_log=leadfield_dir / "simnibs_simulation_run.log",
        solver_runtime_mat=leadfield_dir / "simnibs_simulation_run.mat",
        manifest_path=leadfield_dir / "leadfield_run_v1.json",
        manifest={},
    )
    calls: list[str] = []

    def run(**kwargs):
        calls.append("solve")
        assert kwargs["output_dir"].resolve() == leadfield_dir.resolve()
        return fake_run

    def export(**kwargs):
        calls.append("export")
        assert kwargs["leadfield_hdf5"] == fake_run.hdf5_path
        assert kwargs["solver_log"] == fake_run.solver_log
        assert kwargs["leadfield_run_manifest"] == fake_run.manifest_path
        return labels_dir

    monkeypatch.setattr(generation, "run_tdcs_leadfield", run)
    monkeypatch.setattr(generation, "export_fem_labels_from_hdf5", export)
    result = generation.generate_fem_labels(
        prepared_subject_dir=tmp_path / "prepared",
        head_mesh=tmp_path / "head.msh",
        leadfield_output_dir=leadfield_dir,
        labels_output_dir=labels_dir,
    )
    assert result == labels_dir
    assert calls == ["solve", "export"]


def test_fem_cli_has_plan_generate_validate_export_and_no_overwrite() -> None:
    """The CLI must expose generation and read-only validation without overwrite switches."""

    help_text = build_parser().format_help()
    assert "plan" in help_text
    assert "generate" in help_text
    assert "export" in help_text
    assert "validate" in help_text
    assert "--overwrite" not in help_text
    args = build_parser().parse_args(
        [
            "validate",
            "--prepared-subject-dir",
            "prepared",
            "--fem-labels-dir",
            "labels",
        ]
    )
    assert args.fem_labels_dir == Path("labels")
