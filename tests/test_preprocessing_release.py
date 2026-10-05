"""Synthetic end-to-end and failure-contract tests for release preprocessing."""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from head2field.preprocessing import alignment, assembly, pipeline, validation
from head2field.preprocessing.charm_adapter import (
    CHARM_RUNTIME_MANIFEST_NAME,
    SegmentationAdapterError,
    validate_charm_runtime,
    write_charm_runtime_manifest,
)
from head2field.preprocessing.electrodes import (
    STANDARD_76_ELECTRODE_NAMES,
    load_standard_cap_from_head_model,
)
from head2field.preprocessing.synthetic import (
    write_synthetic_preparation_inputs,
)


def _input_bytes(path: Path) -> bytes:
    """读取合成小文件内容，验证处理过程保持输入不变。"""

    return path.read_bytes()


def _write_t1(path: Path, *, phase: float = 0.0) -> Path:
    """Write a small deidentified ellipsoidal T1."""

    shape = (32, 32, 32)
    grid = np.indices(shape, dtype=np.float32)
    mask = ((grid[0] - 15.5) / 8.0) ** 2 + ((grid[1] - 15.5) / 9.0) ** 2 + (
        (grid[2] - 15.5) / 7.0
    ) ** 2 <= 1.0
    data = np.zeros(shape, dtype=np.float32)
    data[mask] = 100.0 + phase + 0.2 * grid[0][mask]
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -16.0],
            [0.0, 1.0, 0.0, -16.0],
            [0.0, 0.0, 1.0, -16.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def small_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Reduce the test grid while retaining production strict-validation paths."""

    shape = (48, 48, 48)
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -24.0],
            [0.0, 1.0, 0.0, -24.0],
            [0.0, 0.0, 1.0, -24.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(alignment, "TARGET_SHAPE", shape)
    monkeypatch.setattr(alignment, "TARGET_AFFINE", affine)
    monkeypatch.setattr(assembly, "PREPARED_SUBJECT_SHAPE", shape)
    monkeypatch.setattr(
        assembly,
        "PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM",
        tuple(tuple(float(value) for value in row) for row in affine),
    )

    def build(root: Path, *, head_model_provenance):
        return validation.build_prepared_subject_manifest(
            root,
            head_model_provenance=head_model_provenance,
            expected_shape=shape,
            expected_affine=affine,
        )

    def validate(root: Path, *, manifest=None, ):
        return validation.validate_prepared_subject_manifest(
            root,
            manifest=manifest,

            expected_shape=shape,
            expected_affine=affine,
        )

    def load(root: Path):
        return validation.load_prepared_subject_v1(
            root,

            expected_shape=shape,
            expected_affine=affine,
        )

    monkeypatch.setattr(assembly, "build_prepared_subject_manifest", build)
    monkeypatch.setattr(assembly, "validate_prepared_subject_manifest", validate)
    monkeypatch.setattr(assembly, "load_prepared_subject_v1", load)
    return shape, affine


def test_lightweight_import_succeeds_when_scipy_is_unavailable() -> None:
    """Minimal inference imports must not depend on preprocessing SciPy modules."""

    code = r"""
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "scipy" or name.startswith("scipy."):
        raise ImportError("synthetic blocked scipy")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import head2field
assert "head2field.preprocessing.alignment" not in sys.modules
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_synthetic_input_builder_is_small_deterministic_and_noreplace(
    tmp_path: Path,
) -> None:
    """Public synthetic inputs must be deidentified and refuse existing directories."""

    first = write_synthetic_preparation_inputs(tmp_path / "synthetic_inputs")
    assert nib.load(str(first.raw_t1)).shape == (32, 32, 32)
    assert json.loads(first.electrode_names_json.read_text(encoding="utf-8")) == list(
        STANDARD_76_ELECTRODE_NAMES
    )
    assert np.load(first.electrode_coords_mm_npy).shape == (76, 3)
    with pytest.raises(FileExistsError):
        write_synthetic_preparation_inputs(first.root)


def test_non_hcp_profile_calls_six_dof_estimator_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_profile,
) -> None:
    """The non-HCP branch must perform exactly one six-degree-of-freedom rigid estimation."""

    del small_profile
    raw = _write_t1(tmp_path / "raw.nii.gz")
    template = _write_t1(tmp_path / "template.nii.gz", phase=5.0)
    before = _input_bytes(raw)
    calls = {"count": 0}

    def estimate(source_image, source, template_image, template_data, max_samples):
        del source, template_image, template_data, max_samples
        calls["count"] += 1
        transform, center = pipeline._hcp_centering_transform(source_image)
        result = SimpleNamespace(
            success=True,
            x=np.zeros(6, dtype=np.float64),
            fun=-0.5,
            message="synthetic success",
        )
        return transform, result, center

    monkeypatch.setattr(alignment, "estimate_rigid", estimate)
    output = tmp_path / "canonical"
    manifest = pipeline.preprocess_t1(
        input_t1=raw,
        output_dir=output,
        profile=pipeline.NON_HCP_RAW_PROFILE,
        template=template,
        minimum_margin_voxels=1,
    )
    assert calls["count"] == 1
    assert manifest["rigid_alignment"]["degrees_of_freedom"] == 6
    assert _input_bytes(raw) == before
    with pytest.raises(FileExistsError):
        pipeline.preprocess_t1(
            input_t1=raw,
            output_dir=output,
            profile=pipeline.NON_HCP_RAW_PROFILE,
            template=template,
        )


def test_complete_prepare_orchestrates_hcp_charm_sigma_cap_and_c_order_queries(
    tmp_path: Path,
    small_profile,
) -> None:
    """The top-level HCP branch must orchestrate T1 through complete prepared-subject output."""

    shape, affine = small_profile
    raw = _write_t1(tmp_path / "hcp.nii.gz")
    source_inputs = write_synthetic_preparation_inputs(tmp_path / "cap_inputs")
    input_contents = {raw: _input_bytes(raw)}

    class SyntheticSegmentation:
        """Generate aligned synthetic labels for orchestration tests without simulating CHARM numerics."""

        calls = 0

        def run(self, *, canonical_t1: Path, output_dir: Path) -> Path:
            self.calls += 1
            output_dir.mkdir()
            (output_dir / "T1.nii.gz").write_bytes(canonical_t1.read_bytes())
            tissues = np.zeros(shape, dtype=np.int16)
            tissues[18:24, 18:24, 18:24] = 1
            tissues[24:30, 18:24, 18:24] = 2
            path = output_dir / "final_tissues.nii.gz"
            nib.save(nib.Nifti1Image(tissues, affine), str(path))
            eeg_dir = output_dir / "eeg_positions"
            eeg_dir.mkdir()
            cap_path = eeg_dir / "EEG10-10_UI_Jurak_2007.csv"
            coordinates = np.load(source_inputs.electrode_coords_mm_npy)
            with cap_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                # Write in reverse order deliberately to verify fixed-inventory reordering rather than CSV trust.
                for index in reversed(range(len(STANDARD_76_ELECTRODE_NAMES))):
                    name = STANDARD_76_ELECTRODE_NAMES[index]
                    coordinate = coordinates[index]
                    row_type = "ReferenceElectrode" if name == "Oz" else "Electrode"
                    writer.writerow([row_type, *coordinate.tolist(), name])
            (output_dir / "synthetic_head.msh").write_bytes(b"synthetic head mesh")
            write_charm_runtime_manifest(
                canonical_t1=canonical_t1,
                head_model_dir=output_dir,
                simnibs_version="4.6.0",
            )
            assert canonical_t1.is_file()
            return path

    adapter = SyntheticSegmentation()
    result = assembly.prepare_subject(
        input_t1=raw,
        profile=pipeline.HCP_ACPC_PROFILE,
        canonical_output_dir=tmp_path / "canonical_stage",
        head_model_output_dir=tmp_path / "head_model",
        prepared_output_dir=tmp_path / "prepared",
        segmentation_adapter=adapter,
        minimum_margin_voxels=1,
    )
    assert adapter.calls == 1
    prepared = validation.load_prepared_subject_v1(
        result.prepared_output_dir,
        expected_shape=shape,
        expected_affine=affine,
    )
    expected_queries = np.argwhere(np.isin(prepared.tissue_labels_1mm, (1, 2)))
    assert np.array_equal(prepared.gmwm_ijk, expected_queries)
    assert prepared.electrode_names_all == STANDARD_76_ELECTRODE_NAMES
    expected_sigma = np.asarray(
        [0.0, 0.126, 0.275],
        dtype=np.float32,
    )
    assert set(np.unique(prepared.sigma_s_per_m_1mm)) == set(expected_sigma)
    assert result.head_model_output_dir.joinpath("final_tissues.nii.gz").is_file()
    charm_identity = validate_charm_runtime(
        result.head_model_output_dir,
        canonical_t1=result.canonical_output_dir / "images" / "T1_1mm.nii.gz",
        final_tissues=result.head_model_output_dir / "final_tissues.nii.gz",
        head_mesh=result.head_model_output_dir / "synthetic_head.msh",
    )
    assert prepared.head_model_provenance == charm_identity.to_prepared_provenance()
    for path, content in input_contents.items():
        assert _input_bytes(path) == content

    base_manifest = copy.deepcopy(dict(prepared.manifest))
    invalid_manifests: list[dict[str, object]] = []
    missing_top = copy.deepcopy(base_manifest)
    del missing_top["head_model_provenance"]
    invalid_manifests.append(missing_top)
    extra_top = copy.deepcopy(base_manifest)
    extra_top["unexpected"] = True
    invalid_manifests.append(extra_top)
    missing_nested = copy.deepcopy(base_manifest)
    del missing_nested["head_model_provenance"]["simnibs_version"]
    invalid_manifests.append(missing_nested)
    extra_nested = copy.deepcopy(base_manifest)
    extra_nested["model_input_transform"]["unexpected"] = True
    invalid_manifests.append(extra_nested)
    extra_file_field = copy.deepcopy(base_manifest)
    extra_file_field["files"]["images/T1_1mm.nii.gz"]["unexpected"] = True
    invalid_manifests.append(extra_file_field)
    tampered_nested = copy.deepcopy(base_manifest)
    tampered_nested["head_model_provenance"]["simnibs_version"] = "4.5.0"
    invalid_manifests.append(tampered_nested)
    for invalid in invalid_manifests:
        with pytest.raises((TypeError, ValueError)):
            validation.validate_prepared_subject_manifest(
                result.prepared_output_dir,
                manifest=invalid,
                expected_shape=shape,
                expected_affine=affine,
            )

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas/prepared_subject_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    def production_grid(payload: dict[str, object]) -> dict[str, object]:
        """Replace only the small fixture's grid declaration with official schema constants."""

        candidate = copy.deepcopy(payload)
        candidate["shape"] = list(validation.PREPARED_SUBJECT_SHAPE)
        candidate["affine_ijk_to_ras_mm"] = [
            list(row) for row in validation.PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM
        ]
        return candidate

    validator.validate(production_grid(base_manifest))
    for invalid in invalid_manifests:
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(production_grid(invalid))


def test_assembly_failure_does_not_publish_partial_directory(
    tmp_path: Path,
    small_profile,
) -> None:
    """Invalid cap order must fail before creating the final prepared directory."""

    shape, affine = small_profile
    canonical = np.zeros(shape, dtype=np.float32)
    canonical[16:32, 16:32, 16:32] = 100.0
    tissues = np.zeros(shape, dtype=np.int16)
    tissues[20:24, 20:24, 20:24] = 1
    t1_path = tmp_path / "canonical.nii.gz"
    head_model = tmp_path / "head_model"
    head_model.mkdir()
    tissue_path = head_model / "final_tissues.nii.gz"
    nib.save(nib.Nifti1Image(canonical, affine), str(t1_path))
    nib.save(nib.Nifti1Image(tissues, affine), str(tissue_path))
    (head_model / "T1.nii.gz").write_bytes(t1_path.read_bytes())
    (head_model / "synthetic_head.msh").write_bytes(b"synthetic head mesh")
    write_charm_runtime_manifest(
        canonical_t1=t1_path,
        head_model_dir=head_model,
        simnibs_version="4.6.0",
    )
    inputs = write_synthetic_preparation_inputs(tmp_path / "inputs")
    names = list(STANDARD_76_ELECTRODE_NAMES)
    names[0], names[1] = names[1], names[0]
    bad_names = tmp_path / "bad_names.json"
    bad_names.write_text(json.dumps(names), encoding="utf-8")
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="固定 76"):
        assembly.assemble_prepared_subject(
            canonical_t1=t1_path,
            final_tissues=tissue_path,
            electrode_names_json=bad_names,
            electrode_coords_mm_npy=inputs.electrode_coords_mm_npy,
            head_model_dir=head_model,
            output_dir=output,
        )
    assert not output.exists()


def test_default_charm_cap_rejects_missing_standard_electrode(tmp_path: Path) -> None:
    """The default CHARM cap must not substitute approximate or incomplete electrodes for production v1."""

    head_model = tmp_path / "head_model"
    eeg_dir = head_model / "eeg_positions"
    eeg_dir.mkdir(parents=True)
    cap = eeg_dir / "EEG10-10_UI_Jurak_2007.csv"
    with cap.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for index, name in enumerate(STANDARD_76_ELECTRODE_NAMES[:-1]):
            row_type = "ReferenceElectrode" if index == 0 else "Electrode"
            writer.writerow([row_type, float(index), 0.0, 0.0, name])
    with pytest.raises(ValueError, match=r"missing=.*I2"):
        load_standard_cap_from_head_model(head_model)


def test_charm_runtime_rejects_missing_extra_and_tampered_identity(
    tmp_path: Path,
) -> None:
    """CHARM identity manifests must use closed fields and bind T1, tissues, and a unique mesh."""

    canonical = tmp_path / "canonical.nii.gz"
    canonical.write_bytes(b"synthetic canonical")
    head_model = tmp_path / "head_model"
    head_model.mkdir()
    tissues = head_model / "final_tissues.nii.gz"
    mesh = head_model / "synthetic.msh"
    (head_model / "T1.nii.gz").write_bytes(canonical.read_bytes())
    tissues.write_bytes(b"synthetic tissues")
    mesh.write_bytes(b"synthetic mesh")
    manifest_path = write_charm_runtime_manifest(
        canonical_t1=canonical,
        head_model_dir=head_model,
        simnibs_version="4.6.0",
    )
    validate_charm_runtime(
        head_model,
        canonical_t1=canonical,
        final_tissues=tissues,
        head_mesh=mesh,
    )
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants: list[dict[str, object]] = []
    missing = copy.deepcopy(baseline)
    del missing["head_mesh"]
    variants.append(missing)
    extra = copy.deepcopy(baseline)
    extra["unexpected"] = True
    variants.append(extra)
    nested_extra = copy.deepcopy(baseline)
    nested_extra["final_tissues"]["unexpected"] = True
    variants.append(nested_extra)
    wrong_size = copy.deepcopy(baseline)
    wrong_size["head_mesh"]["size_bytes"] += 1
    variants.append(wrong_size)
    for variant in variants:
        manifest_path.write_text(
            json.dumps(variant, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(SegmentationAdapterError):
            validate_charm_runtime(
                head_model,
                canonical_t1=canonical,
                final_tissues=tissues,
                head_mesh=mesh,
            )
    manifest_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert manifest_path.name == CHARM_RUNTIME_MANIFEST_NAME
