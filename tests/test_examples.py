from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example(path: Path, module_name: str):
    """Load an uninstalled example script by file path."""

    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_preprocessing_example_generates_only_declared_synthetic_inputs(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    module = _load_example(
        root / "examples" / "generate_synthetic_preprocessing_inputs.py",
        "synthetic_preprocessing_example",
    )
    output = tmp_path / "preprocessing"

    assert module.main(["--output-dir", str(output)]) == 0
    assert sorted(path.name for path in output.iterdir()) == [
        "standard_76_coords_mm.npy",
        "standard_76_names.json",
        "synthetic_orientation_template.nii.gz",
        "synthetic_raw_t1.nii.gz",
    ]


def test_training_example_is_deterministic_and_group_disjoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    module = _load_example(
        root / "examples" / "generate_synthetic_training_dataset.py",
        "synthetic_training_example",
    )
    output = tmp_path / "training"

    assert module.main(["--output-dir", str(output), "--seed", "42"]) == 0
    manifest = output / "synthetic_training_dataset_v1.json"
    assert manifest.is_file()
    assert sorted(path.name for path in output.glob("*.npz")) == [
        "synthetic-train-a.npz",
        "synthetic-validation-b.npz",
    ]
