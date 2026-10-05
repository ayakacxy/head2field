"""Tiny non-anatomical preprocessing inputs shared by release examples and unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .electrodes import STANDARD_76_ELECTRODE_NAMES
from .transaction import staged_output_directory


@dataclass(frozen=True)
class SyntheticPreparationInputs:
    """Four public files in the deterministic synthetic-input directory."""

    root: Path
    raw_t1: Path
    orientation_template: Path
    electrode_names_json: Path
    electrode_coords_mm_npy: Path


def _synthetic_t1(*, phase: float) -> nib.Nifti1Image:
    """Build a deidentified 32-voxel ellipsoidal intensity image."""

    shape = (32, 32, 32)
    grid = np.indices(shape, dtype=np.float32)
    radius = (
        ((grid[0] - 15.5) / 9.0) ** 2
        + ((grid[1] - 15.5) / 10.0) ** 2
        + ((grid[2] - 15.5) / 8.0) ** 2
    )
    data = np.zeros(shape, dtype=np.float32)
    foreground = radius <= 1.0
    data[foreground] = (
        np.float32(100.0 + phase)
        + np.float32(0.4) * grid[0][foreground]
        + np.float32(0.2) * grid[1][foreground]
    )
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -16.0],
            [0.0, 1.0, 0.0, -16.0],
            [0.0, 0.0, 1.0, -16.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return nib.Nifti1Image(data, affine)


def _synthetic_electrode_coordinates() -> np.ndarray:
    """Generate 76 fixed world-RAS points on a sphere of radius 90 mm."""

    count = len(STANDARD_76_ELECTRODE_NAMES)
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    azimuth = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    coordinates = np.column_stack(
        [radius_xy * np.cos(azimuth), radius_xy * np.sin(azimuth), z]
    )
    return np.asarray(90.0 * coordinates, dtype=np.float32)


def write_synthetic_preparation_inputs(output_dir: Path) -> SyntheticPreparationInputs:
    """Transactionally write reproducible small raw-T1, template, and fixed-cap inputs."""

    destination = Path(output_dir).expanduser()
    with staged_output_directory(destination) as staging:
        raw_t1 = staging / "synthetic_raw_t1.nii.gz"
        template = staging / "synthetic_orientation_template.nii.gz"
        names = staging / "standard_76_names.json"
        coordinates = staging / "standard_76_coords_mm.npy"
        nib.save(_synthetic_t1(phase=0.0), str(raw_t1))
        nib.save(_synthetic_t1(phase=5.0), str(template))
        names.write_text(
            json.dumps(
                list(STANDARD_76_ELECTRODE_NAMES),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        np.save(
            coordinates,
            _synthetic_electrode_coordinates(),
            allow_pickle=False,
        )
    return SyntheticPreparationInputs(
        root=destination,
        raw_t1=destination / "synthetic_raw_t1.nii.gz",
        orientation_template=destination / "synthetic_orientation_template.nii.gz",
        electrode_names_json=destination / "standard_76_names.json",
        electrode_coords_mm_npy=destination / "standard_76_coords_mm.npy",
    )


__all__ = ["SyntheticPreparationInputs", "write_synthetic_preparation_inputs"]
