"""Generate optional target caches and stable-energy statistics from validated FEM labels."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from head2field.fem_labels import load_fem_labels_v1
from head2field.preprocessing.validation import load_prepared_subject_v1

from .contracts import (
    DATA_LOSS_DENOMINATOR_RULE_ID,
    DATA_LOSS_RADIAL_EDGES_MM,
)
from .dataset import LOSS_STATISTICS_SCHEMA_ID

TRAINING_CACHE_SCHEMA_ID = "training_target_cache_v1"


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _staging_directory(destination: Path) -> Path:
    resolved = Path(destination).resolve(strict=False)
    if resolved.exists():
        raise FileExistsError(f"输出目录必须尚不存在: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    staging = resolved.with_name(f".{resolved.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(exist_ok=False)
    return staging


def _promote(staging: Path, destination: Path) -> None:
    resolved = Path(destination).resolve(strict=False)
    if resolved.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录: {resolved}")
    os.rename(staging, resolved)
    _fsync_directory(resolved.parent)


def _cleanup_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for child in staging.iterdir():
        if child.is_dir():
            _cleanup_staging(child)
        else:
            child.unlink()
    staging.rmdir()


def prepare_training_cache(
    *,
    prepared_subject_dir: Path,
    fem_labels_dir: Path,
    output_dir: Path,
) -> Path:
    """Losslessly stack 75 labels into a ``[query,source,component]`` mmap cache."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    fem = load_fem_labels_v1(
        fem_labels_dir,
        prepared_subject=prepared,
    )
    destination = Path(output_dir).resolve(strict=False)
    staging = _staging_directory(destination)
    try:
        query_count = len(prepared.gmwm_ijk)
        cache_path = staging / "targets_query_source_component_float32.npy"
        target = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(query_count, len(fem.source_names), 3),
        )
        for source_index, source_name in enumerate(fem.source_names):
            target[:, source_index, :] = fem.load_source(source_name, mmap_mode="r")
        target.flush()
        del target
        _fsync_file(cache_path)
        manifest = {
            "schema_id": TRAINING_CACHE_SCHEMA_ID,
            "query_count": query_count,
            "reference_electrode": "Oz",
            "source_names": list(fem.source_names),
            "axis_order": ["query", "source", "component"],
            "component_order": ["Ex", "Ey", "Ez"],
            "dtype": "float32",
            "unit": "V/m",
            "arrays": {
                "relative_path": cache_path.name,
                "size_bytes": cache_path.stat().st_size,
                "shape": [query_count, len(fem.source_names), 3],
            },
        }
        manifest_path = staging / "training_target_cache_v1.json"
        _write_json(manifest_path, manifest)
        _fsync_directory(staging)
        _promote(staging, destination)
    finally:
        _cleanup_staging(staging)
    return destination / "training_target_cache_v1.json"


def _ijk_to_world(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(ijk, dtype=np.float64)
    return (points @ affine[:3, :3].T + affine[:3, 3]).astype(np.float32)


def _label_energies(
    label: np.ndarray,
    *,
    query_mm: np.ndarray,
    source_mm: np.ndarray,
    reference_mm: np.ndarray,
    radial_edges_mm: tuple[float, float],
    peak_quantile: float,
    chunk_size: int,
) -> tuple[float, np.ndarray, float, float]:
    target_squared = np.empty(len(query_mm), dtype=np.float32)
    radial_energy = np.zeros(3, dtype=np.float64)
    lower, upper = radial_edges_mm
    for start in range(0, len(query_mm), chunk_size):
        end = min(start + chunk_size, len(query_mm))
        block = np.asarray(label[start:end], dtype=np.float32)
        energy = np.einsum("ij,ij->i", block, block, optimize=True)
        target_squared[start:end] = energy
        query = query_mm[start:end]
        remote = np.minimum(
            np.linalg.norm(query - source_mm[None], axis=-1),
            np.linalg.norm(query - reference_mm[None], axis=-1),
        )
        masks = (
            remote < lower,
            (remote >= lower) & (remote < upper),
            remote >= upper,
        )
        for index, mask in enumerate(masks):
            radial_energy[index] += float(energy[mask].sum(dtype=np.float64))
    threshold = float(np.quantile(target_squared, peak_quantile))
    peak = float(target_squared[target_squared >= threshold].sum(dtype=np.float64))
    return (
        float(target_squared.sum(dtype=np.float64)),
        radial_energy,
        threshold,
        peak,
    )


def _positive_median(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        positive = values[values > 0.0]
        return np.asarray(float(np.median(positive)) if positive.size else 0.0)
    result: list[float] = []
    for index in range(values.shape[axis]):
        selected = np.take(values, index, axis=axis)
        positive = selected[selected > 0.0]
        result.append(float(np.median(positive)) if positive.size else 0.0)
    return np.asarray(result, dtype=np.float64)


def prepare_loss_statistics(
    *,
    prepared_subject_dir: Path,
    fem_labels_dir: Path,
    output_dir: Path,
    chunk_size: int = 65_536,
) -> Path:
    """Stream raw labels to generate frozen N128 GP stable-energy statistics."""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    fem = load_fem_labels_v1(
        fem_labels_dir,
        prepared_subject=prepared,
    )
    query_mm = _ijk_to_world(prepared.gmwm_ijk, prepared.affine)
    source_mm = prepared.source_coords_mm
    reference_mm = prepared.reference_coord_mm
    radial_edges = DATA_LOSS_RADIAL_EDGES_MM
    peak_quantile = 0.90
    pair_weight_alpha = 1.0
    pair_weight_sigma_mm = 40.0
    distance = np.linalg.norm(source_mm - reference_mm[None], axis=-1)
    pair_weights = 1.0 + pair_weight_alpha * np.exp(
        -0.5 * np.square(distance / pair_weight_sigma_mm)
    )
    pair_weights /= max(float(pair_weights.mean()), 1e-12)
    count = len(fem.source_names)
    global_energy = np.zeros(count, dtype=np.float64)
    radial_energy = np.zeros((count, 3), dtype=np.float64)
    threshold = np.zeros(count, dtype=np.float64)
    peak_energy = np.zeros(count, dtype=np.float64)
    for index, source_name in enumerate(fem.source_names):
        (
            global_energy[index],
            radial_energy[index],
            threshold[index],
            peak_energy[index],
        ) = _label_energies(
            fem.load_source(source_name, mmap_mode="r"),
            query_mm=query_mm,
            source_mm=source_mm[index],
            reference_mm=reference_mm,
            radial_edges_mm=radial_edges,
            peak_quantile=peak_quantile,
            chunk_size=chunk_size,
        )

    destination = Path(output_dir).resolve(strict=False)
    staging = _staging_directory(destination)
    try:
        arrays_path = staging / "statistics.npz"
        np.savez_compressed(
            arrays_path,
            source_names=np.asarray(fem.source_names),
            pair_weights=pair_weights.astype(np.float32),
            global_energy=global_energy,
            radial_energy=radial_energy,
            peak_threshold_squared=threshold,
            peak_energy=peak_energy,
            global_median_energy=_positive_median(global_energy),
            radial_median_energy=_positive_median(radial_energy, axis=1),
            peak_median_energy=_positive_median(peak_energy),
            radial_edges_mm=np.asarray(radial_edges, dtype=np.float32),
            query_count=np.asarray(len(query_mm), dtype=np.int64),
            query_ijk=prepared.gmwm_ijk,
            affine_ijk_to_ras_mm=prepared.affine,
            source_coords_mm=prepared.source_coords_mm,
            reference_coord_mm=prepared.reference_coord_mm,
        )
        _fsync_file(arrays_path)
        payload = {
            "schema_id": LOSS_STATISTICS_SCHEMA_ID,
            "source_names": list(fem.source_names),
            "query_count": len(query_mm),
            "denominator_rule_id": DATA_LOSS_DENOMINATOR_RULE_ID,
            "configuration": {
                "radial_edges_mm": list(radial_edges),
                "pair_weight_alpha": pair_weight_alpha,
                "pair_weight_sigma_mm": pair_weight_sigma_mm,
                "peak_quantile": peak_quantile,
            },
            "arrays": {
                "relative_path": arrays_path.name,
                "size_bytes": arrays_path.stat().st_size,
            },
        }
        manifest_path = staging / "training_loss_statistics_v2.json"
        _write_json(manifest_path, payload)
        _fsync_directory(staging)
        _promote(staging, destination)
    finally:
        _cleanup_staging(staging)
    return destination / "training_loss_statistics_v2.json"


__all__ = [
    "TRAINING_CACHE_SCHEMA_ID",
    "prepare_loss_statistics",
    "prepare_training_cache",
]
