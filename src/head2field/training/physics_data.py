"""Leakage-free raw, train-fitted statistics, and final-pool workflow for GP primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from head2field.fem_labels import load_fem_labels_v1
from head2field.preprocessing.validation import load_prepared_subject_v1

from .cache import (
    _cleanup_staging,
    _fsync_directory,
    _fsync_file,
    _promote,
    _staging_directory,
    _write_json,
)
from .contracts import (
    PHYSICS_DENOMINATOR_RULE_ID,
    PHYSICS_OPERATOR_VERSION,
)
from .dataset import PHYSICS_POOL_SCHEMA_ID
from .physics import (
    circulation_raw_and_scale,
    flux_raw_and_scale,
    interface_raw_and_scale,
)

PHYSICS_RAW_POOL_SCHEMA_ID = "training_physics_raw_pool_v1"
PHYSICS_STATISTICS_SCHEMA_ID = "training_physics_statistics_v1"
PRIMITIVES_PER_OPERATOR = 512


def _lookup_rows(gmwm_ijk: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    lookup = np.full(shape, -1, dtype=np.int32)
    rows = np.asarray(gmwm_ijk, dtype=np.int64)
    lookup[tuple(rows.T)] = np.arange(len(rows), dtype=np.int32)
    return lookup


def _select_rows(
    values: np.ndarray,
    count: int,
    rng: np.random.Generator,
    role: str,
) -> np.ndarray:
    if len(values) < count:
        raise ValueError(
            f"{role} 可用 primitive 仅 {len(values)}，正式 v1 需要至少 {count}"
        )
    indices = np.sort(rng.choice(len(values), size=count, replace=False))
    return values[indices]


def _circulation_geometry(
    tissue: np.ndarray,
    lookup: np.ndarray,
    affine: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates: list[np.ndarray] = []
    candidate_planes: list[np.ndarray] = []
    planes = ((0, 1), (0, 2), (1, 2))
    shape = np.asarray(tissue.shape)
    for label in (1, 2):
        for plane_index, (axis_a, axis_b) in enumerate(planes):
            valid_shape = shape.copy()
            valid_shape[axis_a] -= 1
            valid_shape[axis_b] -= 1
            base_slices = tuple(slice(0, int(value)) for value in valid_shape)
            valid = np.ones(tuple(valid_shape), dtype=bool)
            for delta_a in (0, 1):
                for delta_b in (0, 1):
                    slices = list(base_slices)
                    slices[axis_a] = slice(delta_a, delta_a + int(valid_shape[axis_a]))
                    slices[axis_b] = slice(delta_b, delta_b + int(valid_shape[axis_b]))
                    valid &= tissue[tuple(slices)] == label
            starts = np.argwhere(valid).astype(np.int32)
            candidates.append(starts)
            candidate_planes.append(np.full(len(starts), plane_index, dtype=np.int8))
    starts_all = np.concatenate(candidates)
    planes_all = np.concatenate(candidate_planes)
    selected_indices = (
        np.sort(rng.choice(len(starts_all), size=count, replace=False))
        if len(starts_all) >= count
        else np.empty(0, dtype=np.int64)
    )
    if len(selected_indices) != count:
        raise ValueError(
            f"circulation 可用 primitive 仅 {len(starts_all)}，正式 v1 需要 {count}"
        )
    starts = starts_all[selected_indices]
    plane_ids = planes_all[selected_indices]
    point_rows = np.empty((count, 4), dtype=np.int64)
    edge_vectors = np.empty((count, 4, 3), dtype=np.float32)
    linear_m = np.asarray(affine, dtype=np.float64)[:3, :3] / 1000.0
    for primitive_index, (start, plane_id) in enumerate(
        zip(starts, plane_ids, strict=True)
    ):
        axis_a, axis_b = planes[int(plane_id)]
        offsets = np.zeros((4, 3), dtype=np.int32)
        offsets[1, axis_a] = 1
        offsets[2, axis_a] = 1
        offsets[2, axis_b] = 1
        offsets[3, axis_b] = 1
        ijk = start[None] + offsets
        rows = lookup[tuple(ijk.T)]
        if np.any(rows < 0):
            raise AssertionError("circulation 点无法映射 gmwm row")
        point_rows[primitive_index] = rows
        world = ijk.astype(np.float64) @ linear_m.T
        edge_vectors[primitive_index] = (np.roll(world, -1, axis=0) - world).astype(
            np.float32
        )
    return point_rows, edge_vectors, np.ones((count, 4), dtype=bool)


def _positive_axis_normal(linear: np.ndarray, axis: int) -> np.ndarray:
    other = [value for value in range(3) if value != axis]
    normal = np.cross(linear[:, other[0]], linear[:, other[1]])
    if float(np.dot(normal, linear[:, axis])) < 0.0:
        normal = -normal
    return normal / max(float(np.linalg.norm(normal)), 1e-20)


def _interface_geometry(
    tissue: np.ndarray,
    lookup: np.ndarray,
    affine: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    linear = np.asarray(affine, dtype=np.float64)[:3, :3]
    for axis in range(3):
        low_slices = [slice(None)] * 3
        high_slices = [slice(None)] * 3
        low_slices[axis] = slice(0, -1)
        high_slices[axis] = slice(1, None)
        low = tissue[tuple(low_slices)]
        high = tissue[tuple(high_slices)]
        valid = ((low == 1) & (high == 2)) | ((low == 2) & (high == 1))
        positions = np.argwhere(valid).astype(np.int32)
        high_positions = positions.copy()
        high_positions[:, axis] += 1
        low_labels = tissue[tuple(positions.T)]
        low_rows = lookup[tuple(positions.T)]
        high_rows = lookup[tuple(high_positions.T)]
        wm_rows = np.where(low_labels == 1, low_rows, high_rows)
        gm_rows = np.where(low_labels == 2, low_rows, high_rows)
        records.append(np.column_stack([wm_rows, gm_rows]))
        positive = _positive_axis_normal(linear, axis)
        sign = np.where(low_labels == 1, 1.0, -1.0)
        normals.append((sign[:, None] * positive[None]).astype(np.float32))
    pairs_all = np.concatenate(records).astype(np.int64)
    normals_all = np.concatenate(normals).astype(np.float32)
    if len(pairs_all) < count:
        raise ValueError(
            f"interface 可用 primitive 仅 {len(pairs_all)}，正式 v1 需要 {count}"
        )
    indices = np.sort(rng.choice(len(pairs_all), size=count, replace=False))
    return pairs_all[indices, 0], pairs_all[indices, 1], normals_all[indices]


def _flux_geometry(
    tissue: np.ndarray,
    lookup: np.ndarray,
    affine: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates: list[np.ndarray] = []
    inner_shape = tuple(value - 2 for value in tissue.shape)
    for label in (1, 2):
        valid = np.ones(inner_shape, dtype=bool)
        for delta_i in (-1, 0, 1):
            for delta_j in (-1, 0, 1):
                for delta_k in (-1, 0, 1):
                    valid &= (
                        tissue[
                            1 + delta_i : tissue.shape[0] - 1 + delta_i,
                            1 + delta_j : tissue.shape[1] - 1 + delta_j,
                            1 + delta_k : tissue.shape[2] - 1 + delta_k,
                        ]
                        == label
                    )
        centers = np.argwhere(valid).astype(np.int32) + 1
        candidates.append(centers)
    centers_all = np.concatenate(candidates)
    centers = _select_rows(centers_all, count, rng, "flux")
    # Reidentify tissues at selected centers rather than using indices from concatenated choices.
    selected_labels = tissue[tuple(centers.T)]
    face_rows = np.empty((count, 6), dtype=np.int64)
    column = 0
    for axis in range(3):
        for sign in (-1, 1):
            face = centers.copy()
            face[:, axis] += sign
            face_rows[:, column] = lookup[tuple(face.T)]
            column += 1
    if np.any(face_rows < 0):
        raise AssertionError("flux face center 无法映射 gmwm row")
    linear_m = np.asarray(affine, dtype=np.float64)[:3, :3] / 1000.0
    area_vectors = np.empty((count, 6, 3), dtype=np.float32)
    column = 0
    for axis in range(3):
        other = [value for value in range(3) if value != axis]
        positive = np.cross(linear_m[:, other[0]], linear_m[:, other[1]])
        if float(np.dot(positive, linear_m[:, axis])) < 0.0:
            positive = -positive
        area_vectors[:, column] = -positive
        area_vectors[:, column + 1] = positive
        column += 2
    conductivity = np.where(selected_labels == 1, 0.126, 0.275).astype(np.float32)
    return face_rows, area_vectors, conductivity


def _stable_union(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[int, int]]:
    seen: dict[int, int] = {}
    values: list[int] = []
    for array in arrays:
        for raw in np.asarray(array).reshape(-1):
            row = int(raw)
            if row not in seen:
                seen[row] = len(values)
                values.append(row)
    return np.asarray(values, dtype=np.int64), seen


def _to_union(array: np.ndarray, lookup: Mapping[int, int]) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    converted = np.fromiter(
        (lookup[int(value)] for value in flat),
        dtype=np.int64,
        count=len(flat),
    )
    return converted.reshape(array.shape)


def prepare_physics_raw_pool(
    *,
    prepared_subject_dir: Path,
    fem_labels_dir: Path,
    output_dir: Path,
    seed: int = 1042,
) -> Path:
    """Generate source-independent geometry and same-operator FEM residuals/scales without fitting thresholds."""

    prepared = load_prepared_subject_v1(
        prepared_subject_dir,
    )
    fem = load_fem_labels_v1(
        fem_labels_dir,
        prepared_subject=prepared,
    )
    tissue = np.asarray(prepared.tissue_labels_1mm, dtype=np.uint8)
    lookup = _lookup_rows(prepared.gmwm_ijk, prepared.full_shape)
    rng = np.random.default_rng(seed)
    circ_rows, circ_edges, circ_mask = _circulation_geometry(
        tissue,
        lookup,
        prepared.affine,
        count=PRIMITIVES_PER_OPERATOR,
        rng=rng,
    )
    wm_rows, gm_rows, normals = _interface_geometry(
        tissue,
        lookup,
        prepared.affine,
        count=PRIMITIVES_PER_OPERATOR,
        rng=rng,
    )
    flux_rows, flux_area, flux_sigma = _flux_geometry(
        tissue,
        lookup,
        prepared.affine,
        count=PRIMITIVES_PER_OPERATOR,
        rng=rng,
    )
    union_rows, union_lookup = _stable_union([circ_rows, wm_rows, gm_rows, flux_rows])
    circ_index = _to_union(circ_rows, union_lookup)
    wm_index = _to_union(wm_rows, union_lookup)
    gm_index = _to_union(gm_rows, union_lookup)
    flux_index = _to_union(flux_rows, union_lookup)
    fem_field = np.stack(
        [
            np.asarray(
                fem.load_source(name, mmap_mode="r")[union_rows], dtype=np.float32
            )
            for name in fem.source_names
        ],
        axis=0,
    )
    field = torch.from_numpy(fem_field).unsqueeze(0)
    circ_raw, circ_scale = circulation_raw_and_scale(
        field,
        torch.from_numpy(circ_index),
        torch.from_numpy(circ_edges),
        torch.from_numpy(circ_mask),
    )
    normal_raw, normal_scale, tangent_raw, tangent_scale = interface_raw_and_scale(
        field,
        torch.from_numpy(wm_index),
        torch.from_numpy(gm_index),
        torch.from_numpy(normals),
    )
    flux_raw, flux_scale = flux_raw_and_scale(
        field,
        torch.from_numpy(flux_index),
        torch.from_numpy(flux_area),
        torch.from_numpy(flux_sigma),
    )
    arrays: dict[str, np.ndarray] = {
        "union_query_rows": union_rows,
        "union_query_ijk": prepared.gmwm_ijk[union_rows],
        "affine_ijk_to_ras_mm": prepared.affine,
        "source_coords_mm": prepared.source_coords_mm,
        "reference_coord_mm": prepared.reference_coord_mm,
        "circ_point_index": circ_index,
        "circ_edge_vector_m": circ_edges,
        "circ_point_mask": circ_mask,
        "interface_wm_point_index": wm_index,
        "interface_gm_point_index": gm_index,
        "interface_normal_wm_to_gm": normals,
        "flux_face_point_index": flux_index,
        "flux_outward_area_vector_m2": flux_area,
        "flux_conductivity_s_per_m": flux_sigma,
        "circ_fem_raw_residual": circ_raw.squeeze(0).numpy(),
        "circ_fem_raw_scale": circ_scale.squeeze(0).numpy(),
        "interface_normal_fem_raw_residual": normal_raw.squeeze(0).numpy(),
        "interface_normal_fem_raw_scale": normal_scale.squeeze(0).numpy(),
        "interface_tangent_fem_raw_residual": tangent_raw.squeeze(0).numpy(),
        "interface_tangent_fem_raw_scale": tangent_scale.squeeze(0).numpy(),
        "flux_fem_raw_residual": flux_raw.squeeze(0).numpy(),
        "flux_fem_raw_scale": flux_scale.squeeze(0).numpy(),
    }
    destination = Path(output_dir).resolve(strict=False)
    staging = _staging_directory(destination)
    try:
        arrays_path = staging / "physics_raw_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        _fsync_file(arrays_path)
        payload = {
            "schema_id": PHYSICS_RAW_POOL_SCHEMA_ID,
            "operator_version": PHYSICS_OPERATOR_VERSION,
            "source_names": list(fem.source_names),
            "query_count": len(prepared.gmwm_ijk),
            "primitive_counts": {
                "circulation": PRIMITIVES_PER_OPERATOR,
                "interface": PRIMITIVES_PER_OPERATOR,
                "flux": PRIMITIVES_PER_OPERATOR,
            },
            "generation_seed": seed,
            "statistics_fitted": False,
            "arrays": {
                "relative_path": arrays_path.name,
                "size_bytes": arrays_path.stat().st_size,
            },
        }
        _write_json(staging / "training_physics_raw_pool_v1.json", payload)
        _fsync_directory(staging)
        _promote(staging, destination)
    finally:
        _cleanup_staging(staging)
    return destination / "training_physics_raw_pool_v1.json"


def _load_raw_pool(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = Path(path).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_id") != PHYSICS_RAW_POOL_SCHEMA_ID:
        raise ValueError("physics raw pool schema 不匹配")
    record = payload.get("arrays")
    if not isinstance(record, Mapping):
        raise ValueError("physics raw pool 缺少 arrays 记录")
    arrays_path = manifest_path.parent / str(record["relative_path"])
    if arrays_path.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError("physics raw arrays 必须与 manifest 同目录")
    if arrays_path.stat().st_size != int(record["size_bytes"]):
        raise ValueError("physics raw arrays 字节数不匹配")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return payload, arrays


def fit_physics_statistics(
    *,
    train_raw_pool_manifests: Sequence[Path],
    output_dir: Path,
) -> Path:
    """Fit frozen thresholds only from explicitly supplied training-partition raw pools."""

    if not train_raw_pool_manifests:
        raise ValueError("至少需要一个 train raw pool")
    loaded = [_load_raw_pool(path) for path in train_raw_pool_manifests]
    source_orders = {payload.get("source_order_sha256", "") for payload, _ in loaded}
    if len(source_orders) != 1:
        raise ValueError("train raw pools 的 source 顺序不一致")
    statistics: dict[str, dict[str, float]] = {}
    for prefix in ("circ", "interface_normal", "interface_tangent", "flux"):
        raw = np.concatenate(
            [arrays[f"{prefix}_fem_raw_residual"].reshape(-1) for _, arrays in loaded]
        ).astype(np.float64)
        scale = np.concatenate(
            [arrays[f"{prefix}_fem_raw_scale"].reshape(-1) for _, arrays in loaded]
        ).astype(np.float64)
        positive = scale[scale > 0.0]
        if not positive.size:
            raise ValueError(f"train raw pools 的 {prefix} scale 全为零")
        floor = max(float(np.quantile(positive, 0.01)), np.finfo(np.float32).tiny)
        denominator = np.maximum(scale, floor)
        normalized = raw / denominator
        median = float(np.median(normalized))
        mad = float(np.median(np.abs(normalized - median)))
        weak_cutoff = floor
        qmask = float(np.quantile(normalized, 0.999))
        statistics[prefix] = {
            "floor": floor,
            "mad": mad,
            "weak_cutoff": weak_cutoff,
            "qmask": qmask,
        }
    input_records = [
        {
            "manifest_path": str(Path(path).resolve()),
            "arrays_size_bytes": payload["arrays"]["size_bytes"],
            "source_names": payload["source_names"],
            "generation_seed": payload.get("generation_seed"),
        }
        for path, (payload, _) in zip(train_raw_pool_manifests, loaded, strict=True)
    ]
    payload = {
        "schema_id": PHYSICS_STATISTICS_SCHEMA_ID,
        "operator_version": PHYSICS_OPERATOR_VERSION,
        "denominator_rule_id": PHYSICS_DENOMINATOR_RULE_ID,
        "fit_partition": "train",
        "fit_subject_count": len(loaded),
        "quantiles": {"floor": 0.01, "qmask": 0.999},
        "inputs": input_records,
        "operators": statistics,
    }
    destination = Path(output_dir).resolve(strict=False)
    staging = _staging_directory(destination)
    try:
        _write_json(staging / "training_physics_statistics_v1.json", payload)
        _fsync_directory(staging)
        _promote(staging, destination)
    finally:
        _cleanup_staging(staging)
    return destination / "training_physics_statistics_v1.json"


def finalize_physics_pool(
    *,
    raw_pool_manifest: Path,
    statistics_manifest: Path,
    output_dir: Path,
) -> Path:
    """Apply training-fitted statistics to a training or validation raw pool."""

    raw_payload, raw_arrays = _load_raw_pool(raw_pool_manifest)
    statistics_path = Path(statistics_manifest).resolve(strict=True)
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if statistics.get("schema_id") != PHYSICS_STATISTICS_SCHEMA_ID:
        raise ValueError("physics statistics schema 不匹配")
    if statistics.get("fit_partition") != "train":
        raise ValueError("physics statistics 必须仅从训练划分拟合")
    arrays = dict(raw_arrays)
    for prefix in ("circ", "interface_normal", "interface_tangent", "flux"):
        values = statistics["operators"][prefix]
        raw = arrays[f"{prefix}_fem_raw_residual"].astype(np.float32)
        scale = arrays[f"{prefix}_fem_raw_scale"].astype(np.float32)
        floor = float(values["floor"])
        mad = float(values["mad"])
        denominator = np.maximum(scale, floor)
        normalized = raw / denominator
        valid = (scale >= float(values["weak_cutoff"])) & (
            normalized <= float(values["qmask"])
        )
        arrays[f"{prefix}_stratum_floor"] = np.full(
            PRIMITIVES_PER_OPERATOR, floor, dtype=np.float32
        )
        arrays[f"{prefix}_stratum_mad"] = np.full(
            PRIMITIVES_PER_OPERATOR, mad, dtype=np.float32
        )
        arrays[f"{prefix}_valid_mask"] = valid
    destination = Path(output_dir).resolve(strict=False)
    staging = _staging_directory(destination)
    try:
        arrays_path = staging / "physics_pool_arrays.npz"
        np.savez_compressed(arrays_path, **arrays)
        _fsync_file(arrays_path)
        payload = {
            "schema_id": PHYSICS_POOL_SCHEMA_ID,
            "operator_version": PHYSICS_OPERATOR_VERSION,
            "denominator_rule_id": PHYSICS_DENOMINATOR_RULE_ID,
            "source_names": raw_payload["source_names"],
            "query_count": raw_payload["query_count"],
            "primitive_counts": raw_payload["primitive_counts"],
            "statistics_fit_partition": "train",
            "arrays": {
                "relative_path": arrays_path.name,
                "size_bytes": arrays_path.stat().st_size,
            },
        }
        _write_json(staging / "training_physics_pool_v1.json", payload)
        _fsync_directory(staging)
        _promote(staging, destination)
    finally:
        _cleanup_staging(staging)
    return destination / "training_physics_pool_v1.json"


__all__ = [
    "PHYSICS_RAW_POOL_SCHEMA_ID",
    "PHYSICS_STATISTICS_SCHEMA_ID",
    "PRIMITIVES_PER_OPERATOR",
    "finalize_physics_pool",
    "fit_physics_statistics",
    "prepare_physics_raw_pool",
]
