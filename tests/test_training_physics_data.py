"""Tests separating raw physics, training-only fitting, and final-pool stages."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from head2field.training.contracts import (
    PHYSICS_OPERATOR_VERSION,
    canonical_json_sha256,
    sha256_file,
)
from head2field.training.physics_data import (
    PHYSICS_RAW_POOL_SCHEMA_ID,
    finalize_physics_pool,
    fit_physics_statistics,
)
from head2field.training.dataset import load_physics_pool
from head2field.training.contracts import TrainingContractError


def _raw_pool(root, offset: float):
    root.mkdir()
    count = 512
    source_count = 75
    arrays = {
        "union_query_rows": np.arange(6, dtype=np.int64),
        "union_query_ijk": np.arange(18, dtype=np.int64).reshape(6, 3),
        "affine_ijk_to_ras_mm": np.eye(4),
        "source_coords_mm": np.zeros((75, 3), dtype=np.float32),
        "reference_coord_mm": np.zeros(3, dtype=np.float32),
        "circ_point_index": np.zeros((count, 4), dtype=np.int64),
        "circ_edge_vector_m": np.zeros((count, 4, 3), dtype=np.float32),
        "circ_point_mask": np.ones((count, 4), dtype=bool),
        "interface_wm_point_index": np.zeros(count, dtype=np.int64),
        "interface_gm_point_index": np.ones(count, dtype=np.int64),
        "interface_normal_wm_to_gm": np.tile(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1)
        ),
        "flux_face_point_index": np.zeros((count, 6), dtype=np.int64),
        "flux_outward_area_vector_m2": np.zeros((count, 6, 3), dtype=np.float32),
        "flux_conductivity_s_per_m": np.full(count, 0.126, dtype=np.float32),
    }
    for prefix_index, prefix in enumerate(
        ("circ", "interface_normal", "interface_tangent", "flux")
    ):
        base = 0.01 * (prefix_index + 1) + offset
        arrays[f"{prefix}_fem_raw_residual"] = np.full(
            (source_count, count), base, dtype=np.float32
        )
        arrays[f"{prefix}_fem_raw_scale"] = np.full(
            (source_count, count), 1.0 + base, dtype=np.float32
        )
    arrays_path = root / "physics_raw_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "schema_id": PHYSICS_RAW_POOL_SCHEMA_ID,
        "operator_version": PHYSICS_OPERATOR_VERSION,
        "prepared_subject_manifest_sha256": "1" * 64,
        "fem_labels_manifest_sha256": "2" * 64,
        "source_names": [f"E{index:02d}" for index in range(source_count)],
        "source_order_sha256": "3" * 64,
        "query_order_sha256": "4" * 64,
        "query_count": 6,
        "primitive_counts": {
            "circulation": count,
            "interface": count,
            "flux": count,
        },
        "generation_seed": 1042,
        "statistics_fitted": False,
        "arrays": {
            "relative_path": arrays_path.name,
            "size_bytes": arrays_path.stat().st_size,
            "sha256": sha256_file(arrays_path),
        },
    }
    manifest = root / "training_physics_raw_pool_v1.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_physics_pool_rejects_changed_query_coordinates(tmp_path):
    """primitive 与当前查询的直接坐标比较能检测错位。"""
    raw = _raw_pool(tmp_path / "raw", 0.0)
    statistics = fit_physics_statistics(train_raw_pool_manifests=[raw], output_dir=tmp_path / "statistics")
    manifest = finalize_physics_pool(raw_pool_manifest=raw, statistics_manifest=statistics, output_dir=tmp_path / "final")
    prepared = SimpleNamespace(
        gmwm_ijk=np.arange(18, dtype=np.int64).reshape(6, 3),
        affine=np.eye(4), source_coords_mm=np.zeros((75, 3), dtype=np.float32),
        reference_coord_mm=np.zeros(3, dtype=np.float32),
    )
    sources = [f"E{index:02d}" for index in range(75)]
    load_physics_pool(manifest, prepared=prepared, fem=None, expected_sources=sources)
    prepared.gmwm_ijk = prepared.gmwm_ijk[::-1]
    with pytest.raises(TrainingContractError, match="union_query_ijk"):
        load_physics_pool(manifest, prepared=prepared, fem=None, expected_sources=sources)


def test_fit_uses_explicit_train_raw_pools_then_finalizes(tmp_path) -> None:
    first = _raw_pool(tmp_path / "raw-a", 0.0)
    second = _raw_pool(tmp_path / "raw-b", 0.002)
    statistics = fit_physics_statistics(
        train_raw_pool_manifests=[first, second],
        output_dir=tmp_path / "statistics",
    )
    statistics_payload = json.loads(statistics.read_text(encoding="utf-8"))

    assert statistics_payload["fit_partition"] == "train"
    assert statistics_payload["fit_subject_count"] == 2
    assert [item["manifest_path"] for item in statistics_payload["inputs"]] == [str(first.resolve()), str(second.resolve())]
    assert all(item["arrays_size_bytes"] > 0 for item in statistics_payload["inputs"])

    final = finalize_physics_pool(
        raw_pool_manifest=first,
        statistics_manifest=statistics,
        output_dir=tmp_path / "final",
    )
    final_payload = json.loads(final.read_text(encoding="utf-8"))
    assert final_payload["statistics_fit_partition"] == "train"
    with np.load(
        tmp_path / "final/physics_pool_arrays.npz", allow_pickle=False
    ) as arrays:
        assert arrays["circ_valid_mask"].shape == (75, 512)
        assert arrays["circ_stratum_floor"].shape == (512,)
        assert arrays["flux_stratum_mad"].shape == (512,)
