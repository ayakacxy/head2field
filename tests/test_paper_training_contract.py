"""Check manuscript radial boundaries, cache migration, and epoch normalization."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from head2field.training import cache
from head2field.training.contracts import TrainingContractError, sha256_file
from head2field.training.dataset import (
    DeterministicHead2FieldSampler,
    load_loss_statistics,
)


def _statistics(tmp_path, monkeypatch):
    """Generate production statistics from analytic fields straddling both edges."""

    names = tuple(f"E{index:02d}" for index in range(75))
    prepared = SimpleNamespace(
        manifest_sha256="1" * 64,
        manifest={"files": {"voxels/gmwm_ijk.npy": {"sha256": "2" * 64}}},
        gmwm_ijk=np.asarray([[49, 0, 0], [50, 0, 0], [99, 0, 0], [100, 0, 0]]),
        affine=np.eye(4),
        source_coords_mm=np.zeros((75, 3), dtype=np.float32),
        reference_coord_mm=np.asarray([1000, 0, 0], dtype=np.float32),
    )
    fem = SimpleNamespace(
        manifest_sha256="3" * 64,
        source_names=names,
        load_source=lambda *args, **kwargs: np.tile([1.0, 0.0, 0.0], (4, 1)),
    )
    monkeypatch.setattr(cache, "load_prepared_subject_v1", lambda *a, **k: prepared)
    monkeypatch.setattr(cache, "load_fem_labels_v1", lambda *a, **k: fem)
    manifest = cache.prepare_loss_statistics(
        prepared_subject_dir=tmp_path / "prepared",
        fem_labels_dir=tmp_path / "fem",
        output_dir=tmp_path / "statistics",
        chunk_size=2,
    )
    return manifest, dict(prepared=prepared, fem=fem, expected_sources=names)


def test_paper_statistics_put_50_and_100_mm_in_the_correct_regions(
    tmp_path, monkeypatch
) -> None:
    manifest, bindings = _statistics(tmp_path, monkeypatch)
    statistics = load_loss_statistics(manifest, **bindings)
    assert manifest.name == "training_loss_statistics_v2.json"
    np.testing.assert_array_equal(statistics.radial_edges_mm, [50.0, 100.0])
    np.testing.assert_array_equal(statistics.radial_energy, np.tile([1, 2, 1], (75, 1)))
    assert statistics.global_median_energy == 4.0
    np.testing.assert_array_equal(statistics.radial_median_energy, [1, 2, 1])


@pytest.mark.parametrize("legacy_field", ["schema_id", "configuration"])
def test_paper_loader_rejects_legacy_statistics(tmp_path, monkeypatch, legacy_field):
    manifest, bindings = _statistics(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    if legacy_field == "schema_id":
        payload["schema_id"] = "training_loss_statistics_v1"
    else:
        payload["configuration"]["radial_edges_mm"] = [20.0, 50.0]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingContractError, match="regenerate"):
        load_loss_statistics(manifest, **bindings)


def test_cache_cannot_relabel_legacy_array_edges_as_paper_statistics(
    tmp_path, monkeypatch
) -> None:
    manifest, bindings = _statistics(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text())
    arrays_path = manifest.parent / payload["arrays"]["relative_path"]
    with np.load(arrays_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["radial_edges_mm"] = np.asarray([20.0, 50.0], dtype=np.float32)
    np.savez_compressed(arrays_path, **arrays)
    payload["arrays"].update(
        size_bytes=arrays_path.stat().st_size, sha256=sha256_file(arrays_path)
    )
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingContractError, match="radial edges"):
        load_loss_statistics(manifest, **bindings)


def test_epoch_loss_scale_stays_fixed_for_partial_source_batches() -> None:
    sampler = DeterministicHead2FieldSampler(
        ((5, 75), (9, 75)), query_count=4, sources_per_step=8, seed=42
    )
    assert sampler.steps_per_epoch == 50
    assert sampler.data_loss_scale == pytest.approx(50 / (75 * 2))
    assert any(
        len(sampler.next_selection().source_indices) == 3
        for _ in range(sampler.steps_per_epoch)
    )


@pytest.mark.parametrize("field", ["gmwm_ijk", "affine", "source_coords_mm", "reference_coord_mm"])
def test_statistics_reject_changed_query_or_electrode_geometry(tmp_path, monkeypatch, field):
    """能量统计直接绑定实际空间和电极坐标。"""
    manifest, bindings = _statistics(tmp_path, monkeypatch)
    prepared = bindings["prepared"]
    setattr(prepared, field, getattr(prepared, field) + 1)
    with pytest.raises(TrainingContractError, match="不匹配"):
        load_loss_statistics(manifest, **bindings)
