"""Cached inference vertical slice on synthetic inputs."""

from __future__ import annotations

import numpy as np
import torch
from conftest import make_small_model, make_synthetic_subject

from head2field.inference import Head2FieldInferenceSession


def test_inference_session_is_finite_and_reuses_caches() -> None:
    subject = make_synthetic_subject()
    session = Head2FieldInferenceSession(
        make_small_model(),
        subject,
        device="cpu",
        query_cache_size=1,
    )
    prepared = session.prepare_electrodes(
        subject.source_coords_mm,
        source_electrode_ids=np.asarray([1, 2, 3], dtype=np.int64),
        reference_world_ras_mm=subject.reference_coord_mm,
        reference_electrode_id=4,
        electrode_batch_size=2,
    )
    rows = np.arange(len(subject.gmwm_ijk), dtype=np.int64)

    first = session.predict_rows(rows, prepared)
    second = session.predict_rows(rows, prepared)

    assert first.shape == (1, 3, 3, 3)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert session.cached_electrode_count == 1
    assert session.cached_query_count == 1


def test_chunking_preserves_selected_pair_and_query_order() -> None:
    """Repeated selections and shared references retain their identities across tails."""

    subject = make_synthetic_subject()
    session = Head2FieldInferenceSession(make_small_model(), subject, device="cpu")
    full_pairs = session.prepare_electrodes(
        subject.source_coords_mm, electrode_batch_size=3
    )
    full_field = session.predict_rows(np.arange(3, dtype=np.int64), full_pairs)
    rows = np.asarray([2, 0, 2, 1, 0], dtype=np.int64)
    pairs = np.asarray([2, 0, 2], dtype=np.int64)
    batched_pairs = session.prepare_electrodes(
        subject.source_coords_mm, electrode_batch_size=2
    )

    chunks = list(
        session.iter_predictions(
            rows, batched_pairs, query_chunk_size=2, pair_indices=pairs
        )
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk.row_indices for chunk in chunks]), rows
    )
    actual = torch.cat([chunk.prediction for chunk in chunks], dim=2)
    expected = full_field[:, pairs][:, :, rows]
    assert actual.shape == (1, 3, 5, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
