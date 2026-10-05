"""Synthetic fixtures for the release-candidate inference vertical slice."""

from __future__ import annotations

import numpy as np
import torch

from head2field.inputs import PreparedModelInputSubject
from head2field.model import Head2FieldV1

SMALL_MODEL_CONFIG: dict[str, object] = {
    "base_channels": 4,
    "encoder_levels": 3,
    "pyramid_dim": 8,
    "query_dim": 32,
    "query_blocks": 1,
    "query_expansion": 2,
    "electrode_dim": 24,
    "decoder_blocks": 1,
    "decoder_bottleneck_dim": 16,
    "global_dim": 16,
    "coordinate_fourier_bands": 2,
    "context_dropout": 0.0,
    "absolute_fourier_bands": 2,
    "absolute_aux_dim": 8,
    "absolute_coordinate_dropout": 0.0,
    "absolute_branch_dropout": 0.0,
    "absolute_coordinate_jitter_probability": 0.0,
    "absolute_coordinate_jitter_std_mm": 1.0,
    "global_mixer_dim": 16,
    "global_mixer_blocks": 1,
    "global_mixer_heads": 4,
    "global_mixer_local_window": 3,
    "global_mixer_max_relative_distance": 8,
    "global_mixer_swiglu_expansion": 2,
    "variant_id": "GP",
}


def make_small_model() -> Head2FieldV1:
    """Create a small same-architecture model suitable for CPU regression tests."""

    torch.manual_seed(20260901)
    return Head2FieldV1(**SMALL_MODEL_CONFIG).eval()


def make_synthetic_subject() -> PreparedModelInputSubject:
    """Create an eight-cubed input without real subject information."""

    shape = (8, 8, 8)
    tissues = np.ones(shape, dtype=np.uint8)
    tissues[4:, :, :] = 2
    head = torch.zeros(12, *shape, dtype=torch.float32)
    head[0, :4] = 1.0
    head[1, 4:] = 1.0
    head[10] = 0.25
    affine = np.eye(4, dtype=np.float32)
    return PreparedModelInputSubject(
        subject_id="synthetic-subject-a",
        head_volume=head,
        tissue_labels_1mm=tissues,
        t1_normalized_1mm=np.zeros(shape, dtype=np.float16),
        affine=affine,
        inverse_affine=affine.copy(),
        full_shape=shape,
        gmwm_ijk=np.asarray(
            [[1, 1, 1], [3, 4, 2], [6, 5, 6]],
            dtype=np.int64,
        ),
        source_names=("E01", "E02", "E03"),
        source_coords_mm=np.asarray(
            [[1.0, 6.0, 1.0], [3.0, 1.0, 6.0], [6.0, 4.0, 2.0]],
            dtype=np.float32,
        ),
        reference_coord_mm=np.asarray([6.0, 6.0, 6.0], dtype=np.float32),
    )
