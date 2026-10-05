"""Public model types and synthetic forward contracts."""

from __future__ import annotations

import torch
from conftest import make_small_model

from head2field.geometry import (
    ElectrodeCoordinateBundle,
    QueryCoordinateBundle,
    SubjectGeometry,
)


def test_small_model_forward_has_stable_axis_order() -> None:
    model = make_small_model()
    shape = (8, 8, 8)
    affine = torch.eye(4).unsqueeze(0)
    geometry = SubjectGeometry(
        full_shape=shape,
        encoder_input_shape=shape,
        affine_ijk_to_ras_mm=affine,
        inverse_affine_ras_mm_to_ijk=affine.clone(),
        anatomy_origin_ijk=torch.tensor([[3.5, 3.5, 3.5]]),
        anatomy_origin_normalized=torch.zeros(1, 3),
        anatomy_origin_world_mm=torch.tensor([[3.5, 3.5, 3.5]]),
        orientation="RAS",
    )
    query_norm = torch.tensor([[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]])
    query_ijk = geometry.normalized_to_ijk(query_norm)
    queries = QueryCoordinateBundle(
        gmwm_row_index=torch.tensor([[0, 1]]),
        ijk=query_ijk,
        normalized_coord=query_norm,
        world_ras_mm=query_ijk.clone(),
        features_1mm=torch.zeros(1, 2, 12),
    )
    electrode_norm = torch.tensor(
        [[[-0.6, 0.4, -0.2], [0.3, -0.5, 0.4], [0.6, 0.6, 0.6]]]
    )
    electrode_ijk = geometry.normalized_to_ijk(electrode_norm)
    electrodes = ElectrodeCoordinateBundle(
        electrode_id=torch.tensor([[1, 2, 3]]),
        normalized_coord=electrode_norm,
        world_ras_mm=electrode_ijk,
    )
    pairs = torch.tensor([[[0, 2], [1, 2]]], dtype=torch.long)

    with torch.inference_mode():
        output = model.forward_bundled(
            torch.zeros(1, 12, *shape),
            geometry,
            queries,
            electrodes,
            pairs,
            deterministic=True,
        )

    assert output.response_bank.shape == (1, 3, 2, 3)
    assert output.pair_field.shape == (1, 2, 2, 3)
    assert torch.isfinite(output.pair_field).all()
