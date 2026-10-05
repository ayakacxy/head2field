"""Frozen stable-energy data loss for lead-field prediction v1."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _safe_valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_float = valid.to(values.dtype)
    return (values * valid_float).sum() / valid_float.sum().clamp_min(1.0)


def lead_field_data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, object],
    *,
    denominator_floor_ratio: float = 0.01,
    pair_weight: float = 0.60,
    radial_weight: float = 0.30,
    peak_weight: float = 0.10,
    huber_delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return stable-energy supervision and diagnostics for ``[B,P,Q,3]`` fields.

    ``batch`` supplies full-subject energy statistics, world-RAS query/electrode
    coordinates, radial boundaries, high-field thresholds, and schedule weights.
    Numerators use the current query chunk; denominators and peak masks come
    from FEM targets. The fixed schedule scale includes partial source/query
    tails. FP32 arithmetic keeps the objective stable under CUDA autocast.

    The optimized loss combines global, radial, and high-field energy terms.
    Huber, MAE, and relative L2 are diagnostics only; they do not alter the loss.
    """

    if (
        prediction.shape != target.shape
        or prediction.ndim != 4
        or prediction.shape[-1] != 3
    ):
        raise ValueError("prediction and target must have matching shape [B,P,Q,3]")
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise FloatingPointError("prediction or target contains NaN or Inf")
    if denominator_floor_ratio <= 0.0 or huber_delta <= 0.0:
        raise ValueError("denominator_floor_ratio and huber_delta must be positive")
    if any(weight < 0.0 for weight in (pair_weight, radial_weight, peak_weight)):
        raise ValueError("data loss weights must be non-negative")

    prediction_float = prediction.float()
    target_float = target.float()
    error_components = prediction_float - target_float
    error_components_squared = error_components.square()
    error_squared = error_components_squared.sum(dim=-1)
    target_squared = target_float.square().sum(dim=-1)

    def tensor(name: str) -> torch.Tensor:
        if name not in batch:
            raise KeyError(f"data loss batch is missing {name!r}")
        return torch.as_tensor(batch[name], device=prediction.device).float()

    global_energy = tensor("global_energy")
    radial_energy = tensor("radial_energy")
    peak_energy = tensor("peak_energy")
    global_median = tensor("global_median_energy")
    radial_median = tensor("radial_median_energy")
    peak_median = tensor("peak_median_energy")

    # Denominators use complete FEM fields, so chunking cannot refit the floor.
    global_denominator = global_energy + denominator_floor_ratio * global_median
    radial_denominator = (
        radial_energy + denominator_floor_ratio * radial_median[None, None]
    )
    peak_denominator = peak_energy + denominator_floor_ratio * peak_median

    global_numerator = error_squared.sum(dim=-1)
    global_valid = global_energy > 0.0
    global_ratio = torch.where(
        global_valid,
        global_numerator / global_denominator.clamp_min(1e-20),
        torch.zeros_like(global_numerator),
    )

    # Assign each query to a radial region by its nearest pair endpoint.
    radial_edges = tensor("radial_edges_mm")
    if radial_edges.shape != (2,) or not bool(radial_edges[0] < radial_edges[1]):
        raise ValueError("radial_edges_mm must contain two strictly increasing values")
    query_mm = tensor("query_mm")
    source_mm = tensor("source_mm")
    reference_mm = tensor("reference_mm")
    query_expanded = query_mm[:, None]
    source_distance = torch.linalg.vector_norm(
        query_expanded - source_mm[:, :, None], dim=-1
    )
    reference_distance = torch.linalg.vector_norm(
        query_expanded - reference_mm[:, None, None], dim=-1
    )
    nearest_electrode_distance = torch.minimum(source_distance, reference_distance)
    radial_masks = torch.stack(
        [
            nearest_electrode_distance < radial_edges[0],
            (nearest_electrode_distance >= radial_edges[0])
            & (nearest_electrode_distance < radial_edges[1]),
            nearest_electrode_distance >= radial_edges[1],
        ],
        dim=-1,
    )
    radial_numerator = torch.stack(
        [(error_squared * radial_masks[..., index]).sum(dim=-1) for index in range(3)],
        dim=-1,
    )
    radial_valid = radial_energy > 0.0
    radial_ratio = torch.where(
        radial_valid,
        radial_numerator / radial_denominator.clamp_min(1e-20),
        torch.zeros_like(radial_numerator),
    )
    radial_loss_per_pair = (radial_ratio * radial_valid.to(radial_ratio.dtype)).sum(
        dim=-1
    ) / radial_valid.sum(dim=-1).clamp_min(1)

    # Inclusive thresholds and validity masks depend only on FEM supervision.
    peak_threshold_squared = tensor("peak_threshold_squared")
    peak_mask = target_squared >= peak_threshold_squared[:, :, None]
    peak_numerator = (error_squared * peak_mask).sum(dim=-1)
    peak_valid = peak_energy > 0.0
    peak_ratio = torch.where(
        peak_valid,
        peak_numerator / peak_denominator.clamp_min(1e-20),
        torch.zeros_like(peak_numerator),
    )

    # Use the training-derived epoch scale for every chunk, including tails.
    schedule_scale = tensor("schedule_loss_scale")
    source_pair_weights = tensor("pair_weights")
    pair_energy_loss = (
        schedule_scale
        * (
            global_ratio * source_pair_weights * global_valid.to(global_ratio.dtype)
        ).sum()
    )
    radial_energy_loss = schedule_scale * radial_loss_per_pair.sum()
    peak_energy_loss = (
        schedule_scale * (peak_ratio * peak_valid.to(peak_ratio.dtype)).sum()
    )
    loss = (
        pair_weight * pair_energy_loss
        + radial_weight * radial_energy_loss
        + peak_weight * peak_energy_loss
    )

    # Metrics below are diagnostic and are not added to the optimized objective.
    chunk_target_energy = target_squared.sum(dim=-1)
    relative_l2_per_pair = torch.sqrt(global_numerator.clamp_min(0.0)) / torch.sqrt(
        chunk_target_energy.clamp_min(0.0)
    ).clamp_min(1e-8)
    voxel_batch_count = float(batch["voxel_batch_count"])
    estimated_global_relative = torch.sqrt(
        (
            voxel_batch_count * global_numerator / global_energy.clamp_min(1e-20)
        ).clamp_min(0.0)
    )
    estimated_radial_relative = torch.sqrt(
        (
            voxel_batch_count * radial_numerator / radial_energy.clamp_min(1e-20)
        ).clamp_min(0.0)
    )
    absolute_error = error_components.abs()
    huber = torch.where(
        absolute_error <= huber_delta,
        0.5 * error_components_squared,
        huber_delta * (absolute_error - 0.5 * huber_delta),
    ).mean()
    metrics = {
        "pair_energy_loss": pair_energy_loss,
        "radial_energy_loss": radial_energy_loss,
        "peak_energy_loss": peak_energy_loss,
        "relative_l2": relative_l2_per_pair.mean(),
        "estimated_full_relative_l2": _safe_valid_mean(
            estimated_global_relative, global_valid
        ),
        "near_relative_l2": _safe_valid_mean(
            estimated_radial_relative[..., 0], radial_valid[..., 0]
        ),
        "middle_relative_l2": _safe_valid_mean(
            estimated_radial_relative[..., 1], radial_valid[..., 1]
        ),
        "far_relative_l2": _safe_valid_mean(
            estimated_radial_relative[..., 2], radial_valid[..., 2]
        ),
        "huber": huber,
        "mae": absolute_error.mean(),
        "rmse": torch.sqrt(error_components_squared.mean()),
    }
    return loss, metrics


__all__ = ["lead_field_data_loss"]
