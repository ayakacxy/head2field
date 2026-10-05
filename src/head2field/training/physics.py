"""Differentiable world-RAS discrete physics operators and excess losses for N128 GP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

SIGMA_WM_S_PER_M = 0.126
SIGMA_GM_S_PER_M = 0.275


@dataclass(frozen=True)
class PhysicsLossResult:
    """Loss and auditable diagnostics for one physical term."""

    loss: torch.Tensor
    raw_prediction_residual: torch.Tensor
    normalized_prediction_residual: torch.Tensor
    normalized_fem_residual: torch.Tensor
    tolerance: torch.Tensor
    excess: torch.Tensor
    valid_mask: torch.Tensor

    def metrics(self, prefix: str) -> dict[str, torch.Tensor]:
        valid = self.valid_mask
        count = valid.sum()
        safe_count = count.clamp_min(1)

        def mean(value: torch.Tensor) -> torch.Tensor:
            return (value * valid.to(value.dtype)).sum() / safe_count

        return {
            f"{prefix}_loss": self.loss.detach(),
            f"{prefix}_raw_pred_residual": mean(self.raw_prediction_residual.detach()),
            f"{prefix}_normalized_pred_residual": mean(
                self.normalized_prediction_residual.detach()
            ),
            f"{prefix}_normalized_fem_residual": mean(
                self.normalized_fem_residual.detach()
            ),
            f"{prefix}_excess_residual": mean(self.excess.detach()),
            f"{prefix}_above_tolerance_rate": mean(
                (self.normalized_prediction_residual > self.tolerance).to(
                    self.normalized_prediction_residual.dtype
                )
            ),
            f"{prefix}_valid_count": count.detach(),
            f"{prefix}_masked_count": (
                torch.as_tensor(valid.numel(), device=valid.device) - count
            ).detach(),
        }


def _require_field(name: str, field: torch.Tensor) -> torch.Tensor:
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"{name} 必须为 [B,P,Q,3]，实际={tuple(field.shape)}")
    value = field.float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} 含 NaN/Inf，禁止用 physics mask 隐藏")
    return value


def _gather_queries(field: torch.Tensor, row_index: torch.Tensor) -> torch.Tensor:
    """Gather physical sample points from ``[B,P,Q,3]`` using union indices."""

    if row_index.dtype != torch.long:
        raise TypeError("physics union row index 必须使用 torch.long")
    if bool((row_index < 0).any()) or bool((row_index >= field.shape[2]).any()):
        raise IndexError("physics union row index 越界")
    return field[:, :, row_index, :]


def circulation_raw_and_scale(
    field: torch.Tensor,
    point_index: torch.Tensor,
    edge_vector_m: torch.Tensor,
    point_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute trapezoidal loop integrals and the ``L_loop * RMS(|E|)`` scale."""

    values = _require_field("circulation field", field)
    if point_index.ndim != 2 or edge_vector_m.shape != (*point_index.shape, 3):
        raise ValueError("circulation point/edge 必须为 [K,M]/[K,M,3]")
    mask = (
        torch.ones_like(point_index, dtype=torch.bool)
        if point_mask is None
        else point_mask.to(dtype=torch.bool, device=point_index.device)
    )
    if mask.shape != point_index.shape or not bool(mask.any(dim=-1).all()):
        raise ValueError("每个 circulation primitive 至少需要一条有效边")
    sampled = _gather_queries(values, point_index)
    lengths = mask.sum(dim=-1)
    next_position = (
        torch.arange(point_index.shape[1], device=point_index.device)[None].expand_as(
            point_index
        )
        + 1
    )
    next_position = torch.where(
        next_position < lengths[:, None],
        next_position,
        torch.zeros_like(next_position),
    )
    gather_index = next_position[None, None, :, :, None].expand(
        sampled.shape[0], sampled.shape[1], -1, -1, sampled.shape[-1]
    )
    sampled_next = sampled.gather(dim=3, index=gather_index)
    edges = edge_vector_m.float()[None, None]
    weights = mask.float()[None, None]
    raw = (
        ((0.5 * (sampled + sampled_next) * edges).sum(dim=-1) * weights)
        .sum(dim=-1)
        .abs()
    )
    loop_length = (
        torch.linalg.vector_norm(edge_vector_m.float(), dim=-1) * mask.float()
    ).sum(dim=-1)
    mean_energy = (sampled.square().sum(dim=-1) * weights).sum(
        dim=-1
    ) / lengths.float()[None, None].clamp_min(1.0)
    scale = loop_length[None, None] * torch.sqrt(mean_energy.clamp_min(0.0))
    return raw, scale


def flux_raw_and_scale(
    field: torch.Tensor,
    face_point_index: torch.Tensor,
    outward_area_vector_m2: torch.Tensor,
    conductivity_s_per_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute six-face control-volume current flux and the ``A_total * RMS(|J|)`` scale."""

    values = _require_field("flux field", field)
    if face_point_index.ndim != 2 or face_point_index.shape[1] < 6:
        raise ValueError("flux face_point_index 必须为 [K,F] 且 F>=6")
    if outward_area_vector_m2.shape != (*face_point_index.shape, 3):
        raise ValueError("flux outward_area_vector_m2 必须为 [K,F,3]")
    if conductivity_s_per_m.shape != (face_point_index.shape[0],):
        raise ValueError("flux conductivity 必须为 [K]")
    sampled = _gather_queries(values, face_point_index)
    sigma = conductivity_s_per_m.float()[None, None, :, None, None]
    area_vector = outward_area_vector_m2.float()[None, None]
    raw = (sigma * sampled * area_vector).sum(dim=(-1, -2)).abs()
    face_area = torch.linalg.vector_norm(outward_area_vector_m2.float(), dim=-1)
    total_area = face_area.sum(dim=-1)
    current_energy = (sigma * sampled).square().sum(dim=-1).mean(dim=-1)
    scale = total_area[None, None] * torch.sqrt(current_energy.clamp_min(0.0))
    return raw, scale


def interface_raw_and_scale(
    field: torch.Tensor,
    wm_point_index: torch.Tensor,
    gm_point_index: torch.Tensor,
    normal_wm_to_gm: torch.Tensor,
    *,
    sigma_wm: float = SIGMA_WM_S_PER_M,
    sigma_gm: float = SIGMA_GM_S_PER_M,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute WM/GM normal-current and tangential-field continuity residuals."""

    values = _require_field("interface field", field)
    if wm_point_index.ndim != 1 or gm_point_index.shape != wm_point_index.shape:
        raise ValueError("interface WM/GM index 必须为同形 [K]")
    normals = normal_wm_to_gm.float()
    if normals.shape != (wm_point_index.shape[0], 3):
        raise ValueError("interface normal 必须为 [K,3]")
    norm = torch.linalg.vector_norm(normals, dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=2e-4, rtol=2e-4):
        raise ValueError("interface normal 必须是 world-RAS 单位向量")
    wm = _gather_queries(values, wm_point_index[:, None]).squeeze(3)
    gm = _gather_queries(values, gm_point_index[:, None]).squeeze(3)
    normal = normals[None, None]
    current_jump = float(sigma_wm) * wm - float(sigma_gm) * gm
    normal_raw = (current_jump * normal).sum(dim=-1).abs()
    current_scale = torch.sqrt(
        0.5
        * (
            (float(sigma_wm) * wm).square().sum(dim=-1)
            + (float(sigma_gm) * gm).square().sum(dim=-1)
        )
    )
    field_jump = wm - gm
    normal_component = (field_jump * normal).sum(dim=-1, keepdim=True) * normal
    tangent_raw = torch.linalg.vector_norm(field_jump - normal_component, dim=-1)
    tangent_scale = torch.sqrt(
        0.5 * (wm.square().sum(dim=-1) + gm.square().sum(dim=-1))
    )
    return normal_raw, current_scale, tangent_raw, tangent_scale


def frozen_denominator(
    fem_raw_scale: torch.Tensor,
    stratum_floor: torch.Tensor,
) -> torch.Tensor:
    """Apply the sole allowed frozen FEM denominator rule."""

    scale = fem_raw_scale.detach().float()
    floor = stratum_floor.detach().float()
    if bool((scale < 0).any()) or bool((floor < 0).any()):
        raise ValueError("FEM raw scale 和 stratum floor 不能为负数")
    return torch.maximum(scale, floor)


def one_sided_excess_loss(
    prediction_raw_residual: torch.Tensor,
    fem_raw_residual: torch.Tensor,
    fem_raw_scale: torch.Tensor,
    stratum_floor: torch.Tensor,
    stratum_mad: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    kappa: float,
    huber_delta: float,
) -> PhysicsLossResult:
    """Compute one-sided Huber excess loss above FEM residual + kappa*MAD tolerance."""

    if kappa < 0.0 or huber_delta <= 0.0:
        raise ValueError("kappa 必须非负且 physics Huber delta 必须为正")
    prediction = prediction_raw_residual.float()
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("prediction physics residual 含 NaN/Inf")
    denominator = frozen_denominator(fem_raw_scale, stratum_floor).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized_prediction = prediction / denominator
    normalized_fem = fem_raw_residual.detach().float() / denominator
    tolerance = normalized_fem + float(kappa) * stratum_mad.detach().float()
    mask = valid_mask.detach().to(dtype=torch.bool, device=prediction.device)
    if mask.shape != prediction.shape:
        mask = torch.broadcast_to(mask, prediction.shape)
    excess = torch.relu(normalized_prediction - tolerance)
    huber = torch.where(
        excess <= huber_delta,
        0.5 * excess.square(),
        huber_delta * (excess - 0.5 * huber_delta),
    )
    valid_float = mask.to(huber.dtype)
    loss = (huber * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    return PhysicsLossResult(
        loss=loss,
        raw_prediction_residual=prediction,
        normalized_prediction_residual=normalized_prediction,
        normalized_fem_residual=normalized_fem,
        tolerance=tolerance,
        excess=excess,
        valid_mask=mask,
    )


def weighted_physics_sum(
    results: Mapping[str, PhysicsLossResult],
    weights: Mapping[str, float],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine physical terms using explicit nonnegative weights."""

    total = reference.float().sum() * 0.0
    metrics: dict[str, torch.Tensor] = {}
    for name, result in results.items():
        weight = float(weights.get(name, 0.0))
        if weight < 0.0:
            raise ValueError(f"physics weight {name} 不能为负数")
        total = total + weight * result.loss
        metrics.update(result.metrics(name))
        metrics[f"{name}_weight"] = torch.tensor(
            weight, device=reference.device, dtype=torch.float32
        )
    metrics["physics_loss"] = total.detach()
    return total, metrics


def curriculum_scale(global_step: int, *, warmup_steps: int, ramp_steps: int) -> float:
    """Compute the fixed 0-to-1 physics curriculum scale from the global step."""

    if global_step < 0 or warmup_steps < 0 or ramp_steps <= 0:
        raise ValueError("global_step/warmup/ramp 合同无效")
    if global_step <= warmup_steps:
        return 0.0
    return min((global_step - warmup_steps) / ramp_steps, 1.0)


__all__ = [
    "SIGMA_GM_S_PER_M",
    "SIGMA_WM_S_PER_M",
    "PhysicsLossResult",
    "circulation_raw_and_scale",
    "curriculum_scale",
    "flux_raw_and_scale",
    "frozen_denominator",
    "interface_raw_and_scale",
    "one_sided_excess_loss",
    "weighted_physics_sum",
]
