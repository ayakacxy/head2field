"""Combine frozen N128 GP forwards, data loss, and four physics terms into one step."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from head2field.model import Head2FieldV1

from .contracts import DataLossConfig, PhysicsLossConfig
from .dataset import Head2FieldTrainingBatch
from .losses import lead_field_data_loss
from .physics import (
    PhysicsLossResult,
    circulation_raw_and_scale,
    curriculum_scale,
    flux_raw_and_scale,
    interface_raw_and_scale,
    one_sided_excess_loss,
    weighted_physics_sum,
)


def _physics_result(
    prediction_raw: torch.Tensor,
    arrays: Mapping[str, torch.Tensor],
    prefix: str,
    *,
    kappa: float,
    huber_delta: float,
) -> PhysicsLossResult:
    return one_sided_excess_loss(
        prediction_raw,
        arrays[f"{prefix}_fem_raw_residual"],
        arrays[f"{prefix}_fem_raw_scale"],
        arrays[f"{prefix}_stratum_floor"],
        arrays[f"{prefix}_stratum_mad"],
        arrays[f"{prefix}_valid_mask"],
        kappa=kappa,
        huber_delta=huber_delta,
    )


def compute_gp_physics_loss(
    physics_field: torch.Tensor,
    arrays: Mapping[str, torch.Tensor],
    physics_config: PhysicsLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate four integral operators on the predicted bipolar fields.

    ``physics_field`` uses the query union indexed by ``arrays``. FEM residuals,
    validity masks, scale floors, and MAD values were prepared before training.
    This function compares predicted residuals to that frozen calibration and
    returns the weighted excess loss plus per-operator diagnostics.
    """

    physics_config.validate()
    circ_raw, _ = circulation_raw_and_scale(
        physics_field,
        arrays["circ_point_index"].long(),
        arrays["circ_edge_vector_m"],
        arrays["circ_point_mask"].bool(),
    )
    normal_raw, _, tangent_raw, _ = interface_raw_and_scale(
        physics_field,
        arrays["interface_wm_point_index"].long(),
        arrays["interface_gm_point_index"].long(),
        arrays["interface_normal_wm_to_gm"],
    )
    flux_raw, _ = flux_raw_and_scale(
        physics_field,
        arrays["flux_face_point_index"].long(),
        arrays["flux_outward_area_vector_m2"],
        arrays["flux_conductivity_s_per_m"],
    )
    results = {
        "circ": _physics_result(
            circ_raw,
            arrays,
            "circ",
            kappa=physics_config.kappa,
            huber_delta=physics_config.huber_delta,
        ),
        "interface_normal": _physics_result(
            normal_raw,
            arrays,
            "interface_normal",
            kappa=physics_config.kappa,
            huber_delta=physics_config.huber_delta,
        ),
        "interface_tangent": _physics_result(
            tangent_raw,
            arrays,
            "interface_tangent",
            kappa=physics_config.kappa,
            huber_delta=physics_config.huber_delta,
        ),
        "flux": _physics_result(
            flux_raw,
            arrays,
            "flux",
            kappa=physics_config.kappa,
            huber_delta=physics_config.huber_delta,
        ),
    }
    return weighted_physics_sum(
        results,
        {
            "circ": physics_config.lambda_circ,
            "interface_normal": physics_config.lambda_interface_normal,
            "interface_tangent": physics_config.lambda_interface_tangent,
            "flux": physics_config.lambda_flux,
        },
        physics_field,
    )


def build_n128_gp_step_function(
    data_config: DataLossConfig,
    physics_config: PhysicsLossConfig,
) -> Callable[
    [torch.nn.Module, object, Mapping[str, torch.Generator], int],
    tuple[torch.Tensor, Mapping[str, object]],
]:
    """Bind the GP objective for use by the generic training runner.

    Each step shares one head encoding between data and physics branches, then
    returns ``data_loss + curriculum_scale * physics_loss``. The runner owns
    autocast, gradient scaling, backward, clipping, and optimizer updates.
    """

    data_config.validate()
    physics_config.validate()

    def step(
        raw_model: torch.nn.Module,
        raw_batch: object,
        generators: Mapping[str, torch.Generator],
        global_step: int,
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        if not isinstance(raw_model, Head2FieldV1):
            raise TypeError("N128 GP step requires Head2FieldV1")
        if raw_model.variant_id != "GP":
            raise ValueError("N128 GP step requires the GP variant")
        if not isinstance(raw_batch, Head2FieldTrainingBatch):
            raise TypeError("N128 GP step requires Head2FieldTrainingBatch")
        required_generators = {"augmentation", "dropout"}
        if set(generators) != required_generators:
            raise ValueError(
                "training step requires exactly the augmentation and dropout generators"
            )
        arrays = raw_batch.physics.arrays
        output = raw_model.forward_data_and_physics(
            raw_batch.head_volume,
            raw_batch.geometry,
            raw_batch.data_queries,
            raw_batch.physics.queries,
            raw_batch.electrodes,
            raw_batch.pair_index,
            data_augmentation_generator=generators["augmentation"],
            data_dropout_generator=generators["dropout"],
            validate_coordinates=True,
            physics_requires_grad=True,
        )
        data_loss, data_metrics = lead_field_data_loss(
            output.data.pair_field,
            raw_batch.target,
            raw_batch.data_loss_batch,
            denominator_floor_ratio=data_config.denominator_floor_ratio,
            pair_weight=data_config.pair_weight,
            radial_weight=data_config.radial_weight,
            peak_weight=data_config.peak_weight,
            huber_delta=data_config.huber_delta,
        )

        physics_loss, physics_metrics = compute_gp_physics_loss(
            output.physics.pair_field,
            arrays,
            physics_config,
        )
        scale = curriculum_scale(
            global_step,
            warmup_steps=physics_config.curriculum_warmup_steps,
            ramp_steps=physics_config.curriculum_ramp_steps,
        )
        total_loss = data_loss + float(scale) * physics_loss
        metrics: dict[str, object] = {
            "subject_key": raw_batch.subject_key,
            "data_loss": data_loss.detach(),
            "physics_loss": physics_loss.detach(),
            "physics_curriculum_scale": scale,
        }
        metrics.update(data_metrics)
        metrics.update(physics_metrics)
        return total_loss, metrics

    return step


__all__ = ["build_n128_gp_step_function", "compute_gp_physics_loss"]
