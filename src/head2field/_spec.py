"""Head2Field v1 specifications, fixed model factory, and constructor settings."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    HEAD2FIELD_V1_COORDINATE_FRAME_ID,
    HEAD2FIELD_V1_MODEL_VERSION,
    MODEL_VARIANTS,
    Head2FieldV1,
    architecture_id_for_variant,
)

PHYSICS_PROFILES = (
    "none",
    "circ",
    "interface",
    "circ_interface",
    "flux",
    "all",
)
PHYSICS_OBJECTIVES = (
    "none",
    "one_sided_excess_v1",
    "zero_target_diag",
    "residual_match_diag",
)
DEFAULT_VARIANT = "GP"
DEFAULT_PHYSICS_PROFILE = "all"
DEFAULT_PHYSICS_OBJECTIVE = "one_sided_excess_v1"

HEAD2FIELD_V1_DATA_LOSS_ID = "head2field_v1_stable_energy"
HEAD2FIELD_V1_PHYSICS_LOSS_ID = "head2field_v1_stable_energy_integral_physics"
MODEL_CONFIG_FIELDS = frozenset(
    {
        "input_channels",
        "base_channels",
        "use_1mm_features",
        "encoder_levels",
        "encoder_activation_checkpointing",
        "pyramid_dim",
        "query_dim",
        "query_blocks",
        "query_expansion",
        "electrode_dim",
        "decoder_blocks",
        "decoder_bottleneck_dim",
        "global_dim",
        "coordinate_fourier_bands",
        "context_dropout",
        "patch_radius_mm",
        "physical_coordinate_scale_mm",
        "absolute_fourier_bands",
        "absolute_aux_dim",
        "absolute_coordinate_dropout",
        "absolute_branch_dropout",
        "absolute_coordinate_jitter_probability",
        "absolute_coordinate_jitter_std_mm",
        "absolute_gate_initial_value",
        "variant_id",
        "global_mixer_dim",
        "global_mixer_blocks",
        "global_mixer_heads",
        "global_mixer_local_window",
        "global_mixer_max_relative_distance",
        "global_mixer_swiglu_expansion",
    }
)

# 论文第 2.3 节的架构参数；未在正文列出的实现细节在此显式记录。
PAPER_MODEL_CONFIG: dict[str, object] = {
    "input_channels": 12,
    "base_channels": 16,
    "use_1mm_features": True,
    "encoder_levels": 4,
    "encoder_activation_checkpointing": False,
    "pyramid_dim": 32,
    "query_dim": 256,
    "query_blocks": 2,
    "query_expansion": 2,
    "electrode_dim": 192,
    "decoder_blocks": 2,
    "decoder_bottleneck_dim": 128,
    "global_dim": 128,
    "coordinate_fourier_bands": 6,
    "context_dropout": 0.10,
    "patch_radius_mm": 6.0,
    "physical_coordinate_scale_mm": 128.0,
    "absolute_fourier_bands": 3,
    "absolute_aux_dim": 64,
    "absolute_coordinate_dropout": 0.20,
    "absolute_branch_dropout": 0.30,
    "absolute_coordinate_jitter_probability": 0.50,
    "absolute_coordinate_jitter_std_mm": 12.0,
    "absolute_gate_initial_value": 0.12,
    "variant_id": "GP",
    "global_mixer_dim": 96,
    "global_mixer_blocks": 2,
    "global_mixer_heads": 4,
    "global_mixer_local_window": 5,
    "global_mixer_max_relative_distance": 32,
    "global_mixer_swiglu_expansion": 2,
}


@dataclass(frozen=True)
class ModelContract:
    """Stable contract fields shared by the v1 model and training configuration."""

    model_version: str
    architecture_id: str
    coordinate_frame_id: str
    loss_id: str
    variant_id: str
    physics_schedule_profile_id: str
    physics_loss_profile_id: str
    physics_objective_id: str

    @property
    def physics_enabled(self) -> bool:
        """Return whether the current variant backpropagates through the physics branch."""

        return self.variant_id in ("P", "GP")


def _validate_composite(
    *,
    variant_id: str,
    physics_schedule_profile_id: str,
    physics_loss_profile_id: str,
    physics_objective_id: str,
) -> None:
    """Validate the model composite identifier without importing the training package."""

    if variant_id not in MODEL_VARIANTS:
        raise ValueError(f"Unknown v1 variant: {variant_id}")
    if physics_schedule_profile_id not in PHYSICS_PROFILES:
        raise ValueError(
            f"Unknown physics schedule profile: {physics_schedule_profile_id}"
        )
    if physics_loss_profile_id not in PHYSICS_PROFILES:
        raise ValueError(f"Unknown physics loss profile: {physics_loss_profile_id}")
    if physics_objective_id not in PHYSICS_OBJECTIVES:
        raise ValueError(f"Unknown physics objective: {physics_objective_id}")
    if variant_id in ("P", "GP"):
        if physics_schedule_profile_id == "none":
            raise ValueError("P/GP requires a physics schedule profile other than none")
        if physics_loss_profile_id != physics_schedule_profile_id:
            raise ValueError(
                "P/GP physics schedule and loss profiles must match exactly"
            )
        if physics_objective_id == "none":
            raise ValueError("P/GP requires an explicit physics objective")
    else:
        if physics_loss_profile_id != "none":
            raise ValueError(
                "C/G_LOCAL/G uses data supervision only; physics loss profile must be none"
            )
        if physics_objective_id != "none":
            raise ValueError("C/G_LOCAL/G physics objective must be none")


def get_model_contract(
    *,
    variant_id: str | None = None,
    physics_schedule_profile_id: str | None = None,
    physics_loss_profile_id: str | None = None,
    physics_objective_id: str | None = None,
) -> ModelContract:
    """Resolve the v1 composite, using the frozen official GP defaults when omitted."""

    use_complete_default = variant_id is None
    resolved_variant = (
        DEFAULT_VARIANT if variant_id is None else str(variant_id).upper()
    )
    resolved_schedule = (
        DEFAULT_PHYSICS_PROFILE
        if physics_schedule_profile_id is None and use_complete_default
        else (physics_schedule_profile_id or "none")
    )
    resolved_loss = (
        DEFAULT_PHYSICS_PROFILE
        if physics_loss_profile_id is None and use_complete_default
        else (physics_loss_profile_id or "none")
    )
    resolved_objective = (
        DEFAULT_PHYSICS_OBJECTIVE
        if physics_objective_id is None and use_complete_default
        else (physics_objective_id or "none")
    )
    _validate_composite(
        variant_id=resolved_variant,
        physics_schedule_profile_id=resolved_schedule,
        physics_loss_profile_id=resolved_loss,
        physics_objective_id=resolved_objective,
    )
    physics_enabled = resolved_variant in ("P", "GP")
    return ModelContract(
        model_version=HEAD2FIELD_V1_MODEL_VERSION,
        architecture_id=architecture_id_for_variant(resolved_variant),
        coordinate_frame_id=HEAD2FIELD_V1_COORDINATE_FRAME_ID,
        loss_id=(
            HEAD2FIELD_V1_PHYSICS_LOSS_ID
            if physics_enabled
            else HEAD2FIELD_V1_DATA_LOSS_ID
        ),
        variant_id=resolved_variant,
        physics_schedule_profile_id=resolved_schedule,
        physics_loss_profile_id=resolved_loss,
        physics_objective_id=resolved_objective,
    )


def build_model_config(
    contract: ModelContract,
    *,
    input_channels: int = 12,
    base_channels: int,
    use_1mm_features: bool = True,
    encoder_levels: int,
    encoder_activation_checkpointing: bool,
    pyramid_dim: int = 32,
    query_dim: int = 256,
    query_blocks: int = 2,
    query_expansion: int = 2,
    electrode_dim: int = 192,
    decoder_blocks: int = 2,
    decoder_bottleneck_dim: int = 128,
    global_dim: int = 128,
    coordinate_fourier_bands: int = 6,
    context_dropout: float,
    patch_radius_mm: float,
    physical_coordinate_scale_mm: float,
    absolute_fourier_bands: int = 3,
    absolute_aux_dim: int = 64,
    absolute_coordinate_dropout: float = 0.20,
    absolute_branch_dropout: float = 0.30,
    absolute_coordinate_jitter_probability: float = 0.50,
    absolute_coordinate_jitter_std_mm: float = 12.0,
    absolute_gate_initial_value: float = 0.12,
    global_mixer_dim: int = 96,
    global_mixer_blocks: int = 2,
    global_mixer_heads: int = 4,
    global_mixer_local_window: int = 5,
    global_mixer_max_relative_distance: int = 32,
    global_mixer_swiglu_expansion: int = 2,
) -> dict[str, object]:
    """Build a complete configuration accepted by ``Head2FieldV1``."""

    config: dict[str, object] = {
        "input_channels": input_channels,
        "base_channels": base_channels,
        "use_1mm_features": use_1mm_features,
        "encoder_levels": encoder_levels,
        "encoder_activation_checkpointing": encoder_activation_checkpointing,
        "pyramid_dim": pyramid_dim,
        "query_dim": query_dim,
        "query_blocks": query_blocks,
        "query_expansion": query_expansion,
        "electrode_dim": electrode_dim,
        "decoder_blocks": decoder_blocks,
        "decoder_bottleneck_dim": decoder_bottleneck_dim,
        "global_dim": global_dim,
        "coordinate_fourier_bands": coordinate_fourier_bands,
        "context_dropout": context_dropout,
        "patch_radius_mm": patch_radius_mm,
        "physical_coordinate_scale_mm": physical_coordinate_scale_mm,
        "absolute_fourier_bands": absolute_fourier_bands,
        "absolute_aux_dim": absolute_aux_dim,
        "absolute_coordinate_dropout": absolute_coordinate_dropout,
        "absolute_branch_dropout": absolute_branch_dropout,
        "absolute_coordinate_jitter_probability": (
            absolute_coordinate_jitter_probability
        ),
        "absolute_coordinate_jitter_std_mm": absolute_coordinate_jitter_std_mm,
        "absolute_gate_initial_value": absolute_gate_initial_value,
        "variant_id": contract.variant_id,
        "global_mixer_dim": global_mixer_dim,
        "global_mixer_blocks": global_mixer_blocks,
        "global_mixer_heads": global_mixer_heads,
        "global_mixer_local_window": global_mixer_local_window,
        "global_mixer_max_relative_distance": global_mixer_max_relative_distance,
        "global_mixer_swiglu_expansion": global_mixer_swiglu_expansion,
    }
    if config.keys() != MODEL_CONFIG_FIELDS:
        raise AssertionError(
            "model configuration schema does not match the constructor"
        )
    return config


def create_head2field_v1() -> Head2FieldV1:
    """Create the official v1 model using the frozen N128 GP contract."""

    if PAPER_MODEL_CONFIG.keys() != MODEL_CONFIG_FIELDS:
        raise AssertionError("fixed v1 factory configuration fields are incomplete")
    return Head2FieldV1(**dict(PAPER_MODEL_CONFIG))


__all__ = [
    "DEFAULT_PHYSICS_OBJECTIVE",
    "DEFAULT_PHYSICS_PROFILE",
    "DEFAULT_VARIANT",
    "PAPER_MODEL_CONFIG",
    "HEAD2FIELD_V1_DATA_LOSS_ID",
    "HEAD2FIELD_V1_PHYSICS_LOSS_ID",
    "MODEL_CONFIG_FIELDS",
    "PHYSICS_OBJECTIVES",
    "PHYSICS_PROFILES",
    "ModelContract",
    "build_model_config",
    "create_head2field_v1",
    "get_model_contract",
]
