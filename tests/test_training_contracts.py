"""Tests separating public training execution from historical source-weight provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from head2field.training.contracts import TrainingConfig, TrainingContractError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/training_v1.json"


def test_training_config_matches_the_paper() -> None:
    config = TrainingConfig.load(CONFIG_PATH)

    assert config.execution_profile_id == "cuda_amp_fp16_epoch_training"
    assert config.device == "cuda"
    assert config.precision == "amp_fp16"
    assert config.max_epochs == config.cosine_horizon_epochs == 25
    assert config.early_stopping_patience == 5
    assert config.query_count == 49152
    assert config.sources_per_step == 8
    assert config.learning_rate == 2e-4
    assert config.weight_decay == 1e-4
    assert config.scheduler == "cosine"
    assert config.data_loss.radial_edges_mm == (50.0, 100.0)
    assert config.data_loss.denominator_rule_id == (
        "target_energy_plus_0.01_subject_source_median_v1"
    )
    assert config.gradient_clip_norm == 1.0
    assert config.official_weight_reproduction is False
    assert config.physics_loss.profile == "all"
    assert config.physics_loss.circulation_primitives == 512
    assert config.physics_loss.lambda_flux == 5.0



@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "official_weight_reproduction", True),
        ("physics_loss", "lambda_circ", 4.0),
        ("data_loss", "radial_edges_mm", [20.0, 50.0]),
    ],
)
def test_training_config_rejects_contract_drift(
    section: str | None,
    field: str,
    value: object,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target = payload if section is None else payload[section]
    target[field] = value

    with pytest.raises(TrainingContractError):
        TrainingConfig.from_mapping(payload)
