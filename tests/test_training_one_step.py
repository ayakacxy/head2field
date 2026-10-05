"""One-step CPU FP32 test of a real small TES GP network."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from head2field.training.contracts import TrainingConfig, canonical_json_sha256
from head2field.training.dataset import DeterministicHead2FieldSampler
from head2field.training.runner import LoopConfig, run_training_loop
from head2field.training.steps import build_n128_gp_step_function
from head2field.training.synthetic import (
    SyntheticHead2FieldDataset,
    create_synthetic_smoke_model,
    load_synthetic_training_dataset,
    write_synthetic_training_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cpu_fp32_small_model_completes_one_gp_step(tmp_path) -> None:
    config = TrainingConfig.load(PROJECT_ROOT / "configs/training_smoke_v1.json")
    manifest = write_synthetic_training_dataset(tmp_path / "dataset")
    subjects = load_synthetic_training_dataset(manifest)
    dataset = SyntheticHead2FieldDataset(subjects, partition="train")
    sampler = DeterministicHead2FieldSampler(
        dataset.subject_sizes,
        query_count=config.query_count,
        sources_per_step=config.sources_per_step,
        seed=config.seed,
    )
    model = create_synthetic_smoke_model()
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    run_contract = {
        "schema_id": "synthetic_training_run_contract_v1",
        "config_sha256": config.sha256,
        "split_sha256": canonical_json_sha256(["synthetic-train-a"]),
        "data_sha256": canonical_json_sha256(manifest.read_text(encoding="utf-8")),
        "official_weight_reproduction": False,
    }
    result = run_training_loop(
        model=model,
        sampler=sampler,
        batch_provider=dataset.batch,
        step_function=build_n128_gp_step_function(
            config.data_loss, config.physics_loss
        ),
        loop_config=LoopConfig.from_training_config(
            config, steps_per_epoch=sampler.steps_per_epoch
        ),
        run_contract=run_contract,
        output_dir=tmp_path / "run",
        stop_after_steps=1,
    )

    assert result.global_step == 1
    assert result.state_path.is_file()
    assert result.log_path.is_file()
    assert result.log_rows[0]["metrics"]["subject_key"] == "synthetic-train-a"
    assert result.log_rows[0]["metrics"]["physics_curriculum_scale"] == 0.0
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in model.state_dict().items()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_cuda_fp16_small_model_completes_gp_training_and_persists_amp_state(
    tmp_path,
) -> None:
    config = TrainingConfig.load(PROJECT_ROOT / "configs/training_smoke_v1.json")
    manifest = write_synthetic_training_dataset(tmp_path / "dataset")
    dataset = SyntheticHead2FieldDataset(
        load_synthetic_training_dataset(manifest), partition="train"
    )
    sampler = DeterministicHead2FieldSampler(
        dataset.subject_sizes,
        query_count=config.query_count,
        sources_per_step=8,
        seed=config.seed,
    )
    model = create_synthetic_smoke_model()
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    loop = replace(
        LoopConfig.from_training_config(
            config, steps_per_epoch=sampler.steps_per_epoch
        ),
        device="cuda",
        precision="amp_fp16",
    )
    result = run_training_loop(
        model=model,
        sampler=sampler,
        batch_provider=dataset.batch,
        step_function=build_n128_gp_step_function(
            config.data_loss, config.physics_loss
        ),
        loop_config=loop,
        run_contract={"test": "cuda-amp-fp16"},
        output_dir=tmp_path / "run",
    )
    assert next(model.parameters()).is_cuda
    assert all(row["loss"] >= 0.0 for row in result.log_rows)
    assert any(not row["optimizer_step_skipped"] for row in result.log_rows)
    assert any(
        not torch.equal(before[name], tensor.cpu())
        for name, tensor in model.state_dict().items()
    )
    state = torch.load(result.state_path, weights_only=False)
    assert state["grad_scaler"]["scale"] > 0
    assert state["rng"]["torch_cuda_rng_states"]
