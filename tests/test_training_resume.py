"""Resume-equivalence tests for tensors, optimizer, scheduler, RNGs, sampler, and logs."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping

import numpy as np
import pytest
import torch

from head2field.training.contracts import TrainingContractError
from head2field.training.runner import (
    DeterministicIndexSampler,
    LoopConfig,
    run_training_loop,
)
from head2field.training.state import load_training_state


def _model() -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        return torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 2),
        )


SAMPLES = tuple(
    (
        torch.tensor([index, index + 1, index + 2], dtype=torch.float32) / 10.0,
        torch.tensor([index - 1, 2 - index], dtype=torch.float32) / 7.0,
    )
    for index in range(5)
)


def _step(model, batch, generators, global_step):
    inputs, target = batch
    prediction = model(inputs)
    random_scale = 0.9 + 0.01 * random.random()
    numpy_scale = 0.9 + 0.01 * float(np.random.random())
    torch_scale = 0.9 + 0.01 * float(torch.rand(()))
    named_scale = 0.9 + 0.01 * float(torch.rand((), generator=generators["dropout"]))
    # 同时推进增强随机数生成器，验证恢复后状态一致。
    augmentation_probe = float(torch.rand((), generator=generators["augmentation"]))
    scale = random_scale * numpy_scale * torch_scale * named_scale
    loss = (prediction - target).square().mean() * scale
    return loss, {
        "global_step_before_update": global_step,
        "augmentation_probe": augmentation_probe,
    }


def _loop_config() -> LoopConfig:
    return LoopConfig(
        total_steps=6,
        seed=42,
        learning_rate=2e-3,
        min_learning_rate=1e-4,
        weight_decay=1e-4,
        steps_per_epoch=2,
        gradient_clip_norm=1.0,
        state_interval_steps=3,
    )


RUN_CONTRACT = {
    "schema_id": "resume_equivalence_test_v1",
    "configuration": {"learning_rate": 2e-3},
    "split": {"train": ["s1"], "validation": ["s2"]},
    "dataset": {"manifest_path": "synthetic.json"},
}


def _assert_nested_equal(first, second) -> None:
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, Mapping):
        assert set(first) == set(second)
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_interrupted_resume_is_deterministically_identical(tmp_path) -> None:
    continuous_model = _model()
    continuous = run_training_loop(
        model=continuous_model,
        sampler=DeterministicIndexSampler(len(SAMPLES), seed=42),
        batch_provider=lambda index: SAMPLES[index],
        step_function=_step,
        loop_config=_loop_config(),
        run_contract=RUN_CONTRACT,
        output_dir=tmp_path / "continuous",
    )

    resumed_model = _model()
    interrupted = run_training_loop(
        model=resumed_model,
        sampler=DeterministicIndexSampler(len(SAMPLES), seed=42),
        batch_provider=lambda index: SAMPLES[index],
        step_function=_step,
        loop_config=_loop_config(),
        run_contract=RUN_CONTRACT,
        output_dir=tmp_path / "interrupted",
        stop_after_steps=3,
    )
    resumed = run_training_loop(
        model=resumed_model,
        sampler=DeterministicIndexSampler(len(SAMPLES), seed=42),
        batch_provider=lambda index: SAMPLES[index],
        step_function=_step,
        loop_config=_loop_config(),
        run_contract=RUN_CONTRACT,
        output_dir=tmp_path / "resumed",
        resume_state=interrupted.state_path,
        trusted_local_resume_state=True,
    )

    for name, tensor in continuous_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name]), name
    continuous_state = torch.load(
        continuous.state_path, map_location="cpu", weights_only=False
    )
    resumed_state = torch.load(
        resumed.state_path, map_location="cpu", weights_only=False
    )
    for field in (
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "sampler",
        "log_rows",
        "training_control",
        "grad_scaler",
    ):
        _assert_nested_equal(continuous_state[field], resumed_state[field])
    assert continuous.log_path.read_bytes() == resumed.log_path.read_bytes()


def test_resume_rejects_changed_parameters_groups_or_dataset_paths(tmp_path) -> None:
    result = run_training_loop(
        model=_model(),
        sampler=DeterministicIndexSampler(len(SAMPLES), seed=42),
        batch_provider=lambda index: SAMPLES[index],
        step_function=_step,
        loop_config=_loop_config(),
        run_contract=RUN_CONTRACT,
        output_dir=tmp_path / "source",
        stop_after_steps=3,
    )
    for field in ("configuration", "split", "dataset"):
        changed = copy.deepcopy(RUN_CONTRACT)
        changed[field] = {"changed": True}
        with pytest.raises(TrainingContractError, match="训练参数、数据路径或分组"):
            load_training_state(
                result.state_path,
                expected_run_contract=changed,
                trusted_local_state=True,
            )


def test_init_weights_and_resume_state_are_not_interchangeable(tmp_path) -> None:
    pure_weights = tmp_path / "weights.pth"
    torch.save(_model().state_dict(), pure_weights)

    with pytest.raises(ValueError, match="互斥"):
        run_training_loop(
            model=_model(),
            sampler=DeterministicIndexSampler(len(SAMPLES), seed=42),
            batch_provider=lambda index: SAMPLES[index],
            step_function=_step,
            loop_config=_loop_config(),
            run_contract=RUN_CONTRACT,
            output_dir=tmp_path / "both",
            init_weights=pure_weights,
            resume_state=pure_weights,
            trusted_local_resume_state=True,
        )
    with pytest.raises(TrainingContractError, match="resume state 字段"):
        load_training_state(
            pure_weights,
            expected_run_contract=RUN_CONTRACT,
            trusted_local_state=True,
        )
