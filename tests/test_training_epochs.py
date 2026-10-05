"""Check complete epochs, epoch-based cosine decay, early stopping, and resume."""

from __future__ import annotations

import json
import math

import pytest
import torch

from head2field.training.dataset import DeterministicHead2FieldSampler
from head2field.training.runner import (
    DeterministicIndexSampler,
    LoopConfig,
    run_training_loop,
)


def test_epoch_covers_every_source_query_pair_including_tail_batches() -> None:
    sizes = ((5, 75), (9, 75))
    sampler = DeterministicHead2FieldSampler(
        sizes, query_count=4, sources_per_step=8, seed=42
    )
    assert sampler.steps_per_epoch == 50
    for epoch in range(2):
        observed = set()
        for _ in range(sampler.steps_per_epoch):
            selection = sampler.next_selection()
            for source in selection.source_indices:
                for query in selection.query_indices:
                    key = (selection.subject_index, source, query)
                    assert key not in observed
                    observed.add(key)
        assert sampler.epoch == epoch
        assert observed == {
            (subject, source, query)
            for subject, (queries, sources) in enumerate(sizes)
            for source in range(sources)
            for query in range(queries)
        }


def test_sampler_resume_in_the_middle_of_a_query_chunk_cycle() -> None:
    kwargs = dict(query_count=4, sources_per_step=8, seed=42)
    sampler = DeterministicHead2FieldSampler(((9, 75),), **kwargs)
    for _ in range(7):
        sampler.next_selection()
    restored = DeterministicHead2FieldSampler(((9, 75),), **kwargs)
    restored.load_state_dict(sampler.state_dict())
    assert [sampler.next_selection() for _ in range(60)] == [
        restored.next_selection() for _ in range(60)
    ]


def _model() -> torch.nn.Module:
    return torch.nn.Linear(1, 1)


def _step(model, batch, generators, global_step):
    return model(torch.ones(1, 1)).square().mean(), {}


def _config() -> LoopConfig:
    return LoopConfig(
        total_steps=50,
        steps_per_epoch=2,
        seed=42,
        learning_rate=2e-4,
        min_learning_rate=0.0,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        state_interval_steps=7,
    )


def test_cosine_changes_only_after_complete_epochs_with_a_25_epoch_horizon(
    tmp_path,
) -> None:
    calls = []

    def validate(model, step):
        calls.append(step)
        return {"combined_loss_mean": 1.0 / step}

    result = run_training_loop(
        model=_model(),
        sampler=DeterministicIndexSampler(2, 42),
        batch_provider=lambda index: index,
        step_function=_step,
        loop_config=_config(),
        run_contract={"test": "epoch-cosine"},
        output_dir=tmp_path / "run",
        validation_callback=validate,
    )
    assert calls == list(range(2, 51, 2))
    for row in result.log_rows:
        completed_epochs = (row["step"] - 1) // 2
        expected = 2e-4 * (1 + math.cos(math.pi * completed_epochs / 25)) / 2
        assert row["learning_rate_used"] == pytest.approx(expected)
    assert result.log_rows[-1]["learning_rate"] == pytest.approx(0.0)
    assert result.best_epoch == 25


def test_five_epoch_early_stopping_and_resume_keep_the_best_checkpoint(
    tmp_path,
) -> None:
    snapshots = {}
    test_calls = []

    def validate(model, step):
        epoch = step // 2
        snapshots[epoch] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        return {"combined_loss_mean": 0.5 if epoch == 2 else 1.0}

    def evaluate_test(model, step):
        test_calls.append(step)
        assert not model.training
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, snapshots[2][name])
        return {"test_loss": 0.25}

    kwargs = dict(
        batch_provider=lambda index: index,
        step_function=_step,
        loop_config=_config(),
        run_contract={"test": "early-stop"},
        validation_callback=validate,
        test_callback=evaluate_test,
    )
    torch.manual_seed(9)
    initial = _model().state_dict()
    continuous_model = _model()
    continuous_model.load_state_dict(initial)
    continuous = run_training_loop(
        model=continuous_model,
        sampler=DeterministicIndexSampler(2, 42),
        output_dir=tmp_path / "continuous",
        **kwargs,
    )
    interrupted_model = _model()
    interrupted_model.load_state_dict(initial)
    interrupted = run_training_loop(
        model=interrupted_model,
        sampler=DeterministicIndexSampler(2, 42),
        output_dir=tmp_path / "part",
        stop_after_steps=7,
        **kwargs,
    )
    resumed = run_training_loop(
        model=interrupted_model,
        sampler=DeterministicIndexSampler(2, 42),
        output_dir=tmp_path / "resumed",
        resume_state=interrupted.state_path,
        trusted_local_resume_state=True,
        **kwargs,
    )
    assert continuous.global_step == resumed.global_step == 14
    assert continuous.best_epoch == resumed.best_epoch == 2
    assert continuous.stopped_early and resumed.stopped_early
    assert continuous.log_path.read_bytes() == resumed.log_path.read_bytes()
    assert interrupted.test_metrics_path is None
    assert test_calls == [14, 14]
    assert json.loads(resumed.test_metrics_path.read_text(encoding="utf-8")) == {
        "best_epoch": 2,
        "checkpoint": "best_model.pt",
        "metrics": {"test_loss": 0.25},
    }
    best = torch.load(resumed.best_checkpoint_path, weights_only=True)
    for name, tensor in best.items():
        assert torch.equal(tensor, snapshots[2][name])
        assert torch.equal(tensor, interrupted_model.state_dict()[name])
