"""Two-subject end-to-end smoke tests for training commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from head2field.training.cli import preflight_main, train_main
from head2field.training.synthetic import write_synthetic_training_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_preflight_and_two_step_cli(tmp_path, capsys) -> None:
    manifest = write_synthetic_training_dataset(tmp_path / "dataset")
    config = PROJECT_ROOT / "configs/training_smoke_v1.json"

    preflight_main(["--config", str(config), "--synthetic-dataset", str(manifest)])
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["status"] == "PASS"
    assert preflight["subjects"] == {"train": 1, "validation": 1}
    assert preflight["official_weight_reproduction"] is False

    output = tmp_path / "run"
    train_main(
        [
            "--config",
            str(config),
            "--synthetic-dataset",
            str(manifest),
            "--output-dir",
            str(output),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["global_step"] == 2
    assert result["official_weight_reproduction"] is False
    log_rows = [
        json.loads(line)
        for line in (output / "training_log_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(log_rows) == 2
    assert all(row["validation"]["subject_count"] == 1 for row in log_rows)
    assert Path(result["best_checkpoint_path"]).is_file()
    best = torch.load(result["best_checkpoint_path"], weights_only=True)
    assert best and all(torch.is_tensor(tensor) for tensor in best.values())
    for row in log_rows:
        validation = row["validation"]
        assert validation["physics_curriculum_scale"] == 1.0
        assert validation["selection_policy"] == "complete_source_query_chunks_v1"
        assert validation["combined_loss_mean"] == pytest.approx(
            validation["data_loss_mean"] + validation["physics_loss_mean"]
        )
