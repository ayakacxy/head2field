"""Epoch-scheduled training, CUDA FP16, validation early stopping, and complete resume."""

from __future__ import annotations

import copy
import json
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .contracts import TRAINING_LOG_SCHEMA_ID, TrainingConfig, TrainingContractError
from .state import (
    atomic_save_training_state,
    build_training_state,
    load_training_state,
    restore_training_state,
)


class StatefulSampler(Protocol):
    """Minimal deterministic sampler interface required by the training loop."""

    def next_selection(self) -> object: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


StepFunction = Callable[
    [torch.nn.Module, object, Mapping[str, torch.Generator], int],
    tuple[torch.Tensor, Mapping[str, object]],
]
BatchProvider = Callable[[object], object]
ValidationCallback = Callable[[torch.nn.Module, int], Mapping[str, object]]


@dataclass(frozen=True)
class LoopConfig:
    """Derive epoch length from all data combinations and update LR only at epoch boundaries."""

    total_steps: int
    seed: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    state_interval_steps: int
    steps_per_epoch: int = 1
    cosine_horizon_epochs: int = 25
    early_stopping_patience: int = 5
    device: str = "cpu"
    precision: str = "fp32"
    amp_initial_scale: float = 1024.0
    amp_growth_interval: int = 2000

    @classmethod
    def from_training_config(
        cls, config: TrainingConfig, *, steps_per_epoch: int
    ) -> LoopConfig:
        return cls(
            total_steps=config.max_epochs * steps_per_epoch,
            seed=config.seed,
            learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            state_interval_steps=config.state_interval_steps,
            steps_per_epoch=steps_per_epoch,
            cosine_horizon_epochs=config.cosine_horizon_epochs,
            early_stopping_patience=config.early_stopping_patience,
            device=config.device,
            precision=config.precision,
            amp_initial_scale=config.amp_initial_scale,
            amp_growth_interval=config.amp_growth_interval,
        )

    def validate(self) -> None:
        if self.total_steps <= 0 or self.seed < 0 or self.steps_per_epoch <= 0:
            raise ValueError("训练步数和 epoch 长度必须为正，seed 必须非负")
        if (
            self.total_steps > 25 * self.steps_per_epoch
            or self.cosine_horizon_epochs != 25
        ):
            raise ValueError("训练最多 25 epochs，cosine horizon 固定为 25")
        if (
            self.learning_rate <= 0.0
            or not 0 <= self.min_learning_rate <= self.learning_rate
        ):
            raise ValueError("learning rate 范围无效")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("weight_decay/gradient_clip_norm 合同无效")
        if self.state_interval_steps <= 0 or self.early_stopping_patience != 5:
            raise ValueError("保存间隔必须为正，早停 patience 必须为 5")
        if self.precision not in {"fp32", "amp_fp16"} or self.device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError("训练 device/precision 无效")
        if self.precision == "amp_fp16" and self.device != "cuda":
            raise ValueError("AMP FP16 训练需要 CUDA")


@dataclass(frozen=True)
class TrainingRunResult:
    """Result of a complete or deliberately interrupted training run."""

    output_dir: Path
    global_step: int
    state_path: Path
    log_path: Path
    log_rows: tuple[Mapping[str, Any], ...]
    best_checkpoint_path: Path | None
    best_epoch: int | None
    stopped_early: bool
    test_metrics_path: Path | None


class DeterministicIndexSampler:
    """Deterministic sampler retaining permutation, cursor, and NumPy bit-generator state."""

    schema_id = "deterministic_index_sampler_v1"

    def __init__(self, length: int, seed: int) -> None:
        if length <= 0 or seed < 0:
            raise ValueError("sampler length 必须为正且 seed 必须非负")
        self.length = int(length)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.epoch = -1
        self.selections_emitted = 0

    def next_selection(self) -> int:
        if self.cursor >= len(self.order):
            self.order = self.rng.permutation(self.length).astype(np.int64, copy=False)
            self.cursor = 0
            self.epoch += 1
        value = int(self.order[self.cursor])
        self.cursor += 1
        self.selections_emitted += 1
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "length": self.length,
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state,
            "order": self.order.copy(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "selections_emitted": self.selections_emitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "schema_id",
            "length",
            "seed",
            "rng_state",
            "order",
            "cursor",
            "epoch",
            "selections_emitted",
        }
        if set(state) != required:
            raise TrainingContractError("sampler state 字段不匹配")
        if state["schema_id"] != self.schema_id:
            raise TrainingContractError("sampler state schema 不匹配")
        if int(state["length"]) != self.length or int(state["seed"]) != self.seed:
            raise TrainingContractError("sampler length/seed 与当前运行不一致")
        order = np.asarray(state["order"])
        if order.dtype.kind not in "iu" or order.ndim != 1:
            raise TrainingContractError("sampler order 必须是一维整数数组")
        if len(order) not in (0, self.length):
            raise TrainingContractError("sampler order 长度不合法")
        if len(order) and not np.array_equal(np.sort(order), np.arange(self.length)):
            raise TrainingContractError("sampler order 不是完整 permutation")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= len(order):
            raise TrainingContractError("sampler cursor 越界")
        self.rng.bit_generator.state = state["rng_state"]
        self.order = order.astype(np.int64, copy=True)
        self.cursor = cursor
        self.epoch = int(state["epoch"])
        self.selections_emitted = int(state["selections_emitted"])


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: LoopConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Build AdamW with cosine annealing over a fixed 25-epoch horizon."""
    config.validate()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.cosine_horizon_epochs,
        eta_min=config.min_learning_rate,
    )
    return optimizer, scheduler


def move_to_device(value: object, device: torch.device) -> object:
    """Move tensors in immutable batches recursively, preserving geometry and other fields."""
    if torch.is_tensor(value):
        return value.to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                f.name: move_to_device(getattr(value, f.name), device)
                for f in fields(value)
                if f.init
            },
        )
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def _selection_log(selection: object) -> object:
    """Log exact query-index hashes instead of printing tens of thousands of indices per step."""
    if hasattr(selection, "query_indices") and hasattr(selection, "source_indices"):
        indices = np.asarray(selection.query_indices, dtype="<i8")
        return {
            "subject_index": selection.subject_index,
            "source_indices": list(selection.source_indices),
            "query_count": len(indices),
        }
    return _json_value(selection)


def load_initial_weights(model: torch.nn.Module, path: Path) -> None:
    """Accept only a bare ``str -> Tensor`` mapping without restoring training state."""

    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"init weights 必须是普通非软链接文件: {resolved}")
    value = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping) or not value:
        raise TrainingContractError("--init-weights 顶层必须是非空纯 state_dict")
    if not all(
        isinstance(key, str) and torch.is_tensor(item) for key, item in value.items()
    ):
        raise TrainingContractError(
            "--init-weights 只能包含 str -> Tensor；完整训练状态必须走 --resume-state"
        )
    incompatible = model.load_state_dict(value, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TrainingContractError("init weights strict load 后仍有键不匹配")


def _json_value(value: object) -> object:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError("训练日志 tensor 必须是标量")
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    """Exclusively write and flush the complete canonical JSONL log."""

    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def _append_log_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def run_training_loop(
    *,
    model: torch.nn.Module,
    sampler: StatefulSampler,
    batch_provider: BatchProvider,
    step_function: StepFunction,
    loop_config: LoopConfig,
    run_contract: Mapping[str, Any],
    output_dir: Path,
    init_weights: Path | None = None,
    resume_state: Path | None = None,
    trusted_local_resume_state: bool = False,
    stop_after_steps: int | None = None,
    validation_callback: ValidationCallback | None = None,
    test_callback: ValidationCallback | None = None,
) -> TrainingRunResult:
    """Run interruptible training with identical complete log rows after resume."""

    loop_config.validate()
    if init_weights is not None and resume_state is not None:
        raise ValueError("--init-weights 与 --resume-state 互斥")
    stop_step = (
        loop_config.total_steps if stop_after_steps is None else stop_after_steps
    )
    if not 0 < stop_step <= loop_config.total_steps:
        raise ValueError("stop_after_steps 必须位于 (0,total_steps]")
    resolved_output = Path(output_dir).resolve(strict=False)
    if resolved_output.exists():
        raise FileExistsError(f"训练输出目录必须尚不存在: {resolved_output}")

    random.seed(loop_config.seed)
    np.random.seed(loop_config.seed)
    torch.manual_seed(loop_config.seed)
    device = torch.device(loop_config.device)
    amp_enabled = loop_config.precision == "amp_fp16"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 训练配置需要可用的 GPU")
    model.to(device=device, dtype=torch.float32)
    model.train()
    if init_weights is not None:
        load_initial_weights(model, init_weights)
    optimizer, scheduler = build_optimizer_and_scheduler(model, loop_config)
    generators = {
        "augmentation": torch.Generator(device=device).manual_seed(
            loop_config.seed + 1_000_003
        ),
        "dropout": torch.Generator(device=device).manual_seed(
            loop_config.seed + 2_000_003
        ),
    }

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
            init_scale=loop_config.amp_initial_scale,
            growth_interval=loop_config.amp_growth_interval,
        )
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
            init_scale=loop_config.amp_initial_scale,
            growth_interval=loop_config.amp_growth_interval,
        )
    control = {
        "best_validation_loss": None,
        "best_epoch": None,
        "epochs_without_improvement": 0,
        "best_model": None,
    }
    global_step = 0
    log_rows: list[dict[str, Any]] = []
    if resume_state is not None:
        state = load_training_state(
            resume_state,
            expected_run_contract=run_contract,
            trusted_local_state=trusted_local_resume_state,
        )
        global_step, log_rows = restore_training_state(
            state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            generators=generators,
        )
        control = copy.deepcopy(state["training_control"])
        if state["grad_scaler"]:
            scaler.load_state_dict(state["grad_scaler"])
        if control["epochs_without_improvement"] >= loop_config.early_stopping_patience:
            raise TrainingContractError("恢复状态已经触发早停")
        if global_step != len(log_rows):
            raise TrainingContractError("resume step 与日志行数不一致")
        if global_step >= stop_step:
            raise TrainingContractError("resume step 已达到或超过本次 stop step")

    try:
        resolved_output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"训练输出目录必须尚不存在: {resolved_output}") from error
    log_path = resolved_output / "training_log_v1.jsonl"
    _write_log(log_path, log_rows)
    latest_state_path: Path | None = None
    while global_step < stop_step:
        selection = sampler.next_selection()
        batch = move_to_device(batch_provider(selection), device)
        optimizer.zero_grad(set_to_none=True)
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            loss, raw_metrics = step_function(model, batch, generators, global_step)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("训练 step loss 必须是有限标量")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), loop_config.gradient_clip_norm
        )
        gradient_is_finite = bool(torch.isfinite(torch.as_tensor(gradient_norm)))
        if not gradient_is_finite and not amp_enabled:
            raise FloatingPointError("gradient norm 含 NaN/Inf")
        if gradient_is_finite:
            scaler.step(optimizer)
            scaler.update()
        else:
            # Skip an AMP overflow and reduce the scale before the next batch.
            scaler.update(new_scale=scaler.get_scale() / 2.0)
        global_step += 1
        epoch_complete = global_step % loop_config.steps_per_epoch == 0
        epoch = (global_step - 1) // loop_config.steps_per_epoch + 1
        validation_metrics: object = None
        if epoch_complete:
            if validation_callback is not None:
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    validation_metrics = _json_value(
                        validation_callback(model, global_step)
                    )
                metric = validation_metrics.get("combined_loss_mean")
                if not isinstance(metric, (float, int)) or not np.isfinite(metric):
                    raise FloatingPointError("验证必须返回有限 combined_loss_mean")
                if (
                    control["best_validation_loss"] is None
                    or metric < control["best_validation_loss"]
                ):
                    control["best_validation_loss"] = float(metric)
                    control["best_epoch"] = epoch
                    control["epochs_without_improvement"] = 0
                    control["best_model"] = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                else:
                    control["epochs_without_improvement"] += 1
            scheduler.step()
        stopped_early = (
            control["epochs_without_improvement"] >= loop_config.early_stopping_patience
        )
        row = {
            "schema_id": TRAINING_LOG_SCHEMA_ID,
            "step": global_step,
            "selection": _selection_log(selection),
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "learning_rate_used": learning_rate_used,
            "best_validation_loss": control["best_validation_loss"],
            "epochs_without_improvement": control["epochs_without_improvement"],
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu())
            if gradient_is_finite
            else None,
            "optimizer_step_skipped": not gradient_is_finite,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "metrics": _json_value(raw_metrics),
            "validation": validation_metrics,
        }
        log_rows.append(row)
        _append_log_row(log_path, row)

        save_now = (
            global_step % loop_config.state_interval_steps == 0
            or global_step == stop_step
            or epoch_complete
            or stopped_early
        )
        if save_now:
            state = build_training_state(
                run_contract=run_contract,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler_state=sampler.state_dict(),
                generators=generators,
                log_rows=log_rows,
                grad_scaler_state=scaler.state_dict(),
                training_control=control,
            )
            latest_state_path = atomic_save_training_state(
                state,
                resolved_output / f"train_state_step_{global_step:09d}.pt",
            )

        if stopped_early:
            break

    best_checkpoint_path = None
    if control["best_model"] is not None:
        best_checkpoint_path = atomic_save_training_state(
            control["best_model"], resolved_output / "best_model.pt"
        )
        model.load_state_dict(control["best_model"], strict=True)
    test_metrics_path = None
    training_complete = global_step == loop_config.total_steps or stopped_early
    if test_callback is not None and training_complete:
        if best_checkpoint_path is None:
            raise TrainingContractError("test evaluation 需要验证选出的最佳模型")
        # Test evaluation runs once, after restoring the validation-selected model.
        model.eval()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ),
        ):
            test_metrics = _json_value(test_callback(model, global_step))
        test_metrics_path = resolved_output / "test_metrics_v1.json"
        with test_metrics_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "best_epoch": control["best_epoch"],
                    "checkpoint": "best_model.pt",
                    "metrics": test_metrics,
                },
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
    if latest_state_path is None:
        raise AssertionError("训练结束时必须已经写入本地状态")
    return TrainingRunResult(
        output_dir=resolved_output,
        global_step=global_step,
        state_path=latest_state_path,
        log_path=log_path,
        log_rows=tuple(log_rows),
        best_checkpoint_path=best_checkpoint_path,
        best_epoch=control["best_epoch"],
        stopped_early=stopped_early,
        test_metrics_path=test_metrics_path,
    )


__all__ = [
    "BatchProvider",
    "DeterministicIndexSampler",
    "LoopConfig",
    "StatefulSampler",
    "StepFunction",
    "TrainingRunResult",
    "ValidationCallback",
    "build_optimizer_and_scheduler",
    "load_initial_weights",
    "move_to_device",
    "run_training_loop",
]
