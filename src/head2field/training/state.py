"""Transactional local training-state writes, strict reload, and RNG restoration."""

from __future__ import annotations

import copy
import os
import random
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import TRAINING_STATE_SCHEMA_ID, TrainingContractError

STATE_FIELDS = frozenset(
    {
        "schema_id",
        "run_contract",
        "global_step",
        "model",
        "optimizer",
        "scheduler",
        "sampler",
        "rng",
        "log_rows",
        "grad_scaler",
        "training_control",
    }
)


def _cpu_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Copy model parameters so saved states do not reference mutable live tensors."""

    return OrderedDict(
        (name, tensor.detach().to(device="cpu").clone())
        for name, tensor in model.state_dict().items()
    )


def capture_rng_state(
    generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    """Capture Python, NumPy, Torch, and named Torch-generator states."""

    named = {} if generators is None else dict(generators)
    return {
        "python_random_state": copy.deepcopy(random.getstate()),
        "numpy_random_state": copy.deepcopy(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all()
        if any(g.device.type == "cuda" for g in named.values())
        else [],
        "named_torch_generators": {
            name: generator.get_state().clone()
            for name, generator in sorted(named.items())
        },
    }


def restore_rng_state(
    state: Mapping[str, Any],
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    """Restore all RNG states strictly, rejecting added or removed generator names."""

    expected = {
        "python_random_state",
        "numpy_random_state",
        "torch_rng_state",
        "torch_cuda_rng_states",
        "named_torch_generators",
    }
    if set(state) != expected:
        raise TrainingContractError(
            f"RNG state 字段不匹配: expected={sorted(expected)}, actual={sorted(state)}"
        )
    raw_named = state["named_torch_generators"]
    if not isinstance(raw_named, Mapping):
        raise TrainingContractError("named_torch_generators 必须是映射")
    named = {} if generators is None else dict(generators)
    if set(raw_named) != set(named):
        raise TrainingContractError(
            "resume 的命名 Torch generator 集合与当前运行不一致"
        )
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(state["torch_rng_state"], dtype=torch.uint8))
    if state["torch_cuda_rng_states"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
    for name, generator in named.items():
        generator.set_state(
            torch.as_tensor(raw_named[name], dtype=torch.uint8, device="cpu")
        )


def build_training_state(
    *,
    run_contract: Mapping[str, Any],
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler_state: Mapping[str, Any],
    generators: Mapping[str, torch.Generator],
    log_rows: list[dict[str, Any]],
    grad_scaler_state: Mapping[str, Any],
    training_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete local training state distinct from published bare weights."""

    if global_step < 0:
        raise ValueError("global_step 不能为负数")
    return {
        "schema_id": TRAINING_STATE_SCHEMA_ID,
        "run_contract": copy.deepcopy(dict(run_contract)),
        "global_step": int(global_step),
        "model": _cpu_state_dict(model),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "sampler": copy.deepcopy(dict(sampler_state)),
        "rng": capture_rng_state(generators),
        "log_rows": copy.deepcopy(log_rows),
        "grad_scaler": copy.deepcopy(dict(grad_scaler_state)),
        "training_control": copy.deepcopy(dict(training_control)),
    }


def atomic_save_training_state(state: Mapping[str, Any], path: Path) -> Path:
    """Write and fsync a same-directory temporary file, then promote atomically without replacement."""

    destination = Path(path).resolve(strict=False)
    if destination.suffix != ".pt":
        raise ValueError("本地训练状态必须使用 .pt 后缀")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有训练状态: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(dict(state), temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if destination.exists():
            raise FileExistsError(f"拒绝覆盖已有训练状态: {destination}")
        os.link(temporary, destination)
        os.unlink(temporary)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_training_state(
    path: Path,
    *,
    expected_run_contract: Mapping[str, Any],
    trusted_local_state: bool,
) -> dict[str, Any]:
    """Reload and validate local pickle state only after explicit caller trust."""

    if not trusted_local_state:
        raise PermissionError(
            "完整 resume state 使用 pickle；仅对自己生成且信任的本地文件设置显式信任"
        )
    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"resume state 必须是普通非软链接文件: {resolved}")
    value = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TrainingContractError("resume state 顶层必须是对象")
    if set(value) != STATE_FIELDS:
        raise TrainingContractError(
            f"resume state 字段不匹配: expected={sorted(STATE_FIELDS)}, actual={sorted(value)}"
        )
    if value.get("schema_id") != TRAINING_STATE_SCHEMA_ID:
        raise TrainingContractError(
            "文件不是 v1 完整本地训练状态；不能当作 resume state"
        )
    actual_contract = value.get("run_contract")
    if not isinstance(actual_contract, Mapping):
        raise TrainingContractError("resume state 缺少 run_contract")
    expected = dict(expected_run_contract)
    if dict(actual_contract) != expected:
        keys = sorted(set(actual_contract) | set(expected))
        mismatched = {
            key: (actual_contract.get(key), expected.get(key))
            for key in keys
            if actual_contract.get(key) != expected.get(key)
        }
        raise TrainingContractError(
            f"resume run contract 不匹配；训练参数、数据路径或分组已改变: {mismatched}"
        )
    if (
        isinstance(value["global_step"], bool)
        or not isinstance(value["global_step"], int)
        or value["global_step"] < 0
    ):
        raise TrainingContractError("resume global_step 必须是非负整数")
    if not isinstance(value["model"], Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value["model"].items()
    ):
        raise TrainingContractError("resume model 必须是 str -> Tensor state_dict")
    if not isinstance(value["log_rows"], list) or any(
        not isinstance(row, dict) for row in value["log_rows"]
    ):
        raise TrainingContractError("resume log_rows 必须是对象数组")
    return copy.deepcopy(dict(value))


def restore_training_state(
    state: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: object,
    generators: Mapping[str, torch.Generator],
) -> tuple[int, list[dict[str, Any]]]:
    """Strictly restore model, optimizer, scheduler, sampler, RNGs, and existing logs."""

    incompatible = model.load_state_dict(state["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TrainingContractError("resume model strict load 后仍有键不匹配")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    load_sampler = getattr(sampler, "load_state_dict", None)
    if not callable(load_sampler):
        raise TypeError("sampler 必须实现 load_state_dict")
    load_sampler(state["sampler"])
    restore_rng_state(state["rng"], generators)
    return int(state["global_step"]), copy.deepcopy(state["log_rows"])


__all__ = [
    "STATE_FIELDS",
    "atomic_save_training_state",
    "build_training_state",
    "capture_rng_state",
    "load_training_state",
    "restore_rng_state",
    "restore_training_state",
]
