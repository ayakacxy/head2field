"""Strict validation and loading of bare Head2Field v1 weights."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch

from ._spec import create_head2field_v1
from .model import Head2FieldV1
from .resource_paths import materialized_resource

V1_STATE_DICT_TENSOR_COUNT = 208
V1_STATE_DICT_VALUE_COUNT = 2_607_870
V1_WEIGHTS_FILENAME = "head2field_v1_weights.pth"


@contextmanager
def materialized_head2field_v1_weights(path: Path | None = None) -> Iterator[Path]:
    """选择自定义或随包权重；实际加载时核对论文模型的张量结构。"""
    if path is not None:
        yield Path(path)
        return
    with materialized_resource("weights", V1_WEIGHTS_FILENAME) as bundled:
        yield bundled


def validate_pure_v1_state_dict(
    value: object,
) -> OrderedDict[str, torch.Tensor]:
    """Validate a bare ``str -> FP32 Tensor`` mapping, rejecting composite wrappers."""

    if not isinstance(value, Mapping) or not value:
        raise TypeError("v1 weights must be a non-empty str -> Tensor mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise TypeError("every v1 weight key must be a non-empty string")
    non_tensors = sorted(
        key for key, item in value.items() if not torch.is_tensor(item)
    )
    if non_tensors:
        raise TypeError(f"v1 weights contain a non-Tensor field: {non_tensors[:5]}")
    tensors = OrderedDict(
        (str(key), item) for key, item in value.items() if torch.is_tensor(item)
    )
    if len(tensors) != V1_STATE_DICT_TENSOR_COUNT:
        raise ValueError(
            "v1 weight tensor count mismatch: "
            f"expected={V1_STATE_DICT_TENSOR_COUNT}, actual={len(tensors)}"
        )
    wrong_dtype = sorted(
        key for key, tensor in tensors.items() if tensor.dtype != torch.float32
    )
    if wrong_dtype:
        raise TypeError(f"all v1 weights must use FP32: {wrong_dtype[:5]}")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise ValueError("Head2Field 权重含 NaN 或 Inf")
    value_count = sum(int(tensor.numel()) for tensor in tensors.values())
    if value_count != V1_STATE_DICT_VALUE_COUNT:
        raise ValueError(
            "v1 weight value count mismatch: "
            f"expected={V1_STATE_DICT_VALUE_COUNT}, actual={value_count}"
        )

    expected = create_head2field_v1().state_dict()
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    shape_mismatches = {
        key: (tuple(tensors[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(tensors)
        if tensors[key].shape != expected[key].shape
    }
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "v1 weight schema mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={dict(list(shape_mismatches.items())[:5])}"
        )
    return tensors


def read_pure_v1_state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    """Reload with safe weights-only mode and validate the official format."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"v1 weight file does not exist: {resolved}")
    value = torch.load(
        resolved,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return validate_pure_v1_state_dict(value)


def load_head2field_v1_weights(
    path: Path | None = None,
    *,
    device: torch.device | str = "cpu",
) -> Head2FieldV1:
    """Load bundled pretrained v1 weights by default, or a custom bare state dictionary."""

    with materialized_head2field_v1_weights(path) as resolved:
        state_dict = read_pure_v1_state_dict(resolved)
        model = create_head2field_v1()
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(
                "strict loading must not leave missing or unexpected parameter keys"
            )
        del state_dict
        model.eval()
        return model.to(torch.device(device))


__all__ = [
    "V1_STATE_DICT_TENSOR_COUNT",
    "V1_STATE_DICT_VALUE_COUNT",
    "V1_WEIGHTS_FILENAME",
    "load_head2field_v1_weights",
    "materialized_head2field_v1_weights",
    "read_pure_v1_state_dict",
    "validate_pure_v1_state_dict",
]
