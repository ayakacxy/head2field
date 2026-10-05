"""Bundled pretrained weights, custom overrides, and strict bare-weight contracts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from head2field import create_head2field_v1, load_head2field_v1_weights
from head2field.resource_paths import materialized_resource, read_packaged_text
from head2field.weights import (
    V1_STATE_DICT_TENSOR_COUNT,
    V1_STATE_DICT_VALUE_COUNT,
    V1_WEIGHTS_FILENAME,
    validate_pure_v1_state_dict,
)


def test_frozen_factory_matches_pure_state_dict_contract() -> None:
    state = create_head2field_v1().state_dict()
    validated = validate_pure_v1_state_dict(state)
    assert len(validated) == V1_STATE_DICT_TENSOR_COUNT == 208
    assert V1_STATE_DICT_VALUE_COUNT == 2_607_870
    assert sum(tensor.numel() for tensor in validated.values()) == (
        V1_STATE_DICT_VALUE_COUNT
    )
    with pytest.raises(TypeError, match="non-Tensor"):
        validate_pure_v1_state_dict({"model": state})


def test_bundled_pretrained_weights_match_the_loaded_tensors() -> None:
    """随包权重按论文结构读取，并与加载后的张量逐项一致。"""
    with materialized_resource("weights", V1_WEIGHTS_FILENAME) as path:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validated = validate_pure_v1_state_dict(state)
    model = load_head2field_v1_weights()
    assert not model.training
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, validated[name])


def test_custom_weight_override_does_not_use_bundled_parameters(tmp_path) -> None:
    state = create_head2field_v1().state_dict()
    name = next(iter(state))
    state[name].fill_(0.125)
    custom = tmp_path / "custom_v1.pth"
    torch.save(state, custom)
    loaded = load_head2field_v1_weights(custom)
    assert torch.equal(loaded.state_dict()[name], state[name])
