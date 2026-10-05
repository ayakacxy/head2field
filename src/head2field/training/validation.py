"""Deterministic complete validation without consuming training RNGs."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from head2field.model import Head2FieldV1

from .contracts import DataLossConfig, PhysicsLossConfig
from .dataset import Head2FieldSelection, Head2FieldTrainingBatch
from .losses import lead_field_data_loss
from .runner import move_to_device
from .steps import compute_gp_physics_loss


def build_deterministic_validation_callback(
    *,
    dataset: object,
    subject_sizes: tuple[tuple[int, int], ...],
    data_config: DataLossConfig,
    physics_config: PhysicsLossConfig,
    query_count: int,
    sources_per_subject: int,
) -> Callable[[torch.nn.Module, int], dict[str, object]]:
    """Traverse every validation combination, average by subject, and fix a(t)=1."""

    data_config.validate()
    physics_config.validate()
    batch_method = getattr(dataset, "batch", None)
    if not callable(batch_method):
        raise TypeError("validation dataset 必须实现 batch(selection)")
    if not subject_sizes or query_count <= 0 or sources_per_subject <= 0:
        raise ValueError("validation subject/chunk 参数无效")

    def validate(raw_model: torch.nn.Module, global_step: int) -> dict[str, object]:
        if not isinstance(raw_model, Head2FieldV1):
            raise TypeError("TES validation 必须使用 Head2FieldV1")
        was_training = raw_model.training
        raw_model.eval()
        data_losses: list[float] = []
        physics_losses: list[float] = []
        relative_l2: list[float] = []
        try:
            with torch.inference_mode():
                device = next(raw_model.parameters()).device
                for subject_index, (queries, sources) in enumerate(subject_sizes):
                    subject_data = []
                    subject_physics = []
                    subject_relative = []
                    for source_start in range(0, sources, sources_per_subject):
                        for query_start in range(0, queries, query_count):
                            selection = Head2FieldSelection(
                                subject_index=subject_index,
                                source_indices=tuple(
                                    range(
                                        source_start,
                                        min(
                                            source_start + sources_per_subject, sources
                                        ),
                                    )
                                ),
                                query_indices=tuple(
                                    range(
                                        query_start,
                                        min(query_start + query_count, queries),
                                    )
                                ),
                                query_chunk_count=math.ceil(queries / query_count),
                            )
                            batch = move_to_device(batch_method(selection), device)
                            if not isinstance(batch, Head2FieldTrainingBatch):
                                raise TypeError("validation batch 类型不匹配")
                            data_output = raw_model.forward_bundled(
                                batch.head_volume,
                                batch.geometry,
                                batch.data_queries,
                                batch.electrodes,
                                batch.pair_index,
                                deterministic=True,
                                validate_coordinates=True,
                            )
                            physics_output = raw_model.forward_bundled(
                                batch.head_volume,
                                batch.geometry,
                                batch.physics.queries,
                                batch.electrodes,
                                batch.pair_index,
                                deterministic=True,
                                require_unique_queries=True,
                                validate_coordinates=True,
                            )
                            data_loss, data_metrics = lead_field_data_loss(
                                data_output.pair_field.float(),
                                batch.target,
                                batch.data_loss_batch,
                                denominator_floor_ratio=data_config.denominator_floor_ratio,
                                pair_weight=data_config.pair_weight,
                                radial_weight=data_config.radial_weight,
                                peak_weight=data_config.peak_weight,
                                huber_delta=data_config.huber_delta,
                            )
                            physics_loss, _ = compute_gp_physics_loss(
                                physics_output.pair_field.float(),
                                batch.physics.arrays,
                                physics_config,
                            )
                            subject_data.append(float(data_loss.cpu()))
                            subject_physics.append(float(physics_loss.cpu()))
                            subject_relative.append(
                                float(data_metrics["relative_l2"].cpu())
                            )
                    data_losses.append(float(np.mean(subject_data)))
                    physics_losses.append(float(np.mean(subject_physics)))
                    relative_l2.append(float(np.mean(subject_relative)))
        finally:
            raw_model.train(was_training)
        return {
            "validation_step": global_step,
            "subject_count": len(subject_sizes),
            "combined_loss_mean": float(np.mean(data_losses) + np.mean(physics_losses)),
            "physics_curriculum_scale": 1.0,
            "data_loss_mean": float(np.mean(data_losses)),
            "physics_loss_mean": float(np.mean(physics_losses)),
            "relative_l2_mean": float(np.mean(relative_l2)),
            "selection_policy": "complete_source_query_chunks_v1",
        }

    return validate


__all__ = ["build_deterministic_validation_callback"]
