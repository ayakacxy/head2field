"""Two-subject CPU FP32 installation and resume fixtures without research assets."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from head2field.fem_labels.contracts import STANDARD_76_ELECTRODE_NAMES
from head2field.inputs import (
    PreparedModelInputSubject,
    build_electrode_pair_inputs,
    build_query_bundle,
    build_subject_geometry,
)
from head2field.model import Head2FieldV1

from .contracts import DATA_LOSS_RADIAL_EDGES_MM
from .dataset import (
    Head2FieldSelection,
    Head2FieldTrainingBatch,
    PhysicsTrainingBatch,
)
from .physics import (
    circulation_raw_and_scale,
    flux_raw_and_scale,
    interface_raw_and_scale,
)

SYNTHETIC_DATASET_SCHEMA_ID = "synthetic_training_dataset_v1"
SYNTHETIC_SOURCE_NAMES = STANDARD_76_ELECTRODE_NAMES[1:]


SMALL_MODEL_CONFIG: dict[str, object] = {
    "base_channels": 4,
    "encoder_levels": 3,
    "pyramid_dim": 8,
    "query_dim": 32,
    "query_blocks": 1,
    "query_expansion": 2,
    "electrode_dim": 24,
    "decoder_blocks": 1,
    "decoder_bottleneck_dim": 16,
    "global_dim": 16,
    "coordinate_fourier_bands": 2,
    "context_dropout": 0.05,
    "absolute_fourier_bands": 2,
    "absolute_aux_dim": 8,
    "absolute_coordinate_dropout": 0.05,
    "absolute_branch_dropout": 0.10,
    "absolute_coordinate_jitter_probability": 0.50,
    "absolute_coordinate_jitter_std_mm": 1.0,
    "global_mixer_dim": 16,
    "global_mixer_blocks": 1,
    "global_mixer_heads": 4,
    "global_mixer_local_window": 3,
    "global_mixer_max_relative_distance": 8,
    "global_mixer_swiglu_expansion": 2,
    "variant_id": "GP",
}


@dataclass(frozen=True)
class SyntheticSubject:
    """Model inputs and 75-source truth for a compact synthetic subject."""

    subject_key: str
    partition: str
    model_input: PreparedModelInputSubject
    targets: np.ndarray


def create_synthetic_smoke_model(seed: int = 20260901) -> Head2FieldV1:
    """Build a test-only small network sharing the official forward and coordinate contracts."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Head2FieldV1(**SMALL_MODEL_CONFIG)
    return model


def _subject_arrays(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (8, 8, 8)
    tissues = np.ones(shape, dtype=np.uint8)
    tissues[4:, :, :] = 2
    head = np.zeros((12, *shape), dtype=np.float32)
    head[0, :4] = 1.0
    head[1, 4:] = 1.0
    head[10] = np.where(tissues == 1, 0.126 / 1.654, 0.275 / 1.654)
    head[11] = rng.normal(0.0, 0.05, size=shape).astype(np.float32)
    gmwm = np.asarray(
        [
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [2, 2, 1],
            [3, 3, 3],
            [4, 3, 3],
            [3, 4, 3],
            [3, 3, 4],
        ],
        dtype=np.int64,
    )
    grid = np.asarray(
        [(x, y, z) for x in range(1, 6) for y in range(1, 6) for z in range(1, 4)],
        dtype=np.float32,
    )
    if grid.shape != (75, 3):
        raise AssertionError("合成 source grid 必须恰含 75 个点")
    reference = np.asarray([6.0, 6.0, 6.0], dtype=np.float32)
    query = gmwm.astype(np.float32)
    source_delta = query[:, None, :] - grid[None, :, :]
    reference_delta = query[:, None, :] - reference[None, None, :]
    targets = (
        0.015 * source_delta - 0.010 * reference_delta + 0.0005 * float(seed % 13)
    ).astype(np.float32)
    return {
        "head_volume": head,
        "tissues": tissues,
        "t1_normalized": head[11].astype(np.float16),
        "gmwm_ijk": gmwm,
        "source_coords_mm": grid,
        "reference_coord_mm": reference,
        "targets": targets,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_synthetic_training_dataset(output_dir: Path, *, seed: int = 42) -> Path:
    """Transactionally create a compact smoke dataset with two fictional subjects."""

    destination = Path(output_dir).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"合成数据输出目录必须尚不存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(exist_ok=False)
    try:
        records: list[dict[str, object]] = []
        definitions = (
            ("synthetic-train-a", "train", seed + 11),
            ("synthetic-validation-b", "validation", seed + 29),
        )
        for subject_key, partition, subject_seed in definitions:
            filename = f"{subject_key}.npz"
            path = staging / filename
            np.savez(path, **_subject_arrays(subject_seed))
            records.append(
                {
                    "subject_key": subject_key,
                    "group_key": f"{partition}-group",
                    "partition": partition,
                    "relative_path": filename,
                    "size_bytes": path.stat().st_size,
                }
            )
        payload = {
            "schema_id": SYNTHETIC_DATASET_SCHEMA_ID,
            "purpose": "cpu_fp32_installation_and_resume_smoke_only",
            "official_weight_reproduction": False,
            "seed": seed,
            "reference_electrode": "Oz",
            "source_names": list(SYNTHETIC_SOURCE_NAMES),
            "groups": [
                {"group_key": "train-group", "partition": "train"},
                {"group_key": "validation-group", "partition": "validation"},
            ],
            "subjects": records,
        }
        _write_json(staging / "synthetic_training_dataset_v1.json", payload)
        os.rename(staging, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    return destination / "synthetic_training_dataset_v1.json"


def load_synthetic_training_dataset(path: Path) -> tuple[SyntheticSubject, ...]:
    """Reload and validate the two-subject smoke dataset's file hashes and split."""

    manifest_path = Path(path).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed = {
        "schema_id": SYNTHETIC_DATASET_SCHEMA_ID,
        "purpose": "cpu_fp32_installation_and_resume_smoke_only",
        "official_weight_reproduction": False,
        "reference_electrode": "Oz",
        "source_names": list(SYNTHETIC_SOURCE_NAMES),
        "groups": [
            {"group_key": "train-group", "partition": "train"},
            {"group_key": "validation-group", "partition": "validation"},
        ],
    }
    mismatched = {
        key: (payload.get(key), value)
        for key, value in fixed.items()
        if payload.get(key) != value
    }
    if mismatched:
        raise ValueError(f"合成数据 manifest 合同不匹配: {mismatched}")
    records = payload.get("subjects")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("合成数据必须恰含 train/validation 两个 subject")
    subjects: list[SyntheticSubject] = []
    seen_partitions: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("合成 subject 记录必须是对象")
        relative = str(record["relative_path"])
        if "/" in relative or "\\" in relative or relative.startswith("."):
            raise ValueError("合成 subject 文件必须与 manifest 同目录")
        asset = manifest_path.parent / relative
        if asset.is_symlink() or not asset.is_file():
            raise FileNotFoundError(f"合成 subject 文件不存在: {asset}")
        if asset.stat().st_size != int(record["size_bytes"]):
            raise ValueError("合成 subject 字节数不匹配")
        with np.load(asset, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        required = {
            "head_volume",
            "tissues",
            "t1_normalized",
            "gmwm_ijk",
            "source_coords_mm",
            "reference_coord_mm",
            "targets",
        }
        if set(arrays) != required:
            raise ValueError("合成 subject NPZ 字段不匹配")
        if arrays["head_volume"].shape != (12, 8, 8, 8):
            raise ValueError("合成 head_volume shape 不匹配")
        if arrays["gmwm_ijk"].shape != (8, 3):
            raise ValueError("合成 gmwm_ijk shape 不匹配")
        if arrays["source_coords_mm"].shape != (75, 3):
            raise ValueError("合成 source 坐标 shape 不匹配")
        if arrays["targets"].shape != (8, 75, 3):
            raise ValueError("合成 target 必须为 [query,source,component]")
        if any(
            value.dtype.kind in "fc" and not np.isfinite(value).all()
            for value in arrays.values()
        ):
            raise ValueError("合成 subject 含 NaN/Inf")
        partition = str(record["partition"])
        seen_partitions.add(partition)
        affine = np.eye(4, dtype=np.float32)
        model_input = PreparedModelInputSubject(
            subject_id=str(record["subject_key"]),
            head_volume=torch.from_numpy(arrays["head_volume"].astype(np.float32)),
            tissue_labels_1mm=arrays["tissues"].astype(np.uint8),
            t1_normalized_1mm=arrays["t1_normalized"].astype(np.float16),
            affine=affine,
            inverse_affine=affine.copy(),
            full_shape=(8, 8, 8),
            gmwm_ijk=arrays["gmwm_ijk"].astype(np.int64),
            source_names=tuple(SYNTHETIC_SOURCE_NAMES),
            source_coords_mm=arrays["source_coords_mm"].astype(np.float32),
            reference_coord_mm=arrays["reference_coord_mm"].astype(np.float32),
        )
        subjects.append(
            SyntheticSubject(
                subject_key=str(record["subject_key"]),
                partition=partition,
                model_input=model_input,
                targets=arrays["targets"].astype(np.float32),
            )
        )
    if seen_partitions != {"train", "validation"}:
        raise ValueError("合成数据 train/validation split 不完整")
    return tuple(subjects)


def _loss_statistics(
    target: np.ndarray,
    query_mm: np.ndarray,
    source_mm: np.ndarray,
    reference_mm: np.ndarray,
) -> dict[str, object]:
    squared = np.square(target, dtype=np.float32).sum(axis=-1, dtype=np.float64)
    global_energy = squared.sum(axis=0)
    query = query_mm[:, None, :]
    remote = np.minimum(
        np.linalg.norm(query - source_mm[None], axis=-1),
        np.linalg.norm(query - reference_mm[None, None], axis=-1),
    )
    lower, upper = DATA_LOSS_RADIAL_EDGES_MM
    masks = (remote < lower, (remote >= lower) & (remote < upper), remote >= upper)
    radial = np.stack([(squared * mask).sum(axis=0) for mask in masks], axis=-1)
    thresholds = np.quantile(squared, 0.90, axis=0)
    peak = np.stack(
        [
            squared[:, index][squared[:, index] >= thresholds[index]].sum()
            for index in range(squared.shape[1])
        ]
    )
    distance = np.linalg.norm(source_mm - reference_mm[None], axis=-1)
    pair_weights = 1.0 + np.exp(-0.5 * np.square(distance / 40.0))
    pair_weights /= pair_weights.mean()
    positive_global = global_energy[global_energy > 0]
    radial_median = np.asarray(
        [
            np.median(radial[:, index][radial[:, index] > 0])
            if np.any(radial[:, index] > 0)
            else 0.0
            for index in range(3)
        ],
        dtype=np.float64,
    )
    positive_peak = peak[peak > 0]
    return {
        "pair_weights": pair_weights.astype(np.float32),
        "global_energy": global_energy,
        "radial_energy": radial,
        "peak_threshold_squared": thresholds,
        "peak_energy": peak,
        "global_median_energy": float(np.median(positive_global)),
        "radial_median_energy": radial_median,
        "peak_median_energy": float(np.median(positive_peak)),
    }


def _physics_arrays(target_field: torch.Tensor) -> dict[str, torch.Tensor]:
    point_index = torch.tensor([[0, 1, 3, 2]], dtype=torch.long)
    edge_vector = torch.tensor(
        [
            [
                [0.001, 0.0, 0.0],
                [0.0, 0.001, 0.0],
                [-0.001, 0.0, 0.0],
                [0.0, -0.001, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    point_mask = torch.ones(1, 4, dtype=torch.bool)
    wm_index = torch.tensor([0], dtype=torch.long)
    gm_index = torch.tensor([1], dtype=torch.long)
    normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    face_index = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    area = 1e-6 * torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    )
    conductivity = torch.tensor([0.126], dtype=torch.float32)
    circ_raw, circ_scale = circulation_raw_and_scale(
        target_field, point_index, edge_vector, point_mask
    )
    normal_raw, normal_scale, tangent_raw, tangent_scale = interface_raw_and_scale(
        target_field, wm_index, gm_index, normal
    )
    flux_raw, flux_scale = flux_raw_and_scale(
        target_field, face_index, area, conductivity
    )
    arrays: dict[str, torch.Tensor] = {
        "circ_point_index": point_index,
        "circ_edge_vector_m": edge_vector,
        "circ_point_mask": point_mask,
        "interface_wm_point_index": wm_index,
        "interface_gm_point_index": gm_index,
        "interface_normal_wm_to_gm": normal,
        "flux_face_point_index": face_index,
        "flux_outward_area_vector_m2": area,
        "flux_conductivity_s_per_m": conductivity,
    }
    raw_values = {
        "circ": (circ_raw, circ_scale),
        "interface_normal": (normal_raw, normal_scale),
        "interface_tangent": (tangent_raw, tangent_scale),
        "flux": (flux_raw, flux_scale),
    }
    for prefix, (raw, scale) in raw_values.items():
        floor = torch.full_like(scale, torch.finfo(torch.float32).eps)
        arrays[f"{prefix}_fem_raw_residual"] = raw.detach()
        arrays[f"{prefix}_fem_raw_scale"] = scale.detach()
        arrays[f"{prefix}_stratum_floor"] = floor.detach()
        arrays[f"{prefix}_stratum_mad"] = torch.zeros_like(scale)
        arrays[f"{prefix}_valid_mask"] = torch.ones_like(scale, dtype=torch.bool)
    return arrays


class SyntheticHead2FieldDataset:
    """Assemble compact fixtures into real TES model data-plus-physics batches."""

    def __init__(
        self, subjects: tuple[SyntheticSubject, ...], *, partition: str
    ) -> None:
        self.subjects = tuple(
            subject for subject in subjects if subject.partition == partition
        )
        if not self.subjects:
            raise ValueError(f"合成数据没有 partition={partition!r}")
        self.schedule_loss_scale = 1.0

    @property
    def subject_sizes(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (len(subject.model_input.gmwm_ijk), 75) for subject in self.subjects
        )

    def batch(self, raw_selection: object) -> Head2FieldTrainingBatch:
        if not isinstance(raw_selection, Head2FieldSelection):
            raise TypeError("合成数据 batch 需要 Head2FieldSelection")
        subject = self.subjects[raw_selection.subject_index]
        source_indices = np.asarray(raw_selection.source_indices, dtype=np.int64)
        query_indices = np.asarray(raw_selection.query_indices, dtype=np.int64)
        geometry = build_subject_geometry(subject.model_input, device="cpu")
        queries = build_query_bundle(
            subject.model_input, query_indices, device="cpu", geometry=geometry
        )
        pairs = build_electrode_pair_inputs(
            subject.model_input,
            subject.model_input.source_coords_mm[source_indices],
            source_electrode_ids=source_indices,
            reference_electrode_id=75,
            device="cpu",
            geometry=geometry,
        )
        target = subject.targets[:, source_indices, :].transpose(1, 0, 2)
        target = target[:, query_indices, :]
        query_mm_all = subject.model_input.gmwm_ijk.astype(np.float32)
        statistics = _loss_statistics(
            subject.targets,
            query_mm_all,
            subject.model_input.source_coords_mm,
            subject.model_input.reference_coord_mm,
        )
        data_loss_batch = {
            "query_mm": queries.world_ras_mm,
            "source_mm": torch.from_numpy(
                subject.model_input.source_coords_mm[source_indices]
            ).unsqueeze(0),
            "reference_mm": torch.from_numpy(
                subject.model_input.reference_coord_mm
            ).unsqueeze(0),
            "pair_weights": torch.from_numpy(
                statistics["pair_weights"][source_indices]
            ).unsqueeze(0),
            "global_energy": torch.from_numpy(
                statistics["global_energy"][source_indices]
            ).unsqueeze(0),
            "radial_energy": torch.from_numpy(
                statistics["radial_energy"][source_indices]
            ).unsqueeze(0),
            "peak_threshold_squared": torch.from_numpy(
                statistics["peak_threshold_squared"][source_indices]
            ).unsqueeze(0),
            "peak_energy": torch.from_numpy(
                statistics["peak_energy"][source_indices]
            ).unsqueeze(0),
            "global_median_energy": statistics["global_median_energy"],
            "radial_median_energy": torch.from_numpy(
                statistics["radial_median_energy"]
            ),
            "peak_median_energy": statistics["peak_median_energy"],
            "radial_edges_mm": torch.tensor(DATA_LOSS_RADIAL_EDGES_MM),
            "schedule_loss_scale": self.schedule_loss_scale,
            "voxel_batch_count": raw_selection.query_chunk_count,
        }
        physics_queries = build_query_bundle(
            subject.model_input,
            np.arange(8, dtype=np.int64),
            device="cpu",
            geometry=geometry,
        )
        full_target = torch.from_numpy(
            subject.targets[:, source_indices, :].transpose(1, 0, 2)
        ).unsqueeze(0)
        return Head2FieldTrainingBatch(
            subject_key=subject.subject_key,
            head_volume=subject.model_input.head_volume.unsqueeze(0).float(),
            geometry=geometry,
            data_queries=queries,
            electrodes=pairs.electrodes,
            pair_index=pairs.pair_index,
            target=torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0),
            data_loss_batch=data_loss_batch,
            physics=PhysicsTrainingBatch(
                queries=physics_queries,
                arrays=_physics_arrays(full_target),
            ),
        )


__all__ = [
    "SMALL_MODEL_CONFIG",
    "SYNTHETIC_DATASET_SCHEMA_ID",
    "SYNTHETIC_SOURCE_NAMES",
    "SyntheticHead2FieldDataset",
    "SyntheticSubject",
    "create_synthetic_smoke_model",
    "load_synthetic_training_dataset",
    "write_synthetic_training_dataset",
]
