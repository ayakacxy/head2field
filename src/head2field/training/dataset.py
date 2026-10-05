"""Training data strictly binding prepared subjects, FEM labels, statistics, and physics pools."""

from __future__ import annotations

from head2field.metadata import is_digest_key

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from head2field.fem_labels import FEMLabelsV1, load_fem_labels_v1
from head2field.inputs import (
    PreparedModelInputSubject,
    build_electrode_pair_inputs,
    build_prepared_model_input_subject,
    build_query_bundle,
    build_subject_geometry,
)
from head2field.model import (
    ElectrodeCoordinateBundle,
    QueryCoordinateBundle,
    SubjectGeometry,
)
from head2field.preprocessing.validation import (
    PreparedSubjectV1,
    load_prepared_subject_v1,
)

from .contracts import (
    DATA_LOSS_DENOMINATOR_RULE_ID,
    DATA_LOSS_RADIAL_EDGES_MM,
    PHYSICS_DENOMINATOR_RULE_ID,
    PHYSICS_OPERATOR_VERSION,
    SOURCE_COUNT,
    SubjectRecord,
    TrainingContractError,
    TrainingDatasetManifest,
    canonical_json_sha256,
)

LOSS_STATISTICS_SCHEMA_ID = "training_loss_statistics_v2"
PHYSICS_POOL_SCHEMA_ID = "training_physics_pool_v1"


@dataclass(frozen=True)
class LossStatistics:
    """Frozen stable-energy loss statistics for one subject."""

    manifest_sha256: str
    pair_weights: np.ndarray
    global_energy: np.ndarray
    radial_energy: np.ndarray
    peak_threshold_squared: np.ndarray
    peak_energy: np.ndarray
    global_median_energy: float
    radial_median_energy: np.ndarray
    peak_median_energy: float
    radial_edges_mm: np.ndarray


@dataclass(frozen=True)
class PhysicsPool:
    """Frozen GP physics pool mapped to union-query indices."""

    manifest_sha256: str
    union_query_rows: np.ndarray
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class TrainingSubject:
    """Read-only cross-validated handles to the four required training assets."""

    subject_key: str
    prepared: PreparedSubjectV1
    model_input: PreparedModelInputSubject
    fem: FEMLabelsV1
    statistics: LossStatistics
    physics_pool: PhysicsPool


@dataclass(frozen=True)
class Head2FieldSelection:
    """Subject, source, and query selection for one training step."""

    subject_index: int
    source_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    query_chunk_count: int = 1


@dataclass(frozen=True)
class PhysicsTrainingBatch:
    """Union queries, operators, and frozen FEM statistics for the GP physics branch."""

    queries: QueryCoordinateBundle
    arrays: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class Head2FieldTrainingBatch:
    """Complete batch for one model forward and its data/physics losses."""

    subject_key: str
    head_volume: torch.Tensor
    geometry: SubjectGeometry
    data_queries: QueryCoordinateBundle
    electrodes: ElectrodeCoordinateBundle
    pair_index: torch.Tensor
    target: torch.Tensor
    data_loss_batch: Mapping[str, object]
    physics: PhysicsTrainingBatch


def _asset_path(
    manifest: TrainingDatasetManifest,
    relative: PurePosixPath,
) -> Path:
    return manifest.path.parent.joinpath(*relative.parts).resolve(strict=True)


def _json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TrainingContractError(f"{role} 不是有效 JSON: {path}") from error
    if not isinstance(value, dict):
        raise TrainingContractError(f"{role} 顶层必须是 JSON 对象")
    return value


def _relative_file(root: Path, raw: object, role: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise TrainingContractError(f"{role} 必须是安全相对 POSIX 路径")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise TrainingContractError(f"{role} 不能逃逸资产目录")
    base = root.resolve(strict=True)
    resolved = base.joinpath(*relative.parts).resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise TrainingContractError(f"{role} 解析后逃逸资产目录") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise TrainingContractError(f"{role} 必须是普通非软链接文件")
    return resolved


def _positive_median(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        positive = values[values > 0.0]
        return np.asarray(float(np.median(positive)) if positive.size else 0.0)
    result: list[float] = []
    for index in range(values.shape[axis]):
        selected = np.take(values, index, axis=axis)
        positive = selected[selected > 0.0]
        result.append(float(np.median(positive)) if positive.size else 0.0)
    return np.asarray(result, dtype=np.float64)


def load_loss_statistics(
    manifest_path: Path,
    *,
    prepared: PreparedSubjectV1,
    fem: FEMLabelsV1,
    expected_sources: Sequence[str],
) -> LossStatistics:
    """Load the statistics NPZ and bind it to prepared/FEM row order and hashes."""

    path = Path(manifest_path)
    payload = _json_object(path, "loss statistics manifest")
    expected_fixed = {
        "schema_id": LOSS_STATISTICS_SCHEMA_ID,
        "source_names": list(expected_sources),
        "query_count": len(prepared.gmwm_ijk),
        "denominator_rule_id": DATA_LOSS_DENOMINATOR_RULE_ID,
    }
    mismatched = {
        key: (payload.get(key), expected)
        for key, expected in expected_fixed.items()
        if not is_digest_key(key) and payload.get(key) != expected
    }
    if mismatched:
        raise TrainingContractError(
            "Loss statistics do not match the paper contract; regenerate them "
            f"with head2field-prepare-statistics: {mismatched}"
        )
    config = payload.get("configuration")
    expected_config = {
        "radial_edges_mm": list(DATA_LOSS_RADIAL_EDGES_MM),
        "pair_weight_alpha": 1.0,
        "pair_weight_sigma_mm": 40.0,
        "peak_quantile": 0.90,
    }
    if config != expected_config:
        raise TrainingContractError(
            "Loss statistics require 50/100 mm radial edges; regenerate them "
            "with head2field-prepare-statistics."
        )
    record = payload.get("arrays")
    if not isinstance(record, Mapping):
        raise TrainingContractError("loss statistics manifest 缺少 arrays 记录")
    arrays_path = _relative_file(
        path.parent, record.get("relative_path"), "statistics arrays"
    )
    if arrays_path.stat().st_size != record.get("size_bytes"):
        raise TrainingContractError("loss statistics arrays 字节数不匹配")
    expected_keys = {
        "source_names",
        "pair_weights",
        "global_energy",
        "radial_energy",
        "peak_threshold_squared",
        "peak_energy",
        "global_median_energy",
        "radial_median_energy",
        "peak_median_energy",
        "radial_edges_mm",
        "query_count",
        "query_ijk",
        "affine_ijk_to_ras_mm",
        "source_coords_mm",
        "reference_coord_mm",
    }
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise TrainingContractError("loss statistics NPZ 字段不匹配")
        if archive["source_names"].tolist() != list(expected_sources):
            raise TrainingContractError("loss statistics source 顺序不匹配")
        if int(archive["query_count"]) != len(prepared.gmwm_ijk):
            raise TrainingContractError("loss statistics query_count 不匹配")
        # 直接核对统计所用的空间和电极坐标，拒绝错位的旧统计。
        for key, expected in (
            ("query_ijk", prepared.gmwm_ijk),
            ("affine_ijk_to_ras_mm", prepared.affine),
            ("source_coords_mm", prepared.source_coords_mm),
            ("reference_coord_mm", prepared.reference_coord_mm),
        ):
            if not np.array_equal(archive[key], expected):
                raise TrainingContractError(f"loss statistics {key} 不匹配，请重建统计")
        pair_weights = np.asarray(archive["pair_weights"], dtype=np.float32)
        global_energy = np.asarray(archive["global_energy"], dtype=np.float64)
        radial_energy = np.asarray(archive["radial_energy"], dtype=np.float64)
        thresholds = np.asarray(archive["peak_threshold_squared"], dtype=np.float64)
        peak_energy = np.asarray(archive["peak_energy"], dtype=np.float64)
        global_median = float(archive["global_median_energy"])
        radial_median = np.asarray(archive["radial_median_energy"], dtype=np.float64)
        peak_median = float(archive["peak_median_energy"])
        radial_edges = np.asarray(archive["radial_edges_mm"], dtype=np.float32)
    shapes = {
        "pair_weights": (pair_weights.shape, (SOURCE_COUNT,)),
        "global_energy": (global_energy.shape, (SOURCE_COUNT,)),
        "radial_energy": (radial_energy.shape, (SOURCE_COUNT, 3)),
        "peak_threshold_squared": (thresholds.shape, (SOURCE_COUNT,)),
        "peak_energy": (peak_energy.shape, (SOURCE_COUNT,)),
        "radial_median_energy": (radial_median.shape, (3,)),
        "radial_edges_mm": (radial_edges.shape, (2,)),
    }
    wrong_shapes = {
        name: (actual, expected)
        for name, (actual, expected) in shapes.items()
        if actual != expected
    }
    values = [
        pair_weights,
        global_energy,
        radial_energy,
        thresholds,
        peak_energy,
        radial_median,
        radial_edges,
        np.asarray([global_median, peak_median]),
    ]
    if wrong_shapes or any(not np.isfinite(value).all() for value in values):
        raise TrainingContractError(
            f"loss statistics shape/finite 合同不匹配: {wrong_shapes}"
        )
    if not np.array_equal(
        radial_edges, np.asarray(DATA_LOSS_RADIAL_EDGES_MM, dtype=np.float32)
    ):
        raise TrainingContractError("loss statistics radial edges 被修改")
    if global_median != float(_positive_median(global_energy)):
        raise TrainingContractError("global_median_energy 不是正能量中位数")
    if not np.array_equal(radial_median, _positive_median(radial_energy, axis=1)):
        raise TrainingContractError("radial_median_energy 不是正能量中位数")
    if peak_median != float(_positive_median(peak_energy)):
        raise TrainingContractError("peak_median_energy 不是正能量中位数")
    return LossStatistics(
        manifest_sha256=canonical_json_sha256(payload),
        pair_weights=pair_weights,
        global_energy=global_energy,
        radial_energy=radial_energy,
        peak_threshold_squared=thresholds,
        peak_energy=peak_energy,
        global_median_energy=global_median,
        radial_median_energy=radial_median,
        peak_median_energy=peak_median,
        radial_edges_mm=radial_edges,
    )


def load_physics_pool(
    manifest_path: Path,
    *,
    prepared: PreparedSubjectV1,
    fem: FEMLabelsV1,
    expected_sources: Sequence[str],
) -> PhysicsPool:
    """Load the complete all-profile physics pool and training-fitted frozen statistics."""

    path = Path(manifest_path)
    payload = _json_object(path, "physics pool manifest")
    if payload.get("statistics_fit_partition") != "train":
        raise TrainingContractError("physics 统计必须仅由训练划分拟合")
    fixed = {
        "schema_id": PHYSICS_POOL_SCHEMA_ID,
        "operator_version": PHYSICS_OPERATOR_VERSION,
        "denominator_rule_id": PHYSICS_DENOMINATOR_RULE_ID,
        "source_names": list(expected_sources),
        "query_count": len(prepared.gmwm_ijk),
        "primitive_counts": {"circulation": 512, "interface": 512, "flux": 512},
    }
    mismatched = {
        key: (payload.get(key), expected)
        for key, expected in fixed.items()
        if not is_digest_key(key) and payload.get(key) != expected
    }
    if mismatched:
        raise TrainingContractError(f"physics pool 固定合同不匹配: {mismatched}")
    record = payload.get("arrays")
    if not isinstance(record, Mapping):
        raise TrainingContractError("physics pool manifest 缺少 arrays 记录")
    arrays_path = _relative_file(
        path.parent, record.get("relative_path"), "physics arrays"
    )
    if arrays_path.stat().st_size != record.get("size_bytes"):
        raise TrainingContractError("physics pool arrays 字节数不匹配")
    required = {
        "union_query_rows",
        "union_query_ijk",
        "affine_ijk_to_ras_mm",
        "source_coords_mm",
        "reference_coord_mm",
        "circ_point_index",
        "circ_edge_vector_m",
        "circ_point_mask",
        "interface_wm_point_index",
        "interface_gm_point_index",
        "interface_normal_wm_to_gm",
        "flux_face_point_index",
        "flux_outward_area_vector_m2",
        "flux_conductivity_s_per_m",
    }
    for prefix in ("circ", "interface_normal", "interface_tangent", "flux"):
        required.update(
            {
                f"{prefix}_fem_raw_residual",
                f"{prefix}_fem_raw_scale",
                f"{prefix}_stratum_floor",
                f"{prefix}_stratum_mad",
                f"{prefix}_valid_mask",
            }
        )
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise TrainingContractError(
                "physics pool NPZ 字段不匹配: "
                f"missing={sorted(required - set(archive.files))}, "
                f"unexpected={sorted(set(archive.files) - required)}"
            )
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    rows = arrays["union_query_rows"]
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise TrainingContractError("physics union_query_rows 必须是非空整数向量")
    if (
        len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= len(prepared.gmwm_ijk))
    ):
        raise TrainingContractError(
            "physics union_query_rows 必须唯一且位于 query 范围"
        )
    # 直接比较 primitive 使用的空间及电极信息。
    for key, expected in (
        ("union_query_ijk", prepared.gmwm_ijk[rows]),
        ("affine_ijk_to_ras_mm", prepared.affine),
        ("source_coords_mm", prepared.source_coords_mm),
        ("reference_coord_mm", prepared.reference_coord_mm),
    ):
        if not np.array_equal(arrays[key], expected):
            raise TrainingContractError(
                f"physics {key} 与当前输入不匹配，请重建 primitive"
            )
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise TrainingContractError(f"physics arrays {name} 含 NaN/Inf")
    for prefix in ("circ", "interface_normal", "interface_tangent", "flux"):
        for suffix in ("fem_raw_residual", "fem_raw_scale", "valid_mask"):
            value = arrays[f"{prefix}_{suffix}"]
            if value.shape != (SOURCE_COUNT, 512):
                raise TrainingContractError(f"{prefix}_{suffix} 必须为 [75,512]")
        for suffix in ("stratum_floor", "stratum_mad"):
            value = arrays[f"{prefix}_{suffix}"]
            if value.shape not in ((512,), (SOURCE_COUNT, 512)):
                raise TrainingContractError(
                    f"{prefix}_{suffix} 必须为 [512] 或 [75,512]"
                )
    return PhysicsPool(
        manifest_sha256=canonical_json_sha256(payload),
        union_query_rows=rows.astype(np.int64, copy=False),
        arrays=arrays,
    )


class DeterministicHead2FieldSampler:
    """Cover subject times source batches times query chunks completely in each epoch."""

    schema_id = "deterministic_lead_field_epoch_sampler_v1"

    def __init__(
        self,
        subject_sizes: Sequence[tuple[int, int]],
        *,
        query_count: int,
        sources_per_step: int,
        seed: int,
    ) -> None:
        if not subject_sizes or query_count <= 0 or sources_per_step <= 0 or seed < 0:
            raise ValueError("lead-field sampler 初始化参数无效")
        self.subject_sizes = tuple((int(q), int(s)) for q, s in subject_sizes)
        if any(q <= 0 or s <= 0 for q, s in self.subject_sizes):
            raise ValueError("subject query/source 数量必须为正")
        self.query_count = int(query_count)
        self.sources_per_step = int(sources_per_step)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.subject_order = np.empty(0, dtype=np.int64)
        self.source_order = np.empty(0, dtype=np.int64)
        self.query_order = np.empty(0, dtype=np.int64)
        self.subject_cursor = 0
        self.source_cursor = 0
        self.query_cursor = 0
        self.epoch = -1
        self.selections_emitted = 0

    @property
    def steps_per_epoch(self) -> int:
        """Include final partial electrode and query batches in the epoch."""
        return sum(
            math.ceil(q / self.query_count) * math.ceil(s / self.sources_per_step)
            for q, s in self.subject_sizes
        )

    @property
    def data_loss_scale(self) -> float:
        """Return the paper's fixed per-step scale M / (75 * N_train)."""

        return self.steps_per_epoch / (SOURCE_COUNT * len(self.subject_sizes))

    def next_selection(self) -> Head2FieldSelection:
        if self.subject_cursor >= len(self.subject_order):
            self.subject_order = self.rng.permutation(len(self.subject_sizes)).astype(
                np.int64
            )
            self.subject_cursor = 0
            self.epoch += 1
        subject_index = int(self.subject_order[self.subject_cursor])
        queries, sources = self.subject_sizes[subject_index]
        if not len(self.source_order):
            self.source_order = self.rng.permutation(sources).astype(np.int64)
            self.query_order = self.rng.permutation(queries).astype(np.int64)
        source_end = min(self.source_cursor + self.sources_per_step, sources)
        query_end = min(self.query_cursor + self.query_count, queries)
        selection = Head2FieldSelection(
            subject_index=subject_index,
            source_indices=tuple(
                int(v) for v in self.source_order[self.source_cursor : source_end]
            ),
            query_indices=tuple(
                int(v) for v in self.query_order[self.query_cursor : query_end]
            ),
            query_chunk_count=math.ceil(queries / self.query_count),
        )
        self.query_cursor = query_end
        if self.query_cursor == queries:
            self.query_cursor = 0
            self.source_cursor = source_end
            if self.source_cursor == sources:
                self.source_cursor = 0
                self.subject_cursor += 1
                self.source_order = np.empty(0, dtype=np.int64)
                self.query_order = np.empty(0, dtype=np.int64)
        self.selections_emitted += 1
        return selection

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "subject_sizes": self.subject_sizes,
            "query_count": self.query_count,
            "sources_per_step": self.sources_per_step,
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state,
            "subject_order": self.subject_order.copy(),
            "subject_cursor": self.subject_cursor,
            "source_order": self.source_order.copy(),
            "source_cursor": self.source_cursor,
            "query_order": self.query_order.copy(),
            "query_cursor": self.query_cursor,
            "epoch": self.epoch,
            "selections_emitted": self.selections_emitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            set(state) != set(self.state_dict())
            or state.get("schema_id") != self.schema_id
        ):
            raise TrainingContractError("lead-field sampler state schema 不匹配")
        for name in ("subject_sizes", "query_count", "sources_per_step", "seed"):
            actual = state[name]
            if name == "subject_sizes":
                actual = tuple(tuple(int(v) for v in item) for item in actual)
            if actual != getattr(self, name):
                raise TrainingContractError(f"resume sampler {name} 与当前数据不一致")
        subject_order = np.asarray(state["subject_order"])
        cursor = int(state["subject_cursor"])
        if (
            subject_order.ndim != 1
            or subject_order.dtype.kind not in "iu"
            or len(subject_order) not in (0, len(self.subject_sizes))
        ):
            raise TrainingContractError("resume subject_order 长度或类型错误")
        if len(subject_order) and not np.array_equal(
            np.sort(subject_order), np.arange(len(self.subject_sizes))
        ):
            raise TrainingContractError("resume subject_order 不是完整 permutation")
        if not 0 <= cursor <= len(subject_order):
            raise TrainingContractError("resume subject_cursor 越界")
        source_order = np.asarray(state["source_order"])
        query_order = np.asarray(state["query_order"])
        if bool(len(source_order)) != bool(len(query_order)):
            raise TrainingContractError("resume source/query permutation 不一致")
        sizes = (
            self.subject_sizes[int(subject_order[cursor])]
            if cursor < len(subject_order)
            else (0, 0)
        )
        for order, length, name in (
            (source_order, sizes[1], "source"),
            (query_order, sizes[0], "query"),
        ):
            if order.ndim != 1 or order.dtype.kind not in "iu":
                raise TrainingContractError(f"resume {name}_order 类型错误")
            if len(order) and (
                len(order) != length
                or not np.array_equal(np.sort(order), np.arange(length))
            ):
                raise TrainingContractError(f"resume {name}_order 不是完整 permutation")
            saved_cursor = int(state[f"{name}_cursor"])
            if not 0 <= saved_cursor < max(len(order), 1):
                raise TrainingContractError(f"resume {name}_cursor 越界")
        self.rng.bit_generator.state = state["rng_state"]
        self.subject_order = subject_order.astype(np.int64, copy=True)
        self.source_order = source_order.astype(np.int64, copy=True)
        self.query_order = query_order.astype(np.int64, copy=True)
        self.subject_cursor = cursor
        self.source_cursor = int(state["source_cursor"])
        self.query_cursor = int(state["query_cursor"])
        self.epoch = int(state["epoch"])
        self.selections_emitted = int(state["selections_emitted"])


class Head2FieldTrainingDataset:
    """Lazily load fully hash-validated training subjects from a manifest partition."""

    def __init__(
        self,
        manifest: TrainingDatasetManifest,
        *,
        partition: str,
        encoder_input_shape: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        if partition not in {"train", "validation", "test"}:
            raise ValueError("训练 dataset 只接受 train/validation/test partition")
        records = manifest.subjects_for_partition(partition)
        if not records:
            raise TrainingContractError(f"partition {partition!r} 没有 subject")
        self.manifest = manifest
        self.partition = partition
        self.encoder_input_shape = tuple(int(value) for value in encoder_input_shape)
        self.records = records
        self._cache: dict[int, TrainingSubject] = {}
        self.schedule_loss_scale = 1.0

    def _load_subject(self, index: int) -> TrainingSubject:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        if not 0 <= index < len(self.records):
            raise IndexError("subject index 越界")
        record: SubjectRecord = self.records[index]
        prepared_manifest = _asset_path(self.manifest, record.prepared_subject_manifest)
        prepared = load_prepared_subject_v1(
            prepared_manifest.parent,
        )
        if prepared.manifest_path.resolve() != prepared_manifest:
            raise TrainingContractError("prepared manifest 文件名与 loader 合同不一致")
        fem_manifest = _asset_path(self.manifest, record.fem_labels_manifest)
        fem = load_fem_labels_v1(
            fem_manifest.parent,
            prepared_subject=prepared,
        )
        if fem.manifest_path.resolve() != fem_manifest:
            raise TrainingContractError("FEM manifest 文件名与 loader 合同不一致")
        if tuple(prepared.source_names) != self.manifest.source_names:
            raise TrainingContractError(
                "prepared source 顺序与 dataset manifest 不一致"
            )
        if fem.source_names != self.manifest.source_names:
            raise TrainingContractError("FEM source 顺序与 dataset manifest 不一致")
        statistics = load_loss_statistics(
            _asset_path(self.manifest, record.loss_statistics_manifest),
            prepared=prepared,
            fem=fem,
            expected_sources=self.manifest.source_names,
        )
        physics_pool = load_physics_pool(
            _asset_path(self.manifest, record.physics_pool_manifest),
            prepared=prepared,
            fem=fem,
            expected_sources=self.manifest.source_names,
        )
        model_input = build_prepared_model_input_subject(
            prepared,
            encoder_input_shape=self.encoder_input_shape,
        )
        subject = TrainingSubject(
            subject_key=record.subject_key,
            prepared=prepared,
            model_input=model_input,
            fem=fem,
            statistics=statistics,
            physics_pool=physics_pool,
        )
        self._cache[index] = subject
        return subject

    @property
    def subject_sizes(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (len(self._load_subject(index).prepared.gmwm_ijk), SOURCE_COUNT)
            for index in range(len(self.records))
        )

    def batch(self, selection: object) -> Head2FieldTrainingBatch:
        if not isinstance(selection, Head2FieldSelection):
            raise TypeError("Head2FieldTrainingDataset 需要 Head2FieldSelection")
        subject = self._load_subject(selection.subject_index)
        source_indices = np.asarray(selection.source_indices, dtype=np.int64)
        query_indices = np.asarray(selection.query_indices, dtype=np.int64)
        if len(np.unique(source_indices)) != len(source_indices) or np.any(
            (source_indices < 0) | (source_indices >= SOURCE_COUNT)
        ):
            raise TrainingContractError("batch source indices 重复或越界")
        if len(np.unique(query_indices)) != len(query_indices) or np.any(
            (query_indices < 0) | (query_indices >= len(subject.prepared.gmwm_ijk))
        ):
            raise TrainingContractError("batch query indices 重复或越界")
        device = torch.device("cpu")
        geometry = build_subject_geometry(subject.model_input, device=device)
        data_queries = build_query_bundle(
            subject.model_input,
            query_indices,
            device=device,
            geometry=geometry,
        )
        pair_inputs = build_electrode_pair_inputs(
            subject.model_input,
            subject.model_input.source_coords_mm[source_indices],
            source_electrode_ids=source_indices,
            reference_electrode_id=SOURCE_COUNT,
            device=device,
            geometry=geometry,
        )
        target = np.stack(
            [
                np.asarray(
                    subject.fem.load_source(
                        subject.fem.source_names[int(source_index)]
                    )[query_indices],
                    dtype=np.float32,
                )
                for source_index in source_indices
            ],
            axis=0,
        )
        statistics = subject.statistics
        data_loss_batch: dict[str, object] = {
            "query_mm": data_queries.world_ras_mm,
            "source_mm": torch.from_numpy(
                subject.model_input.source_coords_mm[source_indices]
            ).unsqueeze(0),
            "reference_mm": torch.from_numpy(
                subject.model_input.reference_coord_mm
            ).unsqueeze(0),
            "pair_weights": torch.from_numpy(
                statistics.pair_weights[source_indices]
            ).unsqueeze(0),
            "global_energy": torch.from_numpy(
                statistics.global_energy[source_indices]
            ).unsqueeze(0),
            "radial_energy": torch.from_numpy(
                statistics.radial_energy[source_indices]
            ).unsqueeze(0),
            "peak_threshold_squared": torch.from_numpy(
                statistics.peak_threshold_squared[source_indices]
            ).unsqueeze(0),
            "peak_energy": torch.from_numpy(
                statistics.peak_energy[source_indices]
            ).unsqueeze(0),
            "global_median_energy": statistics.global_median_energy,
            "radial_median_energy": torch.from_numpy(statistics.radial_median_energy),
            "peak_median_energy": statistics.peak_median_energy,
            "radial_edges_mm": torch.from_numpy(statistics.radial_edges_mm),
            "schedule_loss_scale": self.schedule_loss_scale,
            "voxel_batch_count": selection.query_chunk_count,
        }
        pool = subject.physics_pool
        physics_queries = build_query_bundle(
            subject.model_input,
            pool.union_query_rows,
            device=device,
            geometry=geometry,
        )
        physics_arrays: dict[str, torch.Tensor] = {}
        for name, value in pool.arrays.items():
            if name == "union_query_rows":
                continue
            selected = value
            if name.endswith(
                (
                    "_fem_raw_residual",
                    "_fem_raw_scale",
                    "_valid_mask",
                )
            ) or (
                name.endswith(("_stratum_floor", "_stratum_mad"))
                and value.ndim == 2
                and value.shape[0] == SOURCE_COUNT
            ):
                selected = value[source_indices]
            tensor = torch.from_numpy(np.ascontiguousarray(selected))
            if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                tensor = tensor.long()
            physics_arrays[name] = tensor
        return Head2FieldTrainingBatch(
            subject_key=subject.subject_key,
            head_volume=subject.model_input.head_volume.unsqueeze(0).float(),
            geometry=geometry,
            data_queries=data_queries,
            electrodes=pair_inputs.electrodes,
            pair_index=pair_inputs.pair_index,
            target=torch.from_numpy(target).unsqueeze(0),
            data_loss_batch=data_loss_batch,
            physics=PhysicsTrainingBatch(
                queries=physics_queries,
                arrays=physics_arrays,
            ),
        )


__all__ = [
    "LOSS_STATISTICS_SCHEMA_ID",
    "PHYSICS_POOL_SCHEMA_ID",
    "DeterministicHead2FieldSampler",
    "Head2FieldSelection",
    "Head2FieldTrainingBatch",
    "Head2FieldTrainingDataset",
    "LossStatistics",
    "PhysicsPool",
    "PhysicsTrainingBatch",
    "TrainingSubject",
    "load_loss_statistics",
    "load_physics_pool",
]
