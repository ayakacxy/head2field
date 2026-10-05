"""Stable contracts for training settings, dataset manifests, and run fingerprints."""

from __future__ import annotations

from head2field.metadata import file_records, scientific_metadata, is_digest_key


import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from head2field._spec import (
    HEAD2FIELD_V1_DATA_LOSS_ID,
    HEAD2FIELD_V1_PHYSICS_LOSS_ID,
)
from head2field.preprocessing.conductivity import (
    CONDUCTIVITY_TABLE_ID,
    MODEL_SIGMA_DIVISOR_S_PER_M,
    MODEL_SIGMA_TRANSFORM_ID,
)

TRAINING_CONFIG_SCHEMA_ID = "training_config_v1"
TRAINING_DATASET_SCHEMA_ID = "training_dataset_v1"
TRAINING_STATE_SCHEMA_ID = "head2field_training_state_v1"
TRAINING_LOG_SCHEMA_ID = "head2field_training_log_v1"
TRAINING_PROFILE_ID = "head2field_v1_n128_gp_all_lambda5"
DATA_LOSS_DENOMINATOR_RULE_ID = "target_energy_plus_0.01_subject_source_median_v1"
DATA_LOSS_RADIAL_EDGES_MM = (50.0, 100.0)
PHYSICS_OPERATOR_VERSION = "head2field_v1_integral_operators"
PHYSICS_DENOMINATOR_RULE_ID = "fem_raw_scale_max_train_stratum_floor_v1"
PHYSICS_OBJECTIVE_ID = "one_sided_excess_v1"
REFERENCE_ELECTRODE = "Oz"
SOURCE_COUNT = 75
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class TrainingContractError(ValueError):
    """Indicate training inputs that violate the public v1 contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Return canonical JSON bytes independent of formatting and key order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Compute the SHA256 of canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute the SHA256 of a regular file with bounded memory."""

    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"训练合同文件必须是普通非软链接文件: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value: object, *, field: str) -> PurePosixPath:
    """Parse relative paths without absolute paths, drive letters, backslashes, or parent traversal."""

    if not isinstance(value, str) or not value:
        raise TrainingContractError(f"{field} 必须是非空相对 POSIX 路径")
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise TrainingContractError(f"{field} 不能包含盘符或反斜杠")
    result = PurePosixPath(value)
    if result.is_absolute() or ".." in result.parts or "." in result.parts:
        raise TrainingContractError(f"{field} 必须位于数据清单目录内")
    return result


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    role: str,
) -> None:
    required = {key for key in required if not is_digest_key(key)}
    actual = {key for key in value if not is_digest_key(key)}
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - optional)
    if missing or unexpected:
        raise TrainingContractError(
            f"{role} 字段不匹配: missing={missing}, unexpected={unexpected}"
        )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingContractError(f"{field} 必须是正整数")
    return int(value)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingContractError(f"{field} 必须是非负整数")
    return int(value)


def _finite_float(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingContractError(f"{field} 必须是数值")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise TrainingContractError(f"{field} 必须为有限值")
    if positive and result <= 0.0:
        raise TrainingContractError(f"{field} 必须为正数")
    return result


@dataclass(frozen=True)
class DataLossConfig:
    """Frozen stable-energy data-loss configuration."""

    loss_id: str
    denominator_rule_id: str
    denominator_floor_ratio: float
    pair_weight: float
    radial_weight: float
    peak_weight: float
    huber_delta: float
    radial_edges_mm: tuple[float, float]
    pair_weight_alpha: float
    pair_weight_sigma_mm: float
    peak_quantile: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DataLossConfig:
        required = {
            "loss_id",
            "denominator_rule_id",
            "denominator_floor_ratio",
            "pair_weight",
            "radial_weight",
            "peak_weight",
            "huber_delta",
            "radial_edges_mm",
            "pair_weight_alpha",
            "pair_weight_sigma_mm",
            "peak_quantile",
        }
        _require_exact_keys(value, required=required, role="data_loss")
        edges = value["radial_edges_mm"]
        if (
            not isinstance(edges, list)
            or len(edges) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in edges
            )
        ):
            raise TrainingContractError("data_loss.radial_edges_mm 必须是两个数值")
        result = cls(
            loss_id=str(value["loss_id"]),
            denominator_rule_id=str(value["denominator_rule_id"]),
            denominator_floor_ratio=_finite_float(
                value["denominator_floor_ratio"],
                "data_loss.denominator_floor_ratio",
                positive=True,
            ),
            pair_weight=_finite_float(value["pair_weight"], "data_loss.pair_weight"),
            radial_weight=_finite_float(
                value["radial_weight"], "data_loss.radial_weight"
            ),
            peak_weight=_finite_float(value["peak_weight"], "data_loss.peak_weight"),
            huber_delta=_finite_float(
                value["huber_delta"], "data_loss.huber_delta", positive=True
            ),
            radial_edges_mm=(float(edges[0]), float(edges[1])),
            pair_weight_alpha=_finite_float(
                value["pair_weight_alpha"], "data_loss.pair_weight_alpha"
            ),
            pair_weight_sigma_mm=_finite_float(
                value["pair_weight_sigma_mm"],
                "data_loss.pair_weight_sigma_mm",
                positive=True,
            ),
            peak_quantile=_finite_float(
                value["peak_quantile"], "data_loss.peak_quantile"
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected = {
            "loss_id": HEAD2FIELD_V1_DATA_LOSS_ID,
            "denominator_rule_id": DATA_LOSS_DENOMINATOR_RULE_ID,
            "denominator_floor_ratio": 0.01,
            "pair_weight": 0.60,
            "radial_weight": 0.30,
            "peak_weight": 0.10,
            "huber_delta": 0.05,
            "radial_edges_mm": DATA_LOSS_RADIAL_EDGES_MM,
            "pair_weight_alpha": 1.0,
            "pair_weight_sigma_mm": 40.0,
            "peak_quantile": 0.90,
        }
        mismatched = {
            name: (getattr(self, name), expected_value)
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        }
        if mismatched:
            raise TrainingContractError(f"正式 v1 data loss 合同被修改: {mismatched}")


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Frozen N128 GP all-physics configuration."""

    loss_id: str
    operator_version: str
    objective_id: str
    denominator_rule_id: str
    profile: str
    circulation_primitives: int
    interface_primitives: int
    flux_primitives: int
    lambda_circ: float
    lambda_interface_normal: float
    lambda_interface_tangent: float
    lambda_flux: float
    curriculum_warmup_steps: int
    curriculum_ramp_steps: int
    kappa: float
    huber_delta: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PhysicsLossConfig:
        required = {
            "loss_id",
            "operator_version",
            "objective_id",
            "denominator_rule_id",
            "profile",
            "circulation_primitives",
            "interface_primitives",
            "flux_primitives",
            "lambda_circ",
            "lambda_interface_normal",
            "lambda_interface_tangent",
            "lambda_flux",
            "curriculum_warmup_steps",
            "curriculum_ramp_steps",
            "kappa",
            "huber_delta",
        }
        _require_exact_keys(value, required=required, role="physics_loss")
        result = cls(
            loss_id=str(value["loss_id"]),
            operator_version=str(value["operator_version"]),
            objective_id=str(value["objective_id"]),
            denominator_rule_id=str(value["denominator_rule_id"]),
            profile=str(value["profile"]),
            circulation_primitives=_positive_int(
                value["circulation_primitives"], "physics_loss.circulation_primitives"
            ),
            interface_primitives=_positive_int(
                value["interface_primitives"], "physics_loss.interface_primitives"
            ),
            flux_primitives=_positive_int(
                value["flux_primitives"], "physics_loss.flux_primitives"
            ),
            lambda_circ=_finite_float(value["lambda_circ"], "physics_loss.lambda_circ"),
            lambda_interface_normal=_finite_float(
                value["lambda_interface_normal"],
                "physics_loss.lambda_interface_normal",
            ),
            lambda_interface_tangent=_finite_float(
                value["lambda_interface_tangent"],
                "physics_loss.lambda_interface_tangent",
            ),
            lambda_flux=_finite_float(value["lambda_flux"], "physics_loss.lambda_flux"),
            curriculum_warmup_steps=_nonnegative_int(
                value["curriculum_warmup_steps"],
                "physics_loss.curriculum_warmup_steps",
            ),
            curriculum_ramp_steps=_positive_int(
                value["curriculum_ramp_steps"],
                "physics_loss.curriculum_ramp_steps",
            ),
            kappa=_finite_float(value["kappa"], "physics_loss.kappa"),
            huber_delta=_finite_float(
                value["huber_delta"], "physics_loss.huber_delta", positive=True
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected = {
            "loss_id": HEAD2FIELD_V1_PHYSICS_LOSS_ID,
            "operator_version": PHYSICS_OPERATOR_VERSION,
            "objective_id": PHYSICS_OBJECTIVE_ID,
            "denominator_rule_id": PHYSICS_DENOMINATOR_RULE_ID,
            "profile": "all",
            "circulation_primitives": 512,
            "interface_primitives": 512,
            "flux_primitives": 512,
            "lambda_circ": 5.0,
            "lambda_interface_normal": 5.0,
            "lambda_interface_tangent": 5.0,
            "lambda_flux": 5.0,
            "curriculum_warmup_steps": 0,
            "curriculum_ramp_steps": 10_000,
            "kappa": 1.0,
            "huber_delta": 0.10,
        }
        mismatched = {
            name: (getattr(self, name), expected_value)
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        }
        if mismatched:
            raise TrainingContractError(
                f"正式 v1 N128 GP physics 合同被修改: {mismatched}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Public epoch-training configuration with separate smoke and source-weight provenance."""

    schema_id: str
    profile_id: str
    execution_profile_id: str
    official_weight_reproduction: bool
    model_profile_id: str
    model_config: str
    encoder_input_shape: tuple[int, int, int]
    device: str
    precision: str
    seed: int
    max_epochs: int
    cosine_horizon_epochs: int
    early_stopping_patience: int
    amp_initial_scale: float
    amp_growth_interval: int
    query_count: int
    sources_per_step: int
    optimizer: str
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    scheduler: str
    gradient_clip_norm: float
    state_interval_steps: int
    data_loss: DataLossConfig
    physics_loss: PhysicsLossConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingConfig:
        required = {
            "schema_id",
            "profile_id",
            "execution_profile_id",
            "official_weight_reproduction",
            "model_profile_id",
            "model_config",
            "encoder_input_shape",
            "device",
            "precision",
            "seed",
            "max_epochs",
            "cosine_horizon_epochs",
            "early_stopping_patience",
            "amp_initial_scale",
            "amp_growth_interval",
            "query_count",
            "sources_per_step",
            "optimizer",
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "scheduler",
            "gradient_clip_norm",
            "state_interval_steps",
            "data_loss",
            "physics_loss",
        }
        _require_exact_keys(value, required=required, role="training config")
        shape = value["encoder_input_shape"]
        if not isinstance(shape, list) or len(shape) != 3:
            raise TrainingContractError("encoder_input_shape 必须是三个正整数")
        if not isinstance(value["data_loss"], Mapping) or not isinstance(
            value["physics_loss"], Mapping
        ):
            raise TrainingContractError("data_loss/physics_loss 必须是 JSON 对象")
        reproduction = value["official_weight_reproduction"]
        if not isinstance(reproduction, bool):
            raise TrainingContractError("official_weight_reproduction 必须是布尔值")
        result = cls(
            schema_id=str(value["schema_id"]),
            profile_id=str(value["profile_id"]),
            execution_profile_id=str(value["execution_profile_id"]),
            official_weight_reproduction=reproduction,
            model_profile_id=str(value["model_profile_id"]),
            model_config=str(value["model_config"]),
            encoder_input_shape=tuple(
                _positive_int(item, "encoder_input_shape") for item in shape
            ),
            device=str(value["device"]),
            precision=str(value["precision"]),
            seed=_nonnegative_int(value["seed"], "seed"),
            max_epochs=_positive_int(value["max_epochs"], "max_epochs"),
            cosine_horizon_epochs=_positive_int(
                value["cosine_horizon_epochs"], "cosine_horizon_epochs"
            ),
            early_stopping_patience=_positive_int(
                value["early_stopping_patience"], "early_stopping_patience"
            ),
            amp_initial_scale=_finite_float(
                value["amp_initial_scale"], "amp_initial_scale", positive=True
            ),
            amp_growth_interval=_positive_int(
                value["amp_growth_interval"], "amp_growth_interval"
            ),
            query_count=_positive_int(value["query_count"], "query_count"),
            sources_per_step=_positive_int(
                value["sources_per_step"], "sources_per_step"
            ),
            optimizer=str(value["optimizer"]),
            learning_rate=_finite_float(
                value["learning_rate"], "learning_rate", positive=True
            ),
            min_learning_rate=_finite_float(
                value["min_learning_rate"], "min_learning_rate"
            ),
            weight_decay=_finite_float(value["weight_decay"], "weight_decay"),
            scheduler=str(value["scheduler"]),
            gradient_clip_norm=_finite_float(
                value["gradient_clip_norm"], "gradient_clip_norm", positive=True
            ),
            state_interval_steps=_positive_int(
                value["state_interval_steps"], "state_interval_steps"
            ),
            data_loss=DataLossConfig.from_mapping(value["data_loss"]),
            physics_loss=PhysicsLossConfig.from_mapping(value["physics_loss"]),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise TrainingContractError(f"训练配置不是有效 JSON: {path}") from error
        if not isinstance(payload, Mapping):
            raise TrainingContractError("训练配置顶层必须是 JSON 对象")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        if self.schema_id != TRAINING_CONFIG_SCHEMA_ID:
            raise TrainingContractError(
                f"training config schema_id 必须为 {TRAINING_CONFIG_SCHEMA_ID!r}"
            )
        if self.profile_id != TRAINING_PROFILE_ID:
            raise TrainingContractError(f"训练 profile 必须为 {TRAINING_PROFILE_ID!r}")
        if self.official_weight_reproduction:
            raise TrainingContractError("自定义训练配置不得声明已经复现正式权重")
        safe_relative_path(self.model_config, field="model_config")
        if self.execution_profile_id == "cpu_fp32_synthetic_smoke":
            if (
                self.model_profile_id != "small_same_io_contract_test_only"
                or self.encoder_input_shape != (8, 8, 8)
            ):
                raise TrainingContractError(
                    "CPU smoke 必须使用八立方体 test-only 小模型"
                )
            if self.device != "cpu" or self.precision != "fp32":
                raise TrainingContractError("CPU synthetic smoke 只允许 CPU FP32")
        elif self.execution_profile_id == "cuda_amp_fp16_epoch_training":
            if self.model_profile_id != "head2field_v1" or self.encoder_input_shape != (
                128,
                128,
                128,
            ):
                raise TrainingContractError(
                    "正式训练必须使用固定 v1 模型和 128³ encoder"
                )
            if self.device != "cuda" or self.precision != "amp_fp16":
                raise TrainingContractError("正式训练使用 CUDA AMP FP16")
        else:
            raise TrainingContractError("未知训练 execution profile")
        if self.optimizer != "adamw" or self.scheduler != "cosine":
            raise TrainingContractError("训练 optimizer/scheduler 必须为 adamw/cosine")
        if self.max_epochs > 25 or self.cosine_horizon_epochs != 25:
            raise TrainingContractError(
                "训练最多 25 epochs，cosine horizon 固定为 25 epochs"
            )
        if self.early_stopping_patience != 5:
            raise TrainingContractError("早停 patience 固定为 5 个验证 epoch")
        if self.min_learning_rate < 0.0:
            raise TrainingContractError("min_learning_rate 不能为负数")
        if self.sources_per_step > SOURCE_COUNT:
            raise TrainingContractError("sources_per_step 不能超过 75")
        if self.min_learning_rate > self.learning_rate:
            raise TrainingContractError("min_learning_rate 不能大于 learning_rate")
        if self.weight_decay < 0.0:
            raise TrainingContractError("weight_decay 不能为负数")
        self.data_loss.validate()
        self.physics_loss.validate()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["encoder_input_shape"] = list(self.encoder_input_shape)
        value["data_loss"]["radial_edges_mm"] = list(self.data_loss.radial_edges_mm)
        return value

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class SubjectRecord:
    """Controlled training-asset references for one anonymous subject."""

    subject_key: str
    group_key: str
    prepared_subject_manifest: PurePosixPath
    prepared_subject_manifest_sha256: str
    fem_labels_manifest: PurePosixPath
    fem_labels_manifest_sha256: str
    loss_statistics_manifest: PurePosixPath
    loss_statistics_manifest_sha256: str
    physics_pool_manifest: PurePosixPath
    physics_pool_manifest_sha256: str


@dataclass(frozen=True)
class TrainingDatasetManifest:
    """已校验文件路径、划分和数据顺序的训练数据清单。"""

    path: Path
    raw: Mapping[str, Any]
    manifest_file_sha256: str
    canonical_sha256: str
    split_sha256: str
    data_sha256: str
    source_names: tuple[str, ...]
    groups: Mapping[str, str]
    subjects: tuple[SubjectRecord, ...]

    def subjects_for_partition(self, partition: str) -> tuple[SubjectRecord, ...]:
        return tuple(
            subject
            for subject in self.subjects
            if self.groups[subject.group_key] == partition
        )


def _legacy_provenance_text(value: object, field: str) -> str:
    result = str(value)
    return result


def _resolve_dataset_file(
    base_dir: Path,
    relative: PurePosixPath,
    expected_sha256: str,
    *,
    field: str,
) -> Path:
    base = base_dir.resolve(strict=True)
    candidate = base.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise TrainingContractError(f"{field} 解析后逃逸数据目录") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise TrainingContractError(f"{field} 必须是普通非软链接文件")
    return resolved


def load_training_dataset_manifest(path: Path) -> TrainingDatasetManifest:
    """Strictly load a generic group-aware training dataset manifest."""

    manifest_path = Path(path).resolve(strict=True)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TrainingContractError("训练数据集 manifest 必须是普通非软链接文件")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TrainingContractError("训练数据集 manifest 不是有效 JSON") from error
    if not isinstance(raw, Mapping):
        raise TrainingContractError("训练数据集 manifest 顶层必须是 JSON 对象")
    required = {
        "schema_id",
        "split_policy",
        "reference_electrode",
        "source_names",
        "source_order_sha256",
        "conductivity_table_id",
        "conductivity_table_sha256",
        "sigma_model_transform_id",
        "sigma_model_divisor_s_per_m",
        "groups",
        "subjects",
    }
    _require_exact_keys(raw, required=required, role="training dataset")
    fixed = {
        "schema_id": TRAINING_DATASET_SCHEMA_ID,
        "split_policy": "group_disjoint_v1",
        "reference_electrode": REFERENCE_ELECTRODE,
        "conductivity_table_id": CONDUCTIVITY_TABLE_ID,
        "sigma_model_transform_id": MODEL_SIGMA_TRANSFORM_ID,
        "sigma_model_divisor_s_per_m": MODEL_SIGMA_DIVISOR_S_PER_M,
    }
    mismatched = {
        key: (raw.get(key), expected)
        for key, expected in fixed.items()
        if not is_digest_key(key) and raw.get(key) != expected
    }
    if mismatched:
        raise TrainingContractError(f"training dataset 固定字段不匹配: {mismatched}")

    source_names_raw = raw["source_names"]
    if (
        not isinstance(source_names_raw, list)
        or len(source_names_raw) != SOURCE_COUNT
        or any(not isinstance(name, str) or not name for name in source_names_raw)
    ):
        raise TrainingContractError("source_names 必须恰含 75 个非空名称")
    source_names = tuple(source_names_raw)
    if len(set(source_names)) != SOURCE_COUNT or REFERENCE_ELECTRODE in source_names:
        raise TrainingContractError("source_names 必须唯一且不能含 Oz")

    groups_raw = raw["groups"]
    if not isinstance(groups_raw, list) or not groups_raw:
        raise TrainingContractError("groups 必须是非空数组")
    groups: dict[str, str] = {}
    for index, group in enumerate(groups_raw):
        if not isinstance(group, Mapping):
            raise TrainingContractError(f"groups[{index}] 必须是对象")
        _require_exact_keys(
            group,
            required={"group_key", "partition"},
            role=f"groups[{index}]",
        )
        key = str(group["group_key"])
        partition = str(group["partition"])
        if not key or key in groups:
            raise TrainingContractError("group_key 必须非空且唯一")
        if partition not in {"train", "validation", "test"}:
            raise TrainingContractError(f"未知 partition: {partition!r}")
        groups[key] = partition
    if "train" not in groups.values() or "validation" not in groups.values():
        raise TrainingContractError("数据集必须同时包含 train 与 validation group")

    subjects_raw = raw["subjects"]
    if not isinstance(subjects_raw, list) or not subjects_raw:
        raise TrainingContractError("subjects 必须是非空数组")
    subject_records: list[SubjectRecord] = []
    seen_subjects: set[str] = set()
    file_hash_records: list[dict[str, str]] = []
    base_dir = manifest_path.parent
    asset_fields = (
        "prepared_subject_manifest",
        "fem_labels_manifest",
        "loss_statistics_manifest",
        "physics_pool_manifest",
    )
    for index, subject in enumerate(subjects_raw):
        if not isinstance(subject, Mapping):
            raise TrainingContractError(f"subjects[{index}] 必须是对象")
        required_subject = {"subject_key", "group_key"}
        for asset_field in asset_fields:
            required_subject.add(asset_field)
            required_subject.add(f"{asset_field}_sha256")
        _require_exact_keys(
            subject,
            required=required_subject,
            role=f"subjects[{index}]",
        )
        subject_key = str(subject["subject_key"])
        group_key = str(subject["group_key"])
        if not subject_key or subject_key in seen_subjects:
            raise TrainingContractError("subject_key 必须非空且唯一")
        if group_key not in groups:
            raise TrainingContractError(f"subject 引用了未知 group_key: {group_key!r}")
        seen_subjects.add(subject_key)
        values: dict[str, Any] = {"subject_key": subject_key, "group_key": group_key}
        for asset_field in asset_fields:
            relative = safe_relative_path(subject[asset_field], field=asset_field)
            expected_hash = _legacy_provenance_text(
                subject.get(f"{asset_field}_sha256", ""),
                f"{asset_field}_sha256",
            )
            _resolve_dataset_file(
                base_dir,
                relative,
                expected_hash,
                field=asset_field,
            )
            values[asset_field] = relative
            values[f"{asset_field}_sha256"] = expected_hash
            file_hash_records.append(
                {
                    "subject_key": subject_key,
                    "role": asset_field,
                    "path": relative.as_posix(),
                }
            )
        subject_records.append(SubjectRecord(**values))

    split_payload = {
        "split_policy": raw["split_policy"],
        "groups": sorted(groups.items()),
        "subjects": sorted(
            (subject.subject_key, subject.group_key) for subject in subject_records
        ),
    }
    data_payload = {
        "sigma_model_transform_id": MODEL_SIGMA_TRANSFORM_ID,
        "files": sorted(
            file_hash_records, key=lambda item: (item["subject_key"], item["role"])
        ),
    }
    return TrainingDatasetManifest(
        path=manifest_path,
        raw=raw,
        manifest_file_sha256=sha256_file(manifest_path),
        canonical_sha256=canonical_json_sha256(raw),
        split_sha256=canonical_json_sha256(split_payload),
        data_sha256=canonical_json_sha256(data_payload),
        source_names=source_names,
        groups=groups,
        subjects=tuple(subject_records),
    )


def build_run_contract(
    config: TrainingConfig,
    dataset: TrainingDatasetManifest,
) -> dict[str, Any]:
    """记录恢复所需的模型参数、训练划分、电极顺序和数据路径。"""
    base = dataset.path.parent
    inputs = {dataset.path}
    for subject in dataset.subjects:
        for relative in (
            subject.prepared_subject_manifest,
            subject.fem_labels_manifest,
            subject.loss_statistics_manifest,
            subject.physics_pool_manifest,
        ):
            manifest = base.joinpath(*relative.parts)
            inputs.add(manifest)
            inputs.update(
                path
                for path in manifest.parent.rglob("*")
                if path.is_file() and path.suffix in {".json", ".npy", ".npz", ".gz"}
            )
    return {
        "schema_id": "head2field_training_run_contract_v3",
        "configuration": config.to_dict(),
        "dataset": scientific_metadata(dataset.raw),
        "input_files": file_records(inputs, relative_to=base),
        "source_names": list(dataset.source_names),
        "conductivity_values_s_per_m": [
            0.126,
            0.275,
            1.654,
            0.01,
            0.465,
            0.5,
            0.008,
            0.025,
            0.6,
            0.16,
        ],
        "sigma_model_divisor_s_per_m": MODEL_SIGMA_DIVISOR_S_PER_M,
        "data_loss_epoch_scale_rule": "steps_per_epoch_divide_75_training_subjects",
    }


__all__ = [
    "DATA_LOSS_DENOMINATOR_RULE_ID",
    "DATA_LOSS_RADIAL_EDGES_MM",
    "PHYSICS_DENOMINATOR_RULE_ID",
    "PHYSICS_OBJECTIVE_ID",
    "PHYSICS_OPERATOR_VERSION",
    "REFERENCE_ELECTRODE",
    "SOURCE_COUNT",
    "TRAINING_CONFIG_SCHEMA_ID",
    "TRAINING_DATASET_SCHEMA_ID",
    "TRAINING_LOG_SCHEMA_ID",
    "TRAINING_PROFILE_ID",
    "TRAINING_STATE_SCHEMA_ID",
    "DataLossConfig",
    "PhysicsLossConfig",
    "SubjectRecord",
    "TrainingConfig",
    "TrainingContractError",
    "TrainingDatasetManifest",
    "build_run_contract",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_training_dataset_manifest",
    "safe_relative_path",
    "sha256_file",
]
