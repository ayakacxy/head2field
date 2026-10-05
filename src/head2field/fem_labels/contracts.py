"""Define the v1 FEM label-generation and HDF5 input contracts."""

from __future__ import annotations


from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing.conductivity import (
    conductivity_table_metadata,
    validate_conductivity_table,
)
from ..preprocessing.electrodes import STANDARD_76_ELECTRODE_NAMES


class FEMV1ContractError(ValueError):
    """Indicate that an input violates the official v1 FEM contract."""


@dataclass(frozen=True)
class FEMV1ProductionProfile:
    """Store immutable settings for official v1 FEM label generation."""

    profile_id: str = "fem-labels-v1-production"
    reference: str = "Oz"
    electrode_count: int = 76
    source_count: int = 75
    electrode_names_all: tuple[str, ...] = STANDARD_76_ELECTRODE_NAMES
    reference_mode: str = "direct_e_to_reference"
    output_dtype: str = "float32"
    scale: float = 0.001
    tissue_labels: tuple[int, int] = (1, 2)
    keep_tissues: tuple[int, int] = (1, 2)
    interpolation_method: str = "linear"
    interpolation_backend: str = "simnibs"
    compute_device: str = "cpu"
    simnibs_version: str = "4.6.0"
    solver_backend: str = "pardiso"
    solver_device: str = "cpu"
    source_current: str = "1A"
    field: str = "E"
    source_data_type: str = "element_data"
    source_units: str = "V/m"
    output_units: str = "V/m"
    output_source_current_ma: float = 1.0
    output_reference_current_ma: float = -1.0
    vector_frame: str = "world-RAS"
    component_order: tuple[str, str, str] = ("Ex", "Ey", "Ez")
    label_axis_order: str = "query,component"
    training_target_axis_order: str = "query,source,component"

    def to_manifest(self) -> dict[str, Any]:
        """Return official generation settings in a JSON-compatible format."""

        payload = asdict(self)
        payload["tissue_labels"] = list(self.tissue_labels)
        payload["keep_tissues"] = list(self.keep_tissues)
        payload["electrode_names_all"] = list(self.electrode_names_all)
        payload["component_order"] = list(self.component_order)
        table = conductivity_table_metadata()
        validate_conductivity_table(table)
        payload["conductivity_table"] = table
        return payload


PRODUCTION_PROFILE = FEMV1ProductionProfile()
SMOKE_PROFILE_ID = "fem-labels-v1-smoke"


def _text(value: Any) -> str:
    """Normalize common HDF5 string representations to stripped text."""

    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _text_list(values: Any) -> list[str]:
    """Normalize an HDF5 string array to a Python string list."""

    if isinstance(values, (str, bytes)):
        return [_text(values)]
    return [_text(value) for value in values]


def validate_production_options(
    *,
    reference: str,
    reference_mode: str,
    output_dtype: str,
    scale: float,
    tissue_labels: Sequence[int],
    keep_tissues: Sequence[int],
    method: str,
    interpolation_backend: str,
    compute_device: str,
    requested_electrodes: Sequence[str] | None,
    limit: int | None,
    profile: FEMV1ProductionProfile = PRODUCTION_PROFILE,
) -> None:
    """Reject parameter drift that could be mistaken for an official v1 artifact."""

    mismatches: list[str] = []
    checks = {
        "reference": (reference, profile.reference),
        "reference_mode": (reference_mode, profile.reference_mode),
        "output_dtype": (np.dtype(output_dtype).name, profile.output_dtype),
        "tissue_labels": (
            tuple(int(v) for v in tissue_labels),
            profile.tissue_labels,
        ),
        "keep_tissues": (
            tuple(int(v) for v in keep_tissues),
            profile.keep_tissues,
        ),
        "method": (method, profile.interpolation_method),
        "interpolation_backend": (
            interpolation_backend,
            profile.interpolation_backend,
        ),
        "compute_device": (compute_device, profile.compute_device),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            mismatches.append(f"{name}={actual!r}，期望 {expected!r}")
    if not np.isclose(float(scale), profile.scale, rtol=0.0, atol=0.0):
        mismatches.append(f"scale={scale!r}，期望 {profile.scale!r}")
    if requested_electrodes is not None:
        mismatches.append("正式生产必须导出全部 75 个 source，不能指定 --electrodes")
    if limit is not None:
        mismatches.append("正式生产必须导出全部 75 个 source，不能指定 --limit")
    if mismatches:
        details = "；".join(mismatches)
        raise FEMV1ContractError(f"正式 v1 FEM 参数合同不匹配：{details}")


def validate_leadfield_dataset_contract(
    *,
    shape: Sequence[int],
    attrs: Mapping[str, Any],
    profile: FEMV1ProductionProfile = PRODUCTION_PROFILE,
) -> dict[str, Any]:
    """Validate the official v1 contract of a SimNIBS lead-field dataset."""

    required = {
        "current",
        "d_type",
        "electrode_names",
        "electrode_pos",
        "field",
        "reference_electrode",
        "tissues",
        "units",
    }
    missing = sorted(required - set(attrs))
    if missing:
        raise FEMV1ContractError(f"leadfield HDF5 缺少属性：{missing}")

    normalized_shape = tuple(int(value) for value in shape)
    if len(normalized_shape) != 3:
        raise FEMV1ContractError(
            f"leadfield dataset 必须是三维，实际 shape={normalized_shape}"
        )
    if normalized_shape[0] != profile.source_count:
        raise FEMV1ContractError(
            "leadfield source 轴长度错误："
            f"实际 {normalized_shape[0]}，期望 {profile.source_count}"
        )
    if normalized_shape[1] <= 0 or normalized_shape[2] != 3:
        raise FEMV1ContractError(
            "leadfield dataset 必须为 [75, N_elements, 3] 且 N_elements>0，"
            f"实际 shape={normalized_shape}"
        )

    names_all = _text_list(attrs["electrode_names"])
    if len(names_all) != profile.electrode_count:
        raise FEMV1ContractError(
            f"电极数量错误：实际 {len(names_all)}，期望 {profile.electrode_count}"
        )
    if any(not name for name in names_all):
        raise FEMV1ContractError("电极名称不得为空")
    if len({name.casefold() for name in names_all}) != len(names_all):
        raise FEMV1ContractError("电极名称必须大小写不敏感地唯一")
    if names_all[0] != profile.reference or names_all.count(profile.reference) != 1:
        raise FEMV1ContractError(
            f"Oz 必须是唯一参考并位于 electrode_names 首位，实际首项={names_all[0]!r}"
        )
    if names_all != list(profile.electrode_names_all):
        mismatch_index = next(
            index
            for index, (actual, expected) in enumerate(
                zip(names_all, profile.electrode_names_all, strict=True)
            )
            if actual != expected
        )
        raise FEMV1ContractError(
            "标准 76 电极名称或顺序不一致："
            f"index={mismatch_index}, actual={names_all[mismatch_index]!r}, "
            f"expected={profile.electrode_names_all[mismatch_index]!r}"
        )

    expected_text = {
        "reference_electrode": profile.reference,
        "current": profile.source_current,
        "field": profile.field,
        "d_type": profile.source_data_type,
        "units": profile.source_units,
    }
    for name, expected in expected_text.items():
        actual = _text(attrs[name])
        if actual != expected:
            raise FEMV1ContractError(
                f"leadfield 属性 {name}={actual!r}，期望 {expected!r}"
            )

    try:
        raw_tissues = np.asarray(attrs["tissues"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise FEMV1ContractError("leadfield tissues 必须是整数数组") from exc
    if not np.isfinite(raw_tissues).all() or not np.array_equal(
        raw_tissues, np.trunc(raw_tissues)
    ):
        raise FEMV1ContractError("leadfield tissues 必须是有限整数")
    tissues = tuple(int(value) for value in raw_tissues)
    if tissues != profile.keep_tissues:
        raise FEMV1ContractError(
            f"leadfield tissues={tissues!r}，期望 {profile.keep_tissues!r}"
        )

    coordinates = np.asarray(attrs["electrode_pos"], dtype=np.float64)
    expected_coordinate_shape = (profile.electrode_count, 3)
    if coordinates.shape != expected_coordinate_shape:
        raise FEMV1ContractError(
            f"电极坐标 shape={coordinates.shape}，期望 {expected_coordinate_shape}"
        )
    if not np.isfinite(coordinates).all():
        raise FEMV1ContractError("电极 world 坐标包含 NaN 或 Inf")

    return {
        "profile_id": profile.profile_id,
        "dataset_shape": list(normalized_shape),
        "electrode_names_all": names_all,
        "electrode_names_no_reference": names_all[1:],
        "reference": profile.reference,
        "reference_index": 0,
        "current": profile.source_current,
        "field": profile.field,
        "data_type": profile.source_data_type,
        "units": profile.source_units,
        "tissues": list(tissues),
        "electrode_coordinate_shape": list(coordinates.shape),
        "electrode_coordinates_mm": coordinates.tolist(),
        "coordinates_finite": True,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash files in fixed-size chunks without loading large HDF5 files into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
