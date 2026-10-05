"""Fixed v1 76-electrode inventory and public input validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STANDARD_76_ELECTRODE_NAMES: tuple[str, ...] = (
    "Oz",
    "Fp1",
    "Fpz",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "PO10",
    "O1",
    "O2",
    "Iz",
    "I1",
    "I2",
)
DEFAULT_SIMNIBS_CAP_RELATIVE_PATH = Path("eeg_positions/EEG10-10_UI_Jurak_2007.csv")


def load_standard_electrode_inputs(
    names_path: Path,
    coordinates_path: Path,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Read and strictly validate the public 76 electrode names and world-RAS coordinates."""

    try:
        raw_names = json.loads(Path(names_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"电极名称文件不是有效 JSON：{names_path}") from exc
    if not isinstance(raw_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in raw_names
    ):
        raise TypeError("电极名称输入必须是非空字符串 JSON 列表")
    names = tuple(raw_names)
    if names != STANDARD_76_ELECTRODE_NAMES:
        raise ValueError(
            "电极名称和顺序必须严格等于 v1 固定 76 电极清单，且 Oz 位于首位"
        )

    raw_coordinates = np.load(Path(coordinates_path), allow_pickle=False)
    if (
        raw_coordinates.dtype == np.bool_
        or not np.issubdtype(raw_coordinates.dtype, np.number)
        or np.iscomplexobj(raw_coordinates)
        or raw_coordinates.shape != (len(names), 3)
    ):
        raise TypeError("电极坐标必须是实数 [76, 3] NPY 数组")
    coordinates = np.asarray(raw_coordinates, dtype=np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError("电极 world-RAS 坐标含 NaN 或 Inf")
    return names, coordinates


def load_standard_cap_from_head_model(
    head_model_dir: Path,
    *,
    cap_relative_path: Path = DEFAULT_SIMNIBS_CAP_RELATIVE_PATH,
) -> tuple[tuple[str, ...], np.ndarray, Path]:
    """Extract the standard cap from this CHARM model and reorder it to the fixed 76 names."""

    relative = Path(cap_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("cap_relative_path 必须是 head-model 内的安全相对路径")
    root = Path(head_model_dir).expanduser().resolve(strict=True)
    cap_path = (root / relative).resolve(strict=True)
    try:
        cap_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("标准 cap 路径逃逸 head-model 目录") from exc
    if not cap_path.is_file() or cap_path.is_symlink():
        raise FileNotFoundError(f"CHARM 标准 cap 不是普通文件：{cap_path}")

    coordinates_by_name: dict[str, np.ndarray] = {}
    with cap_path.open("r", newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row:
                continue
            if row[0].strip() not in {"Electrode", "ReferenceElectrode"}:
                continue
            if len(row) < 5:
                raise ValueError(f"标准 cap 第 {line_number} 行少于 5 列")
            name = row[4].strip()
            if name in coordinates_by_name:
                raise ValueError(f"标准 cap 包含重复电极：{name!r}")
            try:
                coordinate = np.asarray(
                    [float(row[1]), float(row[2]), float(row[3])],
                    dtype=np.float64,
                )
            except ValueError as exc:
                raise ValueError(f"标准 cap 电极 {name!r} 的坐标不是有限实数") from exc
            if not np.isfinite(coordinate).all():
                raise ValueError(f"标准 cap 电极 {name!r} 坐标含 NaN 或 Inf")
            coordinates_by_name[name] = coordinate
    actual_names = set(coordinates_by_name)
    expected_names = set(STANDARD_76_ELECTRODE_NAMES)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"CHARM 标准 cap 不满足固定 76 电极集合：missing={missing}, extra={extra}"
        )
    coordinates = np.asarray(
        [coordinates_by_name[name] for name in STANDARD_76_ELECTRODE_NAMES],
        dtype=np.float32,
    )
    return STANDARD_76_ELECTRODE_NAMES, coordinates, cap_path


__all__ = [
    "DEFAULT_SIMNIBS_CAP_RELATIVE_PATH",
    "STANDARD_76_ELECTRODE_NAMES",
    "load_standard_cap_from_head_model",
    "load_standard_electrode_inputs",
]
