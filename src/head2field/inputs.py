"""Build authoritative v1 model input bundles from a read-only prepared subject."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from .model import (
    ElectrodeCoordinateBundle,
    QueryCoordinateBundle,
    SubjectGeometry,
)
from .preprocessing.conductivity import (
    labels_to_conductivity,
    normalize_conductivity_for_model,
    validate_conductivity_arrays,
)
from .preprocessing.validation import PreparedSubjectV1


class ModelInputSubject(Protocol):
    """Minimal read-only subject fields shared by training and inference."""

    subject_id: str
    head_volume: torch.Tensor
    tissue_labels_1mm: np.ndarray
    t1_normalized_1mm: np.ndarray
    affine: np.ndarray
    inverse_affine: np.ndarray
    full_shape: tuple[int, int, int]
    gmwm_ijk: np.ndarray
    source_names: Sequence[str]
    source_coords_mm: np.ndarray
    reference_coord_mm: np.ndarray


@dataclass(frozen=True)
class PreparedModelInputSubject:
    """Label-free model input constructed read-only from prepared-subject v1."""

    subject_id: str
    head_volume: torch.Tensor
    tissue_labels_1mm: np.ndarray
    t1_normalized_1mm: np.ndarray
    affine: np.ndarray
    inverse_affine: np.ndarray
    full_shape: tuple[int, int, int]
    gmwm_ijk: np.ndarray
    source_names: tuple[str, ...]
    source_coords_mm: np.ndarray
    reference_coord_mm: np.ndarray


def normalize_ijk(ijk: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Scale NIfTI voxel coordinates to the ``[-1,1]`` range used by ``grid_sample``."""

    denominator = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    return 2.0 * np.asarray(ijk, dtype=np.float32) / denominator - 1.0


def world_to_normalized_ijk(
    xyz_mm: np.ndarray,
    inverse_affine: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert world-RAS coordinates in millimeters to normalized voxel coordinates."""

    points = np.asarray(xyz_mm)
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)],
        axis=-1,
    )
    ijk = homogeneous @ np.asarray(inverse_affine).T
    return normalize_ijk(ijk[..., :3], shape)


def normalize_t1_for_model(
    t1_1mm: np.ndarray,
    tissue_labels_1mm: np.ndarray,
) -> np.ndarray:
    """Normalize foreground T1 by its median and 5th-to-95th percentile span.

    Values are clipped to [-2,2] and background is zero. The input is copied;
    this function does not modify the prepared image or tissue array.
    """

    t1 = np.asarray(t1_1mm, dtype=np.float32).copy()
    tissues = np.asarray(tissue_labels_1mm)
    if t1.shape != tissues.shape:
        raise ValueError("T1 and tissue label shapes must match")
    if not np.all(np.isfinite(t1)):
        raise ValueError("T1 contains NaN or Inf")
    foreground = tissues > 0
    if np.any(foreground):
        values = t1[foreground]
        center = float(np.median(values))
        scale = float(np.percentile(values, 95) - np.percentile(values, 5))
        t1 = np.clip((t1 - center) / max(scale, 1e-6), -2.0, 2.0)
    t1[~foreground] = 0.0
    return np.asarray(t1, dtype=np.float32)


def build_model_head_inputs(
    tissue_labels_1mm: np.ndarray,
    sigma_s_per_m_1mm: np.ndarray,
    t1_1mm: np.ndarray,
    encoder_input_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, np.ndarray]:
    """Return a resampled ``[12,I,J,K]`` tensor and native-grid normalized T1.

    Channels are tissue IDs 1..10, conductivity divided by 1.654, and T1. The
    encoder volume is float32; the native T1 array is stored as float16 for
    later query-feature construction. Resampling preserves physical endpoints
    through ``align_corners=True``.
    """

    tissues = np.asarray(tissue_labels_1mm)
    sigma = np.asarray(sigma_s_per_m_1mm)
    validate_conductivity_arrays(tissues, sigma, context="v1 model inputs")
    sigma_model = normalize_conductivity_for_model(sigma)
    t1_normalized = normalize_t1_for_model(t1_1mm, tissues)
    channels = [(tissues == label).astype(np.float32) for label in range(1, 11)]
    channels.extend([sigma_model, t1_normalized])
    volume = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0)
    volume = F.interpolate(
        volume,
        size=tuple(int(value) for value in encoder_input_shape),
        mode="trilinear",
        align_corners=True,
    )
    return volume.squeeze(0).contiguous(), t1_normalized.astype(np.float16)


def build_query_features_1mm(
    subject: ModelInputSubject,
    query_ijk: np.ndarray,
) -> np.ndarray:
    """Build 12-dimensional tissue, conductivity, and T1 query features in frozen row order."""

    indices = tuple(np.asarray(query_ijk, dtype=np.int64).T)
    labels = np.asarray(subject.tissue_labels_1mm)[indices]
    t1 = np.asarray(subject.t1_normalized_1mm)[indices].astype(np.float32)
    one_hot = np.stack(
        [(labels == label).astype(np.float32) for label in range(1, 11)],
        axis=-1,
    )
    sigma = normalize_conductivity_for_model(labels_to_conductivity(labels))
    return np.concatenate(
        [one_hot, sigma[:, None], t1[:, None]],
        axis=1,
    ).astype(np.float32, copy=False)


def build_prepared_model_input_subject(
    prepared: PreparedSubjectV1,
    *,
    encoder_input_shape: tuple[int, int, int],
) -> PreparedModelInputSubject:
    """Adapt a validated prepared subject to label-free model inputs.

    Keep the prepared affine, GM/WM row order, source order, and Oz coordinate.
    Only the encoder volume is resampled; query features retain the 1 mm grid.
    Call ``load_prepared_subject_v1`` first to validate files and hashes.
    """

    head_volume, t1_normalized = build_model_head_inputs(
        prepared.tissue_labels_1mm,
        prepared.sigma_s_per_m_1mm,
        prepared.t1_1mm,
        encoder_input_shape,
    )
    return PreparedModelInputSubject(
        subject_id=prepared.subject_dir.name,
        head_volume=head_volume,
        tissue_labels_1mm=prepared.tissue_labels_1mm,
        t1_normalized_1mm=t1_normalized,
        affine=prepared.affine,
        inverse_affine=prepared.inverse_affine,
        full_shape=prepared.full_shape,
        gmwm_ijk=prepared.gmwm_ijk,
        source_names=prepared.source_names,
        source_coords_mm=prepared.source_coords_mm,
        reference_coord_mm=prepared.reference_coord_mm,
    )


def ijk_to_world_mm(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert ``[N,3]`` voxel coordinates to world-RAS millimeters."""

    homogeneous = np.concatenate(
        [ijk, np.ones((len(ijk), 1), dtype=ijk.dtype)],
        axis=1,
    )
    return (homogeneous @ np.asarray(affine).T)[:, :3]


@dataclass(frozen=True)
class ElectrodePairInputs:
    """Stable unique electrode bank and pair indices referring to that bank."""

    electrodes: ElectrodeCoordinateBundle
    pair_index: torch.Tensor

    @property
    def pair_count(self) -> int:
        return int(self.pair_index.shape[1])


def _as_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _float_tensor(
    value: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    array = np.ascontiguousarray(value, dtype=np.float32)
    return torch.from_numpy(array).to(device=device)


def _row_indices(
    row_indices: Sequence[int] | np.ndarray,
    *,
    row_count: int,
) -> np.ndarray:
    raw = np.asarray(row_indices)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("row_indices must be a non-empty 1D integer sequence")
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("row_indices must use an integer dtype")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= row_count):
        raise IndexError(f"row_indices is out of range; valid range is [0,{row_count})")
    return result


def _world_points(value: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError(
            f"source_world_ras_mm must be a non-empty [P,3] array; got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("source_world_ras_mm contains NaN or Inf")
    return np.ascontiguousarray(result)


def build_subject_geometry(
    subject: ModelInputSubject,
    *,
    device: torch.device | str,
) -> SubjectGeometry:
    """Build and validate the v1 geometry contract for one subject."""

    resolved_device = _as_device(device)
    gmwm_ijk = np.asarray(subject.gmwm_ijk)
    if gmwm_ijk.ndim != 2 or gmwm_ijk.shape[1] != 3 or not len(gmwm_ijk):
        raise ValueError("model-input gmwm_ijk must be a non-empty [N,3] array")
    origin_ijk = gmwm_ijk.astype(np.float64, copy=False).mean(axis=0).astype(np.float32)
    origin_normalized = normalize_ijk(
        origin_ijk[None],
        subject.full_shape,
    )[0]
    origin_world_mm = ijk_to_world_mm(
        origin_ijk[None],
        subject.affine,
    )[0]
    geometry = SubjectGeometry(
        full_shape=tuple(int(value) for value in subject.full_shape),
        encoder_input_shape=tuple(
            int(value) for value in subject.head_volume.shape[-3:]
        ),
        affine_ijk_to_ras_mm=_float_tensor(
            np.asarray(subject.affine)[None],
            device=resolved_device,
        ),
        inverse_affine_ras_mm_to_ijk=_float_tensor(
            np.asarray(subject.inverse_affine)[None],
            device=resolved_device,
        ),
        anatomy_origin_ijk=_float_tensor(
            origin_ijk[None],
            device=resolved_device,
        ),
        anatomy_origin_normalized=_float_tensor(
            origin_normalized[None],
            device=resolved_device,
        ),
        anatomy_origin_world_mm=_float_tensor(
            origin_world_mm[None],
            device=resolved_device,
        ),
    )
    geometry.validate()
    return geometry


def build_query_bundle(
    subject: ModelInputSubject,
    row_indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device | str,
    geometry: SubjectGeometry | None = None,
) -> QueryCoordinateBundle:
    """Build a query bundle whose row identities follow ``subject.gmwm_ijk``."""

    resolved_device = _as_device(device)
    rows = _row_indices(row_indices, row_count=len(subject.gmwm_ijk))
    ijk = np.asarray(subject.gmwm_ijk[rows], dtype=np.float32)
    bundle = QueryCoordinateBundle(
        gmwm_row_index=torch.from_numpy(rows).unsqueeze(0).to(resolved_device),
        ijk=_float_tensor(ijk[None], device=resolved_device),
        normalized_coord=_float_tensor(
            normalize_ijk(ijk, subject.full_shape)[None],
            device=resolved_device,
        ),
        world_ras_mm=_float_tensor(
            ijk_to_world_mm(ijk, subject.affine)[None],
            device=resolved_device,
        ),
        features_1mm=_float_tensor(
            build_query_features_1mm(subject, ijk)[None],
            device=resolved_device,
        ),
    )
    resolved_geometry = geometry or build_subject_geometry(
        subject,
        device=resolved_device,
    )
    bundle.validate(resolved_geometry)
    return bundle


def build_electrode_pair_inputs(
    subject: ModelInputSubject,
    source_world_ras_mm: np.ndarray | Sequence[Sequence[float]],
    *,
    device: torch.device | str,
    source_electrode_ids: Sequence[int] | np.ndarray | None = None,
    reference_world_ras_mm: np.ndarray | Sequence[float] | None = None,
    reference_electrode_id: int | None = None,
    geometry: SubjectGeometry | None = None,
) -> ElectrodePairInputs:
    """Build a stable unique electrode bank and ``source -> reference`` pair indices.

    Repeated integer identities are encoded once. The same identity must carry
    identical world-RAS coordinates, preventing one ID from silently referring
    to two physical electrodes.
    """

    resolved_device = _as_device(device)
    sources = _world_points(source_world_ras_mm)
    if source_electrode_ids is None:
        source_ids = np.arange(len(sources), dtype=np.int64)
    else:
        raw_ids = np.asarray(source_electrode_ids)
        if (
            raw_ids.ndim != 1
            or len(raw_ids) != len(sources)
            or raw_ids.dtype == np.bool_
            or not np.issubdtype(raw_ids.dtype, np.integer)
        ):
            raise TypeError(
                "source_electrode_ids must be a 1D integer sequence matching the source count"
            )
        source_ids = np.ascontiguousarray(raw_ids, dtype=np.int64)
    if np.any(source_ids < 0):
        raise ValueError("source_electrode_ids must be non-negative")

    reference = np.asarray(
        subject.reference_coord_mm
        if reference_world_ras_mm is None
        else reference_world_ras_mm,
        dtype=np.float32,
    )
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference_world_ras_mm must be a finite coordinate with shape (3,)"
        )
    if reference_electrode_id is None:
        largest_source_id = int(source_ids.max(initial=-1))
        reference_id = max(largest_source_id, len(subject.source_names) - 1) + 1
    else:
        if isinstance(reference_electrode_id, (bool, np.bool_)) or not isinstance(
            reference_electrode_id,
            (int, np.integer),
        ):
            raise TypeError("reference_electrode_id must be a non-negative integer")
        reference_id = int(reference_electrode_id)
        if reference_id < 0:
            raise ValueError("reference_electrode_id must be a non-negative integer")

    unique_ids: list[int] = []
    unique_world: list[np.ndarray] = []
    bank_index_by_id: dict[int, int] = {}

    def insert(electrode_id: int, world_mm: np.ndarray) -> int:
        existing = bank_index_by_id.get(electrode_id)
        if existing is not None:
            if not np.allclose(
                unique_world[existing],
                world_mm,
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(
                    f"electrode_id={electrode_id} refers to multiple world-RAS coordinates"
                )
            return existing
        bank_index_by_id[electrode_id] = len(unique_ids)
        unique_ids.append(electrode_id)
        unique_world.append(np.asarray(world_mm, dtype=np.float32))
        return len(unique_ids) - 1

    source_bank_indices = [
        insert(int(electrode_id), world_mm)
        for electrode_id, world_mm in zip(source_ids.tolist(), sources, strict=True)
    ]
    reference_bank_index = insert(reference_id, reference)
    unique_world_array = np.stack(unique_world).astype(np.float32, copy=False)
    unique_normalized = world_to_normalized_ijk(
        unique_world_array,
        subject.inverse_affine,
        subject.full_shape,
    )
    electrodes = ElectrodeCoordinateBundle(
        electrode_id=torch.tensor(
            unique_ids,
            dtype=torch.long,
            device=resolved_device,
        ).unsqueeze(0),
        normalized_coord=_float_tensor(
            unique_normalized[None],
            device=resolved_device,
        ),
        world_ras_mm=_float_tensor(
            unique_world_array[None],
            device=resolved_device,
        ),
    )
    pair_index = torch.tensor(
        [
            [source_bank_index, reference_bank_index]
            for source_bank_index in source_bank_indices
        ],
        dtype=torch.long,
        device=resolved_device,
    ).unsqueeze(0)
    resolved_geometry = geometry or build_subject_geometry(
        subject,
        device=resolved_device,
    )
    electrodes.validate(resolved_geometry)
    return ElectrodePairInputs(electrodes=electrodes, pair_index=pair_index)


__all__ = [
    "ElectrodePairInputs",
    "ModelInputSubject",
    "PreparedModelInputSubject",
    "build_electrode_pair_inputs",
    "build_model_head_inputs",
    "build_prepared_model_input_subject",
    "build_query_bundle",
    "build_query_features_1mm",
    "build_subject_geometry",
    "normalize_ijk",
    "normalize_t1_for_model",
    "world_to_normalized_ijk",
]
