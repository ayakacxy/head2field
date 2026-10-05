"""Head2Field v1: a continuous neural field for subject-specific tES lead fields.

The model predicts electric fields directly in world-RAS coordinates rather than
predicting potentials followed by finite differences. A single-electrode response
is denoted ``G(x, e)``; each bipolar lead is defined exactly as

``E(x; source -> reference) = G(x, source) - G(x, reference)``.

Tensor notation throughout this module: ``B`` subjects, ``U`` unique electrodes,
``P`` source/reference pairs, ``Q`` query points, and ``C`` feature channels.
Coordinate bundles keep voxel i/j/k, normalized i/j/k, and world-RAS millimeters
explicit. Spatial tensors use ``[B,C,I,J,K]``; sampling reverses the final
coordinate axis for PyTorch's W/H/D grid convention.

The bundled pretrained model is created by ``create_head2field_v1`` in
``_spec.py``. Lower-level components are grouped as coordinate bundles, sampling
helpers, encoder/mixer blocks, and the assembled ``Head2FieldV1`` network.

The model only produces fields. Circulation, flux, GM-WM interface residuals, and
FEM-calibrated thresholds belong to the training-loss layer. PDE losses must not
be applied independently to unidentifiable single-electrode responses ``G``;
physical losses apply only to the final pair field.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

HEAD2FIELD_V1_MODEL_VERSION = "v1"
HEAD2FIELD_V1_DEFAULT_ENCODER_INPUT_SHAPE = (128, 128, 128)
HEAD2FIELD_V1_CANONICAL_GRID_EXTENT_MM = 255.0
HEAD2FIELD_V1_COORDINATE_FRAME_ID = "head2field_v1_canonical_ras_1mm"
HEAD2FIELD_V1_CORE_ARCHITECTURE_ID = "head2field_v1_response_bank"
HEAD2FIELD_V1_LOCAL_ARCHITECTURE_ID = "head2field_v1_local_anatomy_mixer"
HEAD2FIELD_V1_GLOBAL_ARCHITECTURE_ID = "head2field_v1_global_anatomy_mixer"
ModelVariant = Literal["C", "P", "G_LOCAL", "G", "GP"]
MODEL_VARIANTS: tuple[ModelVariant, ...] = ("C", "P", "G_LOCAL", "G", "GP")


def architecture_id_for_variant(variant_id: str) -> str:
    """Return the architecture identifier for a variant, rejecting unknown spellings.

    P and C share the same network, as do GP and G. Their difference is whether
    the trainer backpropagates through the deterministic physics branch, not a
    separate hidden network.
    """

    if variant_id in ("C", "P"):
        return HEAD2FIELD_V1_CORE_ARCHITECTURE_ID
    if variant_id == "G_LOCAL":
        return HEAD2FIELD_V1_LOCAL_ARCHITECTURE_ID
    if variant_id in ("G", "GP"):
        return HEAD2FIELD_V1_GLOBAL_ARCHITECTURE_ID
    raise ValueError(
        f"Unknown v1 variant_id={variant_id!r}; allowed values: {', '.join(MODEL_VARIANTS)}"
    )


def _variant_uses_mixer(variant_id: str) -> bool:
    architecture_id_for_variant(variant_id)
    return variant_id in ("G_LOCAL", "G", "GP")


def _group_count(channels: int, maximum: int = 8) -> int:
    """Find the largest GroupNorm group count not exceeding ``maximum`` that divides channels."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _require_floating(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point; got dtype={tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _as_batched_affine(
    affine: torch.Tensor,
    batch: int,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize a ``[4,4]`` or ``[B,4,4]`` affine to batched form."""

    _require_floating(name, affine)
    if affine.shape == (4, 4):
        return affine.unsqueeze(0).expand(batch, -1, -1)
    if affine.shape == (batch, 4, 4):
        return affine
    raise ValueError(
        f"{name} must have shape [4,4] or [B,4,4]; got {tuple(affine.shape)}"
    )


def _autocast_disabled(device: torch.device):
    """Disable AMP for geometry and affine calculations to preserve coordinate validation.

    PyTorch autocast can reduce matmul/einsum to FP16/BF16. Comparing those results
    with authoritative FP32 coordinates using ``allclose`` can yield false failures
    or dtype errors. CPU and CUDA are supported; backends without an autocast
    context use a null context and retain the original dtype.
    """

    if device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _apply_affine(affine: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Apply a batched 4x4 affine to ``[B,N,3]`` coordinates."""

    batch = coordinates.shape[0]
    matrix = _as_batched_affine(affine, batch, name="affine")
    if matrix.device != coordinates.device or matrix.dtype != coordinates.dtype:
        raise ValueError("affine and coordinates must have the same device and dtype")
    with _autocast_disabled(coordinates.device):
        homogeneous = torch.cat(
            [coordinates, torch.ones_like(coordinates[..., :1])],
            dim=-1,
        )
        return torch.einsum("bij,bnj->bni", matrix, homogeneous)[..., :3]


@dataclass(frozen=True)
class SubjectGeometry:
    """Authoritative coordinate-transform contract shared by one or more subjects.

    ``full_shape`` describes canonical 1 mm data, usually 256 cubed.
    ``encoder_input_shape`` is the resampled 3D encoder shape, 128 cubed by default.
    Normalized query and electrode coordinates always refer to the full canonical
    extent in ``full_shape``. The encoder preserves these physical bounds, so the
    coordinates can also be used directly with FPN ``grid_sample``.
    """

    full_shape: tuple[int, int, int]
    encoder_input_shape: tuple[int, int, int]
    affine_ijk_to_ras_mm: torch.Tensor
    inverse_affine_ras_mm_to_ijk: torch.Tensor
    anatomy_origin_ijk: torch.Tensor
    anatomy_origin_normalized: torch.Tensor
    anatomy_origin_world_mm: torch.Tensor
    orientation: str = "RAS"

    @property
    def batch_size(self) -> int:
        if self.anatomy_origin_ijk.ndim != 2:
            raise ValueError("anatomy_origin_ijk must have shape [B,3]")
        return int(self.anatomy_origin_ijk.shape[0])

    def normalized_to_ijk(self, normalized: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(
            [max(length - 1, 1) for length in self.full_shape],
            device=normalized.device,
            dtype=normalized.dtype,
        )
        return (normalized + 1.0) * (scale / 2.0)

    def ijk_to_normalized(self, ijk: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(
            [max(length - 1, 1) for length in self.full_shape],
            device=ijk.device,
            dtype=ijk.dtype,
        )
        return 2.0 * ijk / scale - 1.0

    def ijk_to_world_mm(self, ijk: torch.Tensor) -> torch.Tensor:
        return _apply_affine(self.affine_ijk_to_ras_mm, ijk)

    @torch.compiler.disable
    def validate(self, *, atol: float = 1e-4, rtol: float = 1e-5) -> None:
        """Validate shapes, orientation, inverse affines, and all three anatomy-origin forms."""

        if self.orientation.upper() != "RAS":
            raise ValueError(
                "v1 output components use world-RAS; SubjectGeometry.orientation must be 'RAS'"
            )
        for name, shape in (
            ("full_shape", self.full_shape),
            ("encoder_input_shape", self.encoder_input_shape),
        ):
            if len(shape) != 3 or any(int(value) <= 0 for value in shape):
                raise ValueError(
                    f"{name} must contain three positive integers; got {shape}"
                )

        batch = self.batch_size
        origins = {
            "anatomy_origin_ijk": self.anatomy_origin_ijk,
            "anatomy_origin_normalized": self.anatomy_origin_normalized,
            "anatomy_origin_world_mm": self.anatomy_origin_world_mm,
        }
        for name, value in origins.items():
            _require_floating(name, value)
            if value.shape != (batch, 3):
                raise ValueError(
                    f"{name} must have shape [B,3]; got {tuple(value.shape)}"
                )

        affine = _as_batched_affine(
            self.affine_ijk_to_ras_mm,
            batch,
            name="affine_ijk_to_ras_mm",
        )
        inverse = _as_batched_affine(
            self.inverse_affine_ras_mm_to_ijk,
            batch,
            name="inverse_affine_ras_mm_to_ijk",
        )
        if affine.device != self.anatomy_origin_ijk.device:
            raise ValueError(
                "SubjectGeometry affine and anatomy origin must be on the same device"
            )
        if (
            affine.dtype != self.anatomy_origin_ijk.dtype
            or inverse.dtype != affine.dtype
        ):
            raise ValueError(
                "SubjectGeometry affine, inverse affine, and origin must have the same dtype"
            )
        with _autocast_disabled(affine.device):
            identity = torch.eye(4, device=affine.device, dtype=affine.dtype)[None]
            affine_inverse_product = affine @ inverse
            expected_normalized = self.ijk_to_normalized(self.anatomy_origin_ijk)
            expected_world = self.ijk_to_world_mm(self.anatomy_origin_ijk[:, None])[
                :, 0
            ]
        if not torch.allclose(
            affine_inverse_product,
            identity,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                "affine_ijk_to_ras_mm and inverse_affine must be inverse matrices"
            )

        if not torch.allclose(
            self.anatomy_origin_normalized,
            expected_normalized,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                "anatomy_origin_normalized is inconsistent with full_shape and origin_ijk"
            )
        if not torch.allclose(
            self.anatomy_origin_world_mm,
            expected_world,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                "anatomy_origin_world_mm is inconsistent with affine and origin_ijk"
            )


@dataclass(frozen=True)
class QueryCoordinateBundle:
    """Keep query identity and all three coordinate representations together.

    ``gmwm_row_index`` is int64 ``[B,Q]``; coordinate fields are ``[B,Q,3]``.
    ``features_1mm`` contains ``[B,Q,12]`` tissue, conductivity, and T1 features
    sampled on the original grid. Normalized coordinates refer to ``full_shape``.
    """

    gmwm_row_index: torch.Tensor
    ijk: torch.Tensor
    normalized_coord: torch.Tensor
    world_ras_mm: torch.Tensor
    features_1mm: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return int(self.normalized_coord.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.normalized_coord.shape[1])

    @torch.compiler.disable
    def validate(
        self,
        geometry: SubjectGeometry,
        *,
        require_unique_rows: bool = False,
        check_coordinate_roundtrip: bool = True,
        atol: float = 2e-3,
        rtol: float = 1e-5,
    ) -> None:
        if self.normalized_coord.ndim != 3 or self.normalized_coord.shape[-1] != 3:
            raise ValueError("query normalized_coord must have shape [B,Q,3]")
        batch, points, _ = self.normalized_coord.shape
        if batch != geometry.batch_size:
            raise ValueError("query and SubjectGeometry batch sizes must match")
        if self.gmwm_row_index.shape != (batch, points):
            raise ValueError("gmwm_row_index must have shape [B,Q]")
        if self.gmwm_row_index.dtype != torch.long:
            raise TypeError("gmwm_row_index must use torch.long")
        if bool((self.gmwm_row_index < 0).any()):
            raise ValueError("gmwm_row_index must be non-negative")
        for name, tensor in (
            ("query.ijk", self.ijk),
            ("query.normalized_coord", self.normalized_coord),
            ("query.world_ras_mm", self.world_ras_mm),
        ):
            _require_floating(name, tensor)
            if tensor.shape != (batch, points, 3):
                raise ValueError(f"{name} must have shape [B,Q,3]")
        if self.features_1mm is not None:
            _require_floating("query.features_1mm", self.features_1mm)
            if self.features_1mm.shape[:2] != (batch, points):
                raise ValueError(
                    "query.features_1mm must have leading dimensions [B,Q]"
                )
        if bool((self.normalized_coord.abs() > 1.0001).any()):
            raise ValueError("query normalized_coord must be in [-1,1]")

        if require_unique_rows:
            for row in self.gmwm_row_index:
                if torch.unique(row).numel() != points:
                    raise ValueError(
                        "physics query union must use stable, unique gmwm_row_index values"
                    )

        if check_coordinate_roundtrip:
            with _autocast_disabled(self.normalized_coord.device):
                expected_ijk = geometry.normalized_to_ijk(self.normalized_coord)
                expected_world = geometry.ijk_to_world_mm(self.ijk)
            if not torch.allclose(self.ijk, expected_ijk, rtol=rtol, atol=atol):
                raise ValueError(
                    "query ijk is inconsistent with normalized_coord and full_shape"
                )
            if not torch.allclose(
                self.world_ras_mm,
                expected_world,
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError(
                    "query world_ras_mm is inconsistent with affine and ijk"
                )


@dataclass(frozen=True)
class ElectrodeCoordinateBundle:
    """Stably deduplicated electrode bank; ``electrode_id`` identifies electrodes, not roles."""

    electrode_id: torch.Tensor
    normalized_coord: torch.Tensor
    world_ras_mm: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.normalized_coord.shape[0])

    @property
    def electrode_count(self) -> int:
        return int(self.normalized_coord.shape[1])

    @torch.compiler.disable
    def validate(
        self,
        geometry: SubjectGeometry,
        *,
        require_unique_ids: bool = True,
        check_coordinate_roundtrip: bool = True,
        atol: float = 2e-3,
        rtol: float = 1e-5,
    ) -> None:
        if self.normalized_coord.ndim != 3 or self.normalized_coord.shape[-1] != 3:
            raise ValueError("electrode normalized_coord must have shape [B,U,3]")
        batch, electrodes, _ = self.normalized_coord.shape
        if batch != geometry.batch_size:
            raise ValueError("electrode and SubjectGeometry batch sizes must match")
        if electrodes <= 0:
            raise ValueError(
                "unique electrode bank must contain at least one electrode"
            )
        if self.electrode_id.shape != (batch, electrodes):
            raise ValueError("electrode_id must have shape [B,U]")
        if self.electrode_id.dtype != torch.long:
            raise TypeError("electrode_id must use torch.long")
        for name, tensor in (
            ("electrode.normalized_coord", self.normalized_coord),
            ("electrode.world_ras_mm", self.world_ras_mm),
        ):
            _require_floating(name, tensor)
            if tensor.shape != (batch, electrodes, 3):
                raise ValueError(f"{name} must have shape [B,U,3]")
        if bool((self.normalized_coord.abs() > 1.0001).any()):
            raise ValueError("electrode normalized_coord must be in [-1,1]")
        if require_unique_ids:
            for ids in self.electrode_id:
                if torch.unique(ids).numel() != electrodes:
                    raise ValueError(
                        "encode_unique_electrodes received duplicate electrode_id values; "
                        "deduplicate in first-occurrence order and reference the bank through pair_index"
                    )
        if check_coordinate_roundtrip:
            with _autocast_disabled(self.normalized_coord.device):
                ijk = geometry.normalized_to_ijk(self.normalized_coord)
                expected_world = geometry.ijk_to_world_mm(ijk)
            if not torch.allclose(
                self.world_ras_mm,
                expected_world,
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError(
                    "electrode world_ras_mm is inconsistent with normalized_coord, full_shape, and affine"
                )


@dataclass(frozen=True)
class AbsoluteCoordinateAugmentation:
    """One random realization shared by the absolute query and electrode coordinate branches."""

    branch_mask: torch.Tensor
    coordinate_offset_normalized: torch.Tensor

    @torch.compiler.disable
    def validate(self, batch: int, device: torch.device) -> None:
        if self.branch_mask.shape != (batch, 1):
            raise ValueError("absolute branch_mask must have shape [B,1]")
        if self.coordinate_offset_normalized.shape != (batch, 3):
            raise ValueError(
                "absolute coordinate_offset_normalized must have shape [B,3]"
            )
        if (
            self.branch_mask.device != device
            or self.coordinate_offset_normalized.device != device
        ):
            raise ValueError(
                "absolute augmentation and augmented coordinates must be on the same device"
            )
        _require_floating("absolute branch_mask", self.branch_mask)
        _require_floating(
            "absolute coordinate_offset_normalized",
            self.coordinate_offset_normalized,
        )


@dataclass(frozen=True)
class HeadFeatureBundle:
    """Bundle of v1 FPN features, encoder-resolution head volume, and geometry."""

    pyramid: tuple[torch.Tensor, ...]
    head_volume: torch.Tensor
    subject_geometry: SubjectGeometry


@dataclass(frozen=True)
class QueryContext:
    """Encoded ``[B,Q,query_dim]`` tokens with normalized and world coordinates."""

    token: torch.Tensor
    normalized_coord: torch.Tensor
    world_ras_mm: torch.Tensor


@dataclass(frozen=True)
class ElectrodeContext:
    """Encoded ``[B,U,electrode_dim]`` tokens and their stable electrode IDs."""

    token: torch.Tensor
    electrode_id: torch.Tensor
    normalized_coord: torch.Tensor
    world_ras_mm: torch.Tensor


@dataclass(frozen=True)
class BranchOutput:
    """Shared ``[B,U,Q,3]`` responses and ``[B,P,Q,3]`` source-minus-reference fields.

    Only ``pair_field`` has the bipolar field/current contract. Individual
    ``response_bank`` entries share an unidentifiable additive gauge.
    """

    response_bank: torch.Tensor
    pair_field: torch.Tensor


@dataclass(frozen=True)
class DataPhysicsOutput:
    """Supervised and deterministic physics outputs sharing one 3D encoding."""

    data: BranchOutput
    physics: BranchOutput


def _stable_unique_positions(ids: torch.Tensor) -> tuple[list[int], list[int]]:
    """Return first-occurrence positions and inverse indices for a 1D integer ID array."""

    positions: list[int] = []
    inverse: list[int] = []
    seen: dict[int, int] = {}
    for position, raw_id in enumerate(ids.detach().cpu().tolist()):
        electrode_id = int(raw_id)
        if electrode_id not in seen:
            seen[electrode_id] = len(positions)
            positions.append(position)
        inverse.append(seen[electrode_id])
    return positions, inverse


def stable_unique_electrodes(
    candidates: ElectrodeCoordinateBundle,
    *,
    atol: float = 1e-6,
) -> tuple[ElectrodeCoordinateBundle, torch.Tensor]:
    """Deduplicate electrode_id stably in the data layer and return input-to-bank indices.

    Samples with different unique-electrode counts must be batched separately by
    the sampler or padded explicitly. This function never silently introduces a
    padding electrode that could be mistaken for a physical electrode.
    """

    if candidates.electrode_id.ndim != 2:
        raise ValueError("candidate electrode_id must have shape [B,N]")
    batch, count = candidates.electrode_id.shape
    if candidates.electrode_id.dtype != torch.long:
        raise TypeError("candidate electrode_id must use torch.long")
    if candidates.normalized_coord.shape != (batch, count, 3):
        raise ValueError("candidate normalized_coord must have shape [B,N,3]")
    if candidates.world_ras_mm.shape != (batch, count, 3):
        raise ValueError("candidate world_ras_mm must have shape [B,N,3]")

    all_positions: list[list[int]] = []
    all_inverse: list[list[int]] = []
    unique_counts: list[int] = []
    for batch_index in range(batch):
        positions, inverse = _stable_unique_positions(
            candidates.electrode_id[batch_index]
        )
        all_positions.append(positions)
        all_inverse.append(inverse)
        unique_counts.append(len(positions))

        first_position: dict[int, int] = {}
        for position, raw_id in enumerate(
            candidates.electrode_id[batch_index].detach().cpu().tolist()
        ):
            electrode_id = int(raw_id)
            if electrode_id not in first_position:
                first_position[electrode_id] = position
                continue
            first = first_position[electrode_id]
            if not torch.allclose(
                candidates.normalized_coord[batch_index, position],
                candidates.normalized_coord[batch_index, first],
                rtol=0.0,
                atol=atol,
            ) or not torch.allclose(
                candidates.world_ras_mm[batch_index, position],
                candidates.world_ras_mm[batch_index, first],
                rtol=0.0,
                atol=atol,
            ):
                raise ValueError(
                    f"electrode_id={electrode_id} has inconsistent coordinates"
                )

    if len(set(unique_counts)) != 1:
        raise ValueError(
            "subjects have different unique-electrode counts; batch by U or provide an explicit sampler mask/padding contract"
        )

    position_tensor = torch.tensor(
        all_positions,
        device=candidates.electrode_id.device,
        dtype=torch.long,
    )
    batch_index = torch.arange(batch, device=position_tensor.device)[:, None]
    unique = ElectrodeCoordinateBundle(
        electrode_id=candidates.electrode_id[batch_index, position_tensor],
        normalized_coord=candidates.normalized_coord[batch_index, position_tensor],
        world_ras_mm=candidates.world_ras_mm[batch_index, position_tensor],
    )
    inverse_tensor = torch.tensor(
        all_inverse,
        device=candidates.electrode_id.device,
        dtype=torch.long,
    )
    return unique, inverse_tensor


class FourierEncoding3d(nn.Module):
    """Fixed-frequency 3D Fourier encoding; frequency buffers are excluded from checkpoints."""

    def __init__(self, bands: int) -> None:
        super().__init__()
        if bands <= 0:
            raise ValueError("Fourier bands must be positive")
        frequencies = math.pi * (2.0 ** torch.arange(bands, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_dim = 3 * (1 + 2 * bands)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        angles = coordinates.unsqueeze(-1) * self.frequencies
        return torch.cat(
            [
                coordinates,
                torch.sin(angles).flatten(-2),
                torch.cos(angles).flatten(-2),
            ],
            dim=-1,
        )


def sample_volume_at_points(
    volume: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Sample a 3D field at normalized NIfTI i/j/k coordinates, returning ``[B,N,C]``.

    PyTorch ``grid_sample`` interprets the final grid dimension as W/H/D (x/y/z),
    requiring reversal from i/j/k to k/j/i. Queries and electrodes share this
    axis convention to keep the two paths consistent.
    """

    if volume.ndim != 5 or coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("volume must have shape [B,C,D,H,W] and coordinates [B,N,3]")
    if volume.shape[0] != coordinates.shape[0]:
        raise ValueError("volume and coordinates batch sizes must match")
    grid = coordinates[..., [2, 1, 0]].to(dtype=volume.dtype)
    grid = grid.unsqueeze(1).unsqueeze(1)
    sampled = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(2).squeeze(2).transpose(1, 2)


def _masked_channel_statistics(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = mask.sum(dim=(2, 3, 4)).clamp_min(1e-6)
    mean = (features * mask).sum(dim=(2, 3, 4)) / denominator
    centered = features - mean[:, :, None, None, None]
    variance = (centered.square() * mask).sum(dim=(2, 3, 4)) / denominator
    return mean, torch.sqrt(variance.clamp_min(1e-8))


def build_anatomy_statistics(head_volume: torch.Tensor) -> torch.Tensor:
    """Build 14-dimensional anatomy statistics invariant to canvas-only translation.

    The features are ten tissue fractions, one total foreground fraction, and
    three axial standard deviations centered on the foreground centroid.
    """

    if head_volume.ndim != 5 or head_volume.shape[1] < 10:
        raise ValueError(
            "head_volume must have shape [B,C,D,H,W] with at least 10 tissue channels"
        )
    tissue = head_volume[:, :10].float()
    foreground = tissue.sum(dim=1).clamp(0.0, 1.0)
    tissue_fractions = tissue.mean(dim=(2, 3, 4))
    foreground_fraction = foreground.mean(dim=(1, 2, 3))[:, None]

    deviations: list[torch.Tensor] = []
    for axis, length in enumerate(foreground.shape[1:]):
        reduce_dims = tuple(index + 1 for index in range(3) if index != axis)
        marginal = foreground.sum(dim=reduce_dims)
        coordinates = torch.linspace(
            -1.0,
            1.0,
            steps=length,
            device=head_volume.device,
            dtype=torch.float32,
        )
        mass = marginal.sum(dim=1).clamp_min(1e-6)
        centroid = (marginal * coordinates[None]).sum(dim=1) / mass
        variance = (marginal * (coordinates[None] - centroid[:, None]).square()).sum(
            dim=1
        ) / mass
        deviations.append(torch.sqrt(variance.clamp_min(1e-8)))

    return torch.cat(
        [
            tissue_fractions,
            foreground_fraction,
            torch.stack(deviations, dim=1),
        ],
        dim=1,
    )


def _generator_aware_dropout(
    x: torch.Tensor,
    probability: float,
    *,
    enabled: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Implement inverted dropout with an explicitly supplied independent RNG.

    ``torch.nn.Dropout`` does not accept a generator. Using the global RNG would
    allow a physics call or diagnostic sample to shift the data-branch sequence,
    breaking exact resume and paired experiments. With ``enabled=False`` this
    helper neither allocates a mask nor consumes random numbers.
    """

    if not enabled or probability == 0.0:
        return x
    keep_probability = 1.0 - probability
    random_values = torch.rand(
        x.shape,
        device=x.device,
        dtype=torch.float32,
        generator=generator,
    )
    mask = (random_values < keep_probability).to(dtype=x.dtype)
    return x * mask / keep_probability


class ControlledDropout(nn.Module):
    """Dropout supporting a named generator and an explicit ``deterministic`` switch."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must be in [0,1)")
        self.p = float(probability)

    def forward(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return _generator_aware_dropout(
            x,
            self.p,
            enabled=self.training and not deterministic,
            generator=generator,
        )


class ControlledSequential(nn.Sequential):
    """Preserve frozen v1 numeric layer names while forwarding dropout controls."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        for module in self:
            if isinstance(module, ControlledDropout):
                x = module(x, deterministic=deterministic, generator=generator)
            else:
                x = module(x)
        return x


class ConvNormActivation3d(nn.Module):
    """3D convolution followed by GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BottleneckResidual3d(nn.Module):
    """Efficient residual block with a 3x3x3 spatial convolution on narrow channels."""

    def __init__(self, channels: int, bottleneck_ratio: int = 4) -> None:
        super().__init__()
        inner_channels = max(channels // bottleneck_ratio, 8)
        self.reduce = ConvNormActivation3d(channels, inner_channels, kernel_size=1)
        self.spatial = ConvNormActivation3d(
            inner_channels,
            inner_channels,
            kernel_size=3,
            padding=1,
        )
        self.expand = nn.Sequential(
            nn.Conv3d(inner_channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.expand(self.spatial(self.reduce(x))))


class AnatomyPyramidEncoder(nn.Module):
    """3D encoder with four levels by default and one bottleneck residual block per level."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        levels: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0:
            raise ValueError("input_channels and base_channels must be positive")
        if levels not in (3, 4):
            raise ValueError("v1 encoder supports only 3-level or 4-level pyramids")
        self.activation_checkpointing = bool(activation_checkpointing)
        self.output_channels = [base_channels * (2**index) for index in range(levels)]
        self.stem = nn.Sequential(
            ConvNormActivation3d(
                input_channels,
                self.output_channels[0],
                kernel_size=3,
                padding=1,
            ),
            BottleneckResidual3d(self.output_channels[0]),
        )
        self.downsample_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(
            self.output_channels[:-1],
            self.output_channels[1:],
            strict=True,
        ):
            self.downsample_blocks.append(
                nn.Sequential(
                    ConvNormActivation3d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    BottleneckResidual3d(out_channels),
                )
            )

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, head_volume: torch.Tensor) -> list[torch.Tensor]:
        current = self._run(self.stem, head_volume)
        features = [current]
        for block in self.downsample_blocks:
            current = self._run(block, current)
            features.append(current)
        return features


class ResidualSwiGLUBlock(nn.Module):
    """Pre-norm SwiGLU residual block for the query trunk."""

    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner_width = width * expansion
        self.norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, inner_width * 2)
        self.dropout = ControlledDropout(dropout)
        self.output_projection = nn.Linear(inner_width, width)

    def forward(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        value, gate = self.input_projection(self.norm(x)).chunk(2, dim=-1)
        update = value * F.silu(gate)
        update = self.dropout(
            update,
            deterministic=deterministic,
            generator=generator,
        )
        return x + self.output_projection(update)


class ElectrodeConditionedResidualBlock(nn.Module):
    """Query-electrode response block modulated by a unique-electrode token."""

    def __init__(
        self,
        width: int,
        electrode_dim: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(electrode_dim, width * 2)
        self.input_projection = nn.Linear(width, bottleneck_dim)
        self.dropout = ControlledDropout(dropout)
        self.output_projection = nn.Linear(bottleneck_dim, width)
        # Start near identity modulation to prevent random FiLM from amplifying initial activations.
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        electrode_token: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        scale, shift = self.modulation(electrode_token).chunk(2, dim=-1)
        normalized = self.norm(hidden)
        modulated = (
            normalized * (1.0 + 0.1 * torch.tanh(scale[:, :, None])) + shift[:, :, None]
        )
        update = F.silu(self.input_projection(modulated))
        update = self.dropout(
            update,
            deterministic=deterministic,
            generator=generator,
        )
        return hidden + self.output_projection(update)


def _to_axis_sequences(
    x: torch.Tensor,
    axis: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    """Move a spatial axis to the sequence dimension, returning ``[groups,L,C]``."""

    if axis == 0:  # I / tensor D
        permuted = x.permute(0, 3, 4, 2, 1)  # Input axes are B,H,W,D,C.
    elif axis == 1:  # J / tensor H
        permuted = x.permute(0, 2, 4, 3, 1)  # Input axes are B,D,W,H,C.
    elif axis == 2:  # K / tensor W
        permuted = x.permute(0, 2, 3, 4, 1)  # Input axes are B,D,H,W,C.
    else:
        raise ValueError("axial attention axis must be 0, 1, or 2")
    shape = tuple(int(value) for value in permuted.shape)
    return permuted.reshape(-1, shape[-2], shape[-1]), shape


def _from_axis_sequences(
    sequences: torch.Tensor,
    shape: tuple[int, int, int, int, int],
    axis: int,
) -> torch.Tensor:
    restored = sequences.reshape(shape)
    if axis == 0:
        return restored.permute(0, 4, 3, 1, 2)
    if axis == 1:
        return restored.permute(0, 4, 1, 3, 2)
    if axis == 2:
        return restored.permute(0, 4, 1, 2, 3)
    raise ValueError("axial attention axis must be 0, 1, or 2")


class AxialSelfAttention3d(nn.Module):
    """Pre-norm attention along one I/J/K axis with clipped relative-position bias.

    G and G_LOCAL use identical parameters. The local control changes only the
    fixed attention mask, retaining QKV, output-projection, and relative-bias
    parameters so global arrangement can be distinguished from parameter count.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        axis: int,
        *,
        scope: Literal["global", "local"],
        local_window: int,
        max_relative_distance: int,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads != 0:
            raise ValueError("axial attention width must be divisible by heads")
        if scope not in ("global", "local"):
            raise ValueError("attention scope must be global or local")
        if local_window <= 0 or local_window % 2 == 0:
            raise ValueError("local_window must be a positive odd integer")
        if max_relative_distance <= 0:
            raise ValueError("max_relative_distance must be positive")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        self.axis = int(axis)
        self.scope = scope
        self.local_window = int(local_window)
        self.max_relative_distance = int(max_relative_distance)

        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output_projection = nn.Linear(width, width)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(heads, 2 * max_relative_distance + 1)
        )

    def _bias_and_mask(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions = torch.arange(length, device=device)
        relative = positions[:, None] - positions[None, :]
        clipped = (
            relative.clamp(
                -self.max_relative_distance,
                self.max_relative_distance,
            )
            + self.max_relative_distance
        )
        bias = self.relative_position_bias[:, clipped].to(dtype=dtype)
        if self.scope == "global":
            return bias, None
        radius = self.local_window // 2
        return bias, relative.abs() > radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequences, original_shape = _to_axis_sequences(x, self.axis)
        residual = sequences
        normalized = self.norm(sequences)
        groups, length, _ = normalized.shape
        qkv = self.qkv(normalized).reshape(
            groups,
            length,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias, local_mask = self._bias_and_mask(length, x.device, scores.dtype)
        scores = scores + bias[None]
        if local_mask is not None:
            scores = scores.masked_fill(local_mask[None, None], float("-inf"))
        # Normalize softmax in FP32, then restore autocast dtype to avoid long-axis overflow.
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        mixed = torch.matmul(attention, value)
        mixed = mixed.transpose(1, 2).reshape(groups, length, self.width)
        output = residual + self.output_projection(mixed)
        return _from_axis_sequences(output, original_shape, self.axis)


class ChannelLayerNorm3d(nn.Module):
    """Apply LayerNorm to the channel dimension of each 3D voxel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class GlobalMixerBlock(nn.Module):
    """I/J/K axial attention, a depthwise local branch, and residual SwiGLU."""

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        scope: Literal["global", "local"],
        local_window: int,
        max_relative_distance: int,
        swiglu_expansion: int,
    ) -> None:
        super().__init__()
        if swiglu_expansion <= 0:
            raise ValueError("global mixer swiglu_expansion must be positive")
        self.axial_attention = nn.ModuleList(
            AxialSelfAttention3d(
                width,
                heads,
                axis,
                scope=scope,
                local_window=local_window,
                max_relative_distance=max_relative_distance,
            )
            for axis in range(3)
        )
        self.local_norm = ChannelLayerNorm3d(width)
        self.local_depthwise = nn.Conv3d(
            width,
            width,
            kernel_size=3,
            padding=1,
            groups=width,
            bias=True,
        )
        inner_width = width * swiglu_expansion
        self.channel_norm = nn.LayerNorm(width)
        self.channel_input_projection = nn.Linear(width, inner_width * 2)
        self.channel_output_projection = nn.Linear(inner_width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention in self.axial_attention:
            x = attention(x)
        x = x + self.local_depthwise(self.local_norm(x))
        channel_last = x.permute(0, 2, 3, 4, 1)
        value, gate = self.channel_input_projection(
            self.channel_norm(channel_last)
        ).chunk(2, dim=-1)
        update = self.channel_output_projection(value * F.silu(gate))
        return (channel_last + update).permute(0, 4, 1, 2, 3)


class CoarseGlobalAnatomyMixer(nn.Module):
    """Model full spatial anatomy only on the deepest coarse grid.

    Input/output projections and mixer weights use normal PyTorch initialization.
    Only the final scalar residual gate is zeroed, making the module initially
    an exact identity while allowing gate gradients on the first step. Do not
    also zero the output projection, which would eliminate gate gradients.
    """

    def __init__(
        self,
        input_channels: int,
        mixer_dim: int,
        blocks: int,
        heads: int,
        *,
        scope: Literal["global", "local"],
        local_window: int,
        max_relative_distance: int,
        swiglu_expansion: int,
    ) -> None:
        super().__init__()
        if mixer_dim <= 0 or blocks <= 0:
            raise ValueError("global mixer_dim and blocks must be positive")
        if mixer_dim % heads != 0:
            raise ValueError("global mixer_dim must be divisible by heads")
        self.scope = scope
        self.input_projection = nn.Conv3d(input_channels, mixer_dim, 1)
        self.blocks = nn.ModuleList(
            GlobalMixerBlock(
                mixer_dim,
                heads,
                scope=scope,
                local_window=local_window,
                max_relative_distance=max_relative_distance,
                swiglu_expansion=swiglu_expansion,
            )
            for _ in range(blocks)
        )
        self.output_projection = nn.Conv3d(mixer_dim, input_channels, 1)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, deepest_feature: torch.Tensor) -> torch.Tensor:
        mixed = self.input_projection(deepest_feature)
        for block in self.blocks:
            mixed = block(mixed)
        update = self.output_projection(mixed)
        return deepest_feature + self.residual_gate * update


class Head2FieldV1(nn.Module):
    """Encode anatomy, queries, and electrodes to shared world-RAS responses.

    Use ``create_head2field_v1()`` for the pretrained architecture. Explicit
    constructor settings support small synthetic models and ablation variants.
    ``forward_bundled`` evaluates one branch; ``forward_data_and_physics`` shares
    the anatomy encoder between stochastic supervision and deterministic physics.
    Cached inference calls the encode/decode methods through an inference session.
    """

    def __init__(
        self,
        input_channels: int = 12,
        base_channels: int = 16,
        use_1mm_features: bool = True,
        encoder_levels: int = 4,
        encoder_activation_checkpointing: bool = False,
        pyramid_dim: int = 32,
        query_dim: int = 256,
        query_blocks: int = 2,
        query_expansion: int = 2,
        electrode_dim: int = 192,
        decoder_blocks: int = 2,
        decoder_bottleneck_dim: int = 128,
        global_dim: int = 128,
        coordinate_fourier_bands: int = 6,
        context_dropout: float = 0.10,
        patch_radius_mm: float = 6.0,
        physical_coordinate_scale_mm: float = 128.0,
        absolute_fourier_bands: int = 3,
        absolute_aux_dim: int = 64,
        absolute_coordinate_dropout: float = 0.20,
        absolute_branch_dropout: float = 0.30,
        absolute_coordinate_jitter_probability: float = 0.50,
        absolute_coordinate_jitter_std_mm: float = 12.0,
        absolute_gate_initial_value: float = 0.12,
        *,
        variant_id: ModelVariant,
        global_mixer_dim: int = 96,
        global_mixer_blocks: int = 2,
        global_mixer_heads: int = 4,
        global_mixer_local_window: int = 5,
        global_mixer_max_relative_distance: int = 32,
        global_mixer_swiglu_expansion: int = 2,
    ) -> None:
        super().__init__()
        self.variant_id = str(variant_id)
        self.architecture_id = architecture_id_for_variant(self.variant_id)
        positive_integer_values = {
            "pyramid_dim": pyramid_dim,
            "query_dim": query_dim,
            "query_blocks": query_blocks,
            "query_expansion": query_expansion,
            "electrode_dim": electrode_dim,
            "decoder_blocks": decoder_blocks,
            "decoder_bottleneck_dim": decoder_bottleneck_dim,
            "global_dim": global_dim,
            "coordinate_fourier_bands": coordinate_fourier_bands,
            "absolute_fourier_bands": absolute_fourier_bands,
            "absolute_aux_dim": absolute_aux_dim,
            "global_mixer_dim": global_mixer_dim,
            "global_mixer_blocks": global_mixer_blocks,
            "global_mixer_heads": global_mixer_heads,
            "global_mixer_local_window": global_mixer_local_window,
            "global_mixer_max_relative_distance": global_mixer_max_relative_distance,
            "global_mixer_swiglu_expansion": global_mixer_swiglu_expansion,
        }
        invalid = [
            name for name, value in positive_integer_values.items() if value <= 0
        ]
        if invalid:
            raise ValueError(f"These parameters must be positive: {', '.join(invalid)}")
        if global_mixer_local_window % 2 == 0:
            raise ValueError("global_mixer_local_window must be odd")
        if global_mixer_dim % global_mixer_heads != 0:
            raise ValueError("global_mixer_dim must be divisible by global_mixer_heads")
        if not 0.0 <= context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0,1)")
        if patch_radius_mm <= 0.0:
            raise ValueError("patch_radius_mm must be positive")
        if physical_coordinate_scale_mm <= 0.0:
            raise ValueError("physical coordinate scales must be positive")
        if not 0.0 <= absolute_coordinate_dropout < 1.0:
            raise ValueError("absolute_coordinate_dropout must be in [0,1)")
        if not 0.0 <= absolute_branch_dropout < 1.0:
            raise ValueError("absolute_branch_dropout must be in [0,1)")
        if not 0.0 <= absolute_coordinate_jitter_probability <= 1.0:
            raise ValueError("absolute_coordinate_jitter_probability must be in [0,1]")
        if absolute_coordinate_jitter_std_mm < 0.0:
            raise ValueError("absolute_coordinate_jitter_std_mm must be non-negative")
        if not 0.0 < absolute_gate_initial_value < 1.0:
            raise ValueError("absolute_gate_initial_value must be in (0,1)")

        self.use_1mm_features = bool(use_1mm_features)
        self.encoder_levels = int(encoder_levels)
        self.pyramid_dim = int(pyramid_dim)
        self.query_dim = int(query_dim)
        self.electrode_dim = int(electrode_dim)
        self.global_dim = int(global_dim)
        self.context_dim = self.electrode_dim
        self.hidden_dim = self.query_dim
        self.film_dim = self.query_dim
        self.patch_radius_mm = float(patch_radius_mm)
        self.physical_coordinate_scale_mm = float(physical_coordinate_scale_mm)
        self.canonical_grid_extent_mm = HEAD2FIELD_V1_CANONICAL_GRID_EXTENT_MM
        self.absolute_fourier_bands = int(absolute_fourier_bands)
        self.absolute_aux_dim = int(absolute_aux_dim)
        self.absolute_coordinate_dropout = float(absolute_coordinate_dropout)
        self.absolute_branch_dropout = float(absolute_branch_dropout)
        self.absolute_coordinate_jitter_probability = float(
            absolute_coordinate_jitter_probability
        )
        self.absolute_coordinate_jitter_std_mm = float(
            absolute_coordinate_jitter_std_mm
        )
        self.global_mixer_dim = int(global_mixer_dim)
        self.global_mixer_blocks = int(global_mixer_blocks)
        self.global_mixer_heads = int(global_mixer_heads)
        self.global_mixer_local_window = int(global_mixer_local_window)
        self.global_mixer_max_relative_distance = int(
            global_mixer_max_relative_distance
        )
        self.global_mixer_swiglu_expansion = int(global_mixer_swiglu_expansion)
        # Store the full v1 architecture configuration for strict checkpoint/run-contract checks.
        self._model_config: dict[str, object] = {
            "input_channels": int(input_channels),
            "base_channels": int(base_channels),
            "use_1mm_features": bool(use_1mm_features),
            "encoder_levels": int(encoder_levels),
            "pyramid_dim": int(pyramid_dim),
            "query_dim": int(query_dim),
            "query_blocks": int(query_blocks),
            "query_expansion": int(query_expansion),
            "electrode_dim": int(electrode_dim),
            "decoder_blocks": int(decoder_blocks),
            "decoder_bottleneck_dim": int(decoder_bottleneck_dim),
            "global_dim": int(global_dim),
            "coordinate_fourier_bands": int(coordinate_fourier_bands),
            "context_dropout": float(context_dropout),
            "patch_radius_mm": float(patch_radius_mm),
            "physical_coordinate_scale_mm": float(physical_coordinate_scale_mm),
            "absolute_fourier_bands": int(absolute_fourier_bands),
            "absolute_aux_dim": int(absolute_aux_dim),
            "absolute_coordinate_dropout": float(absolute_coordinate_dropout),
            "absolute_branch_dropout": float(absolute_branch_dropout),
            "absolute_coordinate_jitter_probability": float(
                absolute_coordinate_jitter_probability
            ),
            "absolute_coordinate_jitter_std_mm": float(
                absolute_coordinate_jitter_std_mm
            ),
            "absolute_gate_initial_value": float(absolute_gate_initial_value),
        }

        # 1) The 3D encoder, coarse mixer, and FPN contain no stochastic layers and are shared.
        self.head_encoder = AnatomyPyramidEncoder(
            input_channels=input_channels,
            base_channels=base_channels,
            levels=encoder_levels,
            activation_checkpointing=encoder_activation_checkpointing,
        )

        # 2) Register the mixer slot first and instantiate it later. Delayed initialization ensures
        # C and G variants have identical shared-core values at the same seed without shifting
        # the FPN/query/decoder RNG sequence. The C/P slot stays None and adds no parameters.
        mixer_scope: Literal["global", "local"] | None = None
        if _variant_uses_mixer(self.variant_id):
            mixer_scope = "local" if self.variant_id == "G_LOCAL" else "global"
        self.coarse_global_mixer: CoarseGlobalAnatomyMixer | None = None

        # 3) Lightweight FPN: lateral 1x1 projections, top-down fusion, then depthwise refinement.
        self.pyramid_laterals = nn.ModuleList(
            nn.Conv3d(channels, pyramid_dim, 1)
            for channels in self.head_encoder.output_channels
        )
        self.pyramid_refinements = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(
                    pyramid_dim,
                    pyramid_dim,
                    3,
                    padding=1,
                    groups=pyramid_dim,
                    bias=False,
                ),
                nn.Conv3d(pyramid_dim, pyramid_dim, 1, bias=False),
                nn.GroupNorm(_group_count(pyramid_dim), pyramid_dim),
                nn.SiLU(),
            )
            for _ in range(encoder_levels)
        )

        # 4) Subject token. ControlledSequential preserves v1 numeric keys and can disable dropout.
        global_input_dim = pyramid_dim * 2 + 14
        self.global_projection = ControlledSequential(
            nn.Linear(global_input_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.SiLU(),
            ControlledDropout(context_dropout),
            nn.Linear(global_dim, global_dim),
        )

        self.relative_position = FourierEncoding3d(coordinate_fourier_bands)
        self.absolute_position = FourierEncoding3d(absolute_fourier_bands)

        # 5) Query trunk: process Q queries once, independent of electrode count.
        query_local_dim = pyramid_dim * encoder_levels + (12 if use_1mm_features else 0)
        query_input_dim = query_local_dim + self.relative_position.output_dim
        self.query_input_projection = nn.Sequential(
            nn.Linear(query_input_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
        )
        self.query_trunk = nn.ModuleList(
            ResidualSwiGLUBlock(query_dim, query_expansion, context_dropout)
            for _ in range(query_blocks)
        )
        self.absolute_query_projection = ControlledSequential(
            nn.Linear(self.absolute_position.output_dim, absolute_aux_dim),
            nn.LayerNorm(absolute_aux_dim),
            nn.SiLU(),
            ControlledDropout(absolute_coordinate_dropout),
            nn.Linear(absolute_aux_dim, query_dim, bias=False),
        )

        # 6) Shared unique-electrode encoder without source/reference role embeddings.
        patch_summary_dim = encoder_levels * pyramid_dim * 2
        electrode_input_dim = (
            patch_summary_dim + global_dim + self.relative_position.output_dim + 3
        )
        electrode_hidden_dim = max(electrode_dim, 256)
        self.electrode_encoder = ControlledSequential(
            nn.Linear(electrode_input_dim, electrode_hidden_dim),
            nn.LayerNorm(electrode_hidden_dim),
            nn.SiLU(),
            ControlledDropout(context_dropout),
            nn.Linear(electrode_hidden_dim, electrode_dim),
            nn.LayerNorm(electrode_dim),
        )
        self.absolute_electrode_projection = ControlledSequential(
            nn.Linear(self.absolute_position.output_dim, absolute_aux_dim),
            nn.LayerNorm(absolute_aux_dim),
            nn.SiLU(),
            ControlledDropout(absolute_coordinate_dropout),
            nn.Linear(absolute_aux_dim, electrode_dim, bias=False),
        )

        # 7) Shared G(x,e) decoder. Pair subtraction occurs only after response-bank decoding.
        pair_geometry_dim = self.relative_position.output_dim + 4
        self.electrode_to_query = nn.Linear(electrode_dim, query_dim, bias=False)
        self.pair_geometry_projection = nn.Sequential(
            nn.Linear(pair_geometry_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.SiLU(),
        )
        self.response_blocks = nn.ModuleList(
            ElectrodeConditionedResidualBlock(
                query_dim,
                electrode_dim,
                decoder_bottleneck_dim,
                context_dropout,
            )
            for _ in range(decoder_blocks)
        )
        self.response_head = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, decoder_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(decoder_bottleneck_dim, 3),
        )

        initial_logit = torch.logit(
            torch.tensor(float(absolute_gate_initial_value), dtype=torch.float32)
        )
        self.absolute_query_gate_logit = nn.Parameter(initial_logit.clone())
        self.absolute_electrode_gate_logit = nn.Parameter(initial_logit.clone())
        # The final absolute-coordinate projection initially contributes zero.
        nn.init.zeros_(self.absolute_query_projection[-1].weight)
        nn.init.zeros_(self.absolute_electrode_projection[-1].weight)

        radius = 2.0 * self.patch_radius_mm / self.canonical_grid_extent_mm
        offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
                [0.0, 0.0, radius],
                [0.0, 0.0, -radius],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("patch_offsets_normalized", offsets, persistent=False)

        # 8) Initialize added G-variant parameters after the shared core to preserve same-seed
        # core pairing while retaining normal mixer/output-projection initialization. Only the
        # final scalar residual gate is set exactly to zero by CoarseGlobalAnatomyMixer.
        if mixer_scope is not None:
            self.coarse_global_mixer = CoarseGlobalAnatomyMixer(
                input_channels=self.head_encoder.output_channels[-1],
                mixer_dim=global_mixer_dim,
                blocks=global_mixer_blocks,
                heads=global_mixer_heads,
                scope=mixer_scope,
                local_window=global_mixer_local_window,
                max_relative_distance=global_mixer_max_relative_distance,
                swiglu_expansion=global_mixer_swiglu_expansion,
            )

    @property
    def absolute_query_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.absolute_query_gate_logit)

    @property
    def absolute_electrode_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.absolute_electrode_gate_logit)

    @property
    def has_global_mixer(self) -> bool:
        return self.coarse_global_mixer is not None

    def architecture_config(self) -> dict[str, object]:
        """Return the complete architecture configuration for checkpoint/run-contract hashing."""

        config = dict(self._model_config)
        config.update(
            {
                "model_version": HEAD2FIELD_V1_MODEL_VERSION,
                "variant_id": self.variant_id,
                "architecture_id": self.architecture_id,
                "coordinate_frame_id": HEAD2FIELD_V1_COORDINATE_FRAME_ID,
                "global_mixer_dim": self.global_mixer_dim,
                "global_mixer_blocks": self.global_mixer_blocks,
                "global_mixer_heads": self.global_mixer_heads,
                "global_mixer_local_window": self.global_mixer_local_window,
                "global_mixer_max_relative_distance": (
                    self.global_mixer_max_relative_distance
                ),
                "global_mixer_swiglu_expansion": self.global_mixer_swiglu_expansion,
            }
        )
        return config

    def identity_absolute_augmentation(
        self,
        geometry: SubjectGeometry,
    ) -> AbsoluteCoordinateAugmentation:
        """Return the physics branch identity augmentation with mask=1 and offset=0."""

        geometry.validate()
        batch = geometry.batch_size
        reference = geometry.anatomy_origin_normalized
        return AbsoluteCoordinateAugmentation(
            branch_mask=torch.ones(
                batch,
                1,
                device=reference.device,
                dtype=reference.dtype,
            ),
            coordinate_offset_normalized=torch.zeros(
                batch,
                3,
                device=reference.device,
                dtype=reference.dtype,
            ),
        )

    def sample_absolute_augmentation(
        self,
        geometry: SubjectGeometry,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> AbsoluteCoordinateAugmentation:
        """Sample one branch mask and shared coordinate jitter per subject.

        Queries and electrodes must reuse the same returned object. With
        ``deterministic=True`` or in evaluation mode, return the identity
        without consuming the generator.
        """

        geometry.validate()
        batch = geometry.batch_size
        reference = geometry.anatomy_origin_normalized
        device = reference.device
        dtype = reference.dtype
        if deterministic or not self.training:
            return self.identity_absolute_augmentation(geometry)

        keep_probability = 1.0 - self.absolute_branch_dropout
        if self.absolute_branch_dropout > 0.0:
            values = torch.rand(
                (batch, 1),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            branch_mask = (values < keep_probability).to(dtype=dtype)
            branch_mask = branch_mask / keep_probability
        else:
            branch_mask = torch.ones(batch, 1, device=device, dtype=dtype)

        coordinate_offset = torch.zeros(batch, 3, device=device, dtype=dtype)
        if (
            self.absolute_coordinate_jitter_probability > 0.0
            and self.absolute_coordinate_jitter_std_mm > 0.0
        ):
            use_jitter = (
                torch.rand(
                    (batch, 1),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                < self.absolute_coordinate_jitter_probability
            ).to(dtype=dtype)
            normalized_std = (
                2.0
                * self.absolute_coordinate_jitter_std_mm
                / self.canonical_grid_extent_mm
            )
            coordinate_offset = (
                torch.randn(
                    (batch, 3),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                * normalized_std
                * use_jitter
            )
        return AbsoluteCoordinateAugmentation(branch_mask, coordinate_offset)

    def _resolve_augmentation(
        self,
        coordinates: torch.Tensor,
        augmentation: AbsoluteCoordinateAugmentation,
        *,
        deterministic: bool,
    ) -> AbsoluteCoordinateAugmentation:
        augmentation.validate(coordinates.shape[0], coordinates.device)
        # Prevent stochastic data augmentation from being passed to the physics branch.
        if deterministic and (
            not bool((augmentation.branch_mask == 1).all())
            or not bool((augmentation.coordinate_offset_normalized == 0).all())
        ):
            raise ValueError("deterministic branches require identity augmentation")
        return augmentation

    def encode_head(
        self,
        head_volume: torch.Tensor,
        subject_geometry: SubjectGeometry,
    ) -> HeadFeatureBundle:
        """Encode ``[B,12,I,J,K]`` head inputs into a multiscale FPN bundle.

        The fixed v1 factory uses a 128-cube input and 32-channel levels at
        128/64/32/16 resolution. The deepest encoder feature receives anatomy
        mixing before top-down fusion. This shared path consumes no dropout RNG.
        """

        if head_volume.ndim != 5:
            raise ValueError("head_volume must have shape [B,C,D,H,W]")
        subject_geometry.validate()
        if head_volume.shape[0] != subject_geometry.batch_size:
            raise ValueError("head_volume and SubjectGeometry batch sizes must match")
        if tuple(head_volume.shape[-3:]) != tuple(subject_geometry.encoder_input_shape):
            raise ValueError(
                "head_volume spatial shape must match SubjectGeometry.encoder_input_shape"
            )

        raw_features = self.head_encoder(head_volume)
        if self.coarse_global_mixer is not None:
            raw_features[-1] = self.coarse_global_mixer(raw_features[-1])
        pyramid = [
            lateral(feature)
            for lateral, feature in zip(
                self.pyramid_laterals,
                raw_features,
                strict=True,
            )
        ]
        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-3:],
                mode="nearest",
            )
        refined = tuple(
            refinement(feature)
            for refinement, feature in zip(
                self.pyramid_refinements,
                pyramid,
                strict=True,
            )
        )
        return HeadFeatureBundle(refined, head_volume, subject_geometry)

    def build_global_context(
        self,
        head_features: HeadFeatureBundle,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
        anatomy_moments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a subject token from deepest-FPN foreground and 14D anatomy statistics."""

        if len(head_features.pyramid) != self.encoder_levels:
            raise ValueError("feature level count must match encoder_levels")
        head_volume = head_features.head_volume
        foreground = head_volume[:, :10].float().sum(dim=1, keepdim=True).clamp(0, 1)
        deepest = head_features.pyramid[-1].float()
        mask = F.interpolate(foreground, size=deepest.shape[-3:], mode="nearest")
        mean, std = _masked_channel_statistics(deepest, mask)
        if anatomy_moments is None:
            anatomy_moments = build_anatomy_statistics(head_volume)
        statistics = torch.cat([mean, std, anatomy_moments.float()], dim=-1)
        return self.global_projection(
            statistics,
            deterministic=deterministic,
            generator=generator,
        )

    def sample_patch_summary(
        self,
        features: Sequence[torch.Tensor],
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize seven-point electrode patches as ``[B,U,2*C*levels]``.

        Each level contributes the center feature and the mean over all seven
        samples (center plus six axial neighbors). Coordinates are normalized
        i/j/k and patch offsets use the configured physical radius.
        """

        batch, points, _ = coordinates.shape
        patch_coordinates = (
            coordinates[:, :, None] + self.patch_offsets_normalized[None, None]
        ).clamp(-1.0, 1.0)
        flat_coordinates = patch_coordinates.reshape(batch, points * 7, 3)
        summaries: list[torch.Tensor] = []
        for feature in features:
            sampled = sample_volume_at_points(feature, flat_coordinates)
            sampled = sampled.reshape(batch, points, 7, feature.shape[1])
            summaries.extend([sampled[:, :, 0], sampled.mean(dim=2)])
        return torch.cat(summaries, dim=-1)

    def _encode_electrode_bank(
        self,
        features: Sequence[torch.Tensor],
        global_token: torch.Tensor,
        electrode_coord: torch.Tensor,
        electrode_mm: torch.Tensor,
        anatomy_origin_mm: torch.Tensor,
        *,
        augmentation: AbsoluteCoordinateAugmentation,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        patch = self.sample_patch_summary(features, electrode_coord)
        centered_mm = (
            electrode_mm.float() - anatomy_origin_mm[:, None].float()
        ) / self.physical_coordinate_scale_mm
        global_expanded = global_token[:, None].expand(-1, electrode_coord.shape[1], -1)
        raw_token = torch.cat(
            [
                patch,
                global_expanded,
                self.relative_position(centered_mm),
                centered_mm,
            ],
            dim=-1,
        )
        token = self.electrode_encoder(
            raw_token,
            deterministic=deterministic,
            generator=generator,
        )
        resolved = self._resolve_augmentation(
            electrode_coord,
            augmentation,
            deterministic=deterministic,
        )
        absolute_coord = (
            electrode_coord + resolved.coordinate_offset_normalized[:, None]
        )
        absolute_token = self.absolute_electrode_projection(
            self.absolute_position(absolute_coord),
            deterministic=deterministic,
            generator=generator,
        )
        return token + (
            self.absolute_electrode_gate
            * resolved.branch_mask[:, :, None]
            * absolute_token
        )

    def encode_unique_electrodes(
        self,
        head_features: HeadFeatureBundle,
        global_context: torch.Tensor,
        electrodes: ElectrodeCoordinateBundle,
        *,
        augmentation: AbsoluteCoordinateAugmentation,
        deterministic: bool,
        generator: torch.Generator | None = None,
        validate_coordinates: bool = True,
    ) -> ElectrodeContext:
        """Encode each unique electrode_id once without source/reference role information."""

        geometry = head_features.subject_geometry
        electrodes.validate(
            geometry,
            require_unique_ids=True,
            check_coordinate_roundtrip=validate_coordinates,
        )
        token = self._encode_electrode_bank(
            head_features.pyramid,
            global_context,
            electrodes.normalized_coord,
            electrodes.world_ras_mm,
            geometry.anatomy_origin_world_mm,
            augmentation=augmentation,
            deterministic=deterministic,
            generator=generator,
        )
        return ElectrodeContext(
            token=token,
            electrode_id=electrodes.electrode_id,
            normalized_coord=electrodes.normalized_coord,
            world_ras_mm=electrodes.world_ras_mm,
        )

    def _build_query_token(
        self,
        features: Sequence[torch.Tensor],
        query_coord: torch.Tensor,
        query_features_1mm: torch.Tensor | None,
        anatomy_origin_coord: torch.Tensor,
        *,
        augmentation: AbsoluteCoordinateAugmentation,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        local_parts = [
            sample_volume_at_points(feature, query_coord) for feature in features
        ]
        if self.use_1mm_features:
            if query_features_1mm is None:
                raise ValueError(
                    "query_features_1mm is required when 1 mm query features are enabled"
                )
            if query_features_1mm.shape[:2] != query_coord.shape[:2]:
                raise ValueError(
                    "query_features_1mm and query_coord point counts must match"
                )
            if query_features_1mm.shape[-1] != 12:
                raise ValueError("v1 query_features_1mm must have final dimension 12")
            local_parts.append(query_features_1mm)
        query_centered = query_coord - anatomy_origin_coord[:, None]
        query_input = torch.cat(
            [torch.cat(local_parts, dim=-1), self.relative_position(query_centered)],
            dim=-1,
        )
        hidden = self.query_input_projection(query_input)
        for block in self.query_trunk:
            hidden = block(
                hidden,
                deterministic=deterministic,
                generator=generator,
            )

        resolved = self._resolve_augmentation(
            query_coord,
            augmentation,
            deterministic=deterministic,
        )
        absolute_coord = query_coord + resolved.coordinate_offset_normalized[:, None]
        absolute_hidden = self.absolute_query_projection(
            self.absolute_position(absolute_coord),
            deterministic=deterministic,
            generator=generator,
        )
        return hidden + (
            self.absolute_query_gate
            * resolved.branch_mask[:, :, None]
            * absolute_hidden
        )

    def encode_queries(
        self,
        head_features: HeadFeatureBundle,
        queries: QueryCoordinateBundle,
        *,
        augmentation: AbsoluteCoordinateAugmentation,
        deterministic: bool,
        generator: torch.Generator | None = None,
        require_unique_rows: bool = False,
        validate_coordinates: bool = True,
    ) -> QueryContext:
        """Encode Q queries once; physics unions should set ``require_unique_rows=True``."""

        geometry = head_features.subject_geometry
        queries.validate(
            geometry,
            require_unique_rows=require_unique_rows,
            check_coordinate_roundtrip=validate_coordinates,
        )
        token = self._build_query_token(
            head_features.pyramid,
            queries.normalized_coord,
            queries.features_1mm,
            geometry.anatomy_origin_normalized,
            augmentation=augmentation,
            deterministic=deterministic,
            generator=generator,
        )
        return QueryContext(
            token=token,
            normalized_coord=queries.normalized_coord,
            world_ras_mm=queries.world_ras_mm,
        )

    def _single_electrode_responses(
        self,
        query_token: torch.Tensor,
        electrode_token: torch.Tensor,
        query_coord: torch.Tensor,
        electrode_coord: torch.Tensor,
        query_mm: torch.Tensor,
        electrode_mm: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Compute the ``G(query,electrode)`` bank once with shape ``[B,U,Q,3]``."""

        coordinate_delta = query_coord[:, None] - electrode_coord[:, :, None]
        physical_delta = (
            query_mm[:, None].float() - electrode_mm[:, :, None].float()
        ) / self.physical_coordinate_scale_mm
        physical_distance = torch.linalg.vector_norm(
            physical_delta,
            dim=-1,
            keepdim=True,
        )
        pair_geometry = torch.cat(
            [
                self.relative_position(coordinate_delta),
                physical_delta,
                physical_distance,
            ],
            dim=-1,
        )
        hidden = (
            query_token[:, None]
            + self.electrode_to_query(electrode_token)[:, :, None]
            + self.pair_geometry_projection(pair_geometry)
        )
        for block in self.response_blocks:
            hidden = block(
                hidden,
                electrode_token,
                deterministic=deterministic,
                generator=generator,
            )
        return self.response_head(hidden)

    def decode_response_bank(
        self,
        query_context: QueryContext,
        electrode_context: ElectrodeContext,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return shared responses ``[B,U,Q,3]`` before bipolar subtraction.

        Pair indices refer to the electrode axis of this bank. Apply physical
        losses to the gathered pair fields, which carry the current convention.
        """

        return self._single_electrode_responses(
            query_context.token,
            electrode_context.token,
            query_context.normalized_coord,
            electrode_context.normalized_coord,
            query_context.world_ras_mm,
            electrode_context.world_ras_mm,
            deterministic=deterministic,
            generator=generator,
        )

    @staticmethod
    def gather_pair_field(
        response_bank: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Build pair fields from ``[source_bank_index, reference_bank_index]``.

        Repeated pairs, reversals, identical endpoints, and cocycles within one
        bank share the same ``G`` values. Token-level noise therefore cannot
        break the algebraic identities even in training mode with dropout.
        """

        if response_bank.ndim != 4 or response_bank.shape[-1] != 3:
            raise ValueError("response_bank must have shape [B,U,Q,3]")
        if pair_index.ndim != 3 or pair_index.shape[-1] != 2:
            raise ValueError("pair_index must have shape [B,P,2]")
        if pair_index.dtype != torch.long:
            raise TypeError("pair_index must use torch.long")
        batch, electrodes, _, _ = response_bank.shape
        if pair_index.shape[0] != batch:
            raise ValueError("pair_index and response_bank batch sizes must match")
        if pair_index.shape[1] <= 0:
            raise ValueError("pair_index must contain at least one pair")
        if pair_index.device != response_bank.device:
            raise ValueError("pair_index and response_bank must be on the same device")
        if bool((pair_index < 0).any()) or bool((pair_index >= electrodes).any()):
            raise IndexError(
                "pair_index contains a negative or out-of-range bank index"
            )
        batch_index = torch.arange(batch, device=response_bank.device)[:, None]
        source = response_bank[batch_index, pair_index[..., 0]]
        reference = response_bank[batch_index, pair_index[..., 1]]
        return source - reference

    def decode_pair_field(
        self,
        query_context: QueryContext,
        electrode_context: ElectrodeContext,
        pair_index: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return source-minus-reference fields ``[B,P,Q,3]`` for requested pairs."""

        response_bank = self.decode_response_bank(
            query_context,
            electrode_context,
            deterministic=deterministic,
            generator=generator,
        )
        return self.gather_pair_field(response_bank, pair_index)

    def decode_branch(
        self,
        query_context: QueryContext,
        electrode_context: ElectrodeContext,
        pair_index: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> BranchOutput:
        """Evaluate one shared response bank and gather all requested pair fields."""

        response_bank = self.decode_response_bank(
            query_context,
            electrode_context,
            deterministic=deterministic,
            generator=generator,
        )
        pair_field = self.gather_pair_field(response_bank, pair_index)
        return BranchOutput(response_bank=response_bank, pair_field=pair_field)

    def _decode_bundled_from_features(
        self,
        head_features: HeadFeatureBundle,
        queries: QueryCoordinateBundle,
        electrodes: ElectrodeCoordinateBundle,
        pair_index: torch.Tensor,
        *,
        augmentation: AbsoluteCoordinateAugmentation,
        deterministic: bool,
        dropout_generator: torch.Generator | None,
        anatomy_moments: torch.Tensor | None,
        require_unique_queries: bool,
        validate_coordinates: bool,
    ) -> BranchOutput:
        global_context = self.build_global_context(
            head_features,
            deterministic=deterministic,
            generator=dropout_generator,
            anatomy_moments=anatomy_moments,
        )
        query_context = self.encode_queries(
            head_features,
            queries,
            augmentation=augmentation,
            deterministic=deterministic,
            generator=dropout_generator,
            require_unique_rows=require_unique_queries,
            validate_coordinates=validate_coordinates,
        )
        electrode_context = self.encode_unique_electrodes(
            head_features,
            global_context,
            electrodes,
            augmentation=augmentation,
            deterministic=deterministic,
            generator=dropout_generator,
            validate_coordinates=validate_coordinates,
        )
        return self.decode_branch(
            query_context,
            electrode_context,
            pair_index,
            deterministic=deterministic,
            generator=dropout_generator,
        )

    def forward_bundled(
        self,
        head_volume: torch.Tensor,
        subject_geometry: SubjectGeometry,
        queries: QueryCoordinateBundle,
        unique_electrodes: ElectrodeCoordinateBundle,
        pair_index: torch.Tensor,
        *,
        deterministic: bool = False,
        augmentation: AbsoluteCoordinateAugmentation | None = None,
        augmentation_generator: torch.Generator | None = None,
        dropout_generator: torch.Generator | None = None,
        anatomy_moments: torch.Tensor | None = None,
        require_unique_queries: bool = False,
        validate_coordinates: bool = True,
    ) -> BranchOutput:
        """Evaluate one branch and return both shared responses and pair fields.

        ``pair_index`` is int64 ``[B,P,2]`` indexing ``unique_electrodes`` in
        source/reference order. For inference, set ``model.eval()`` and
        ``deterministic=True``. Training may supply separate augmentation and
        dropout generators. Use an inference session to reuse the head encoding
        across repeated query chunks.
        """

        head_features = self.encode_head(head_volume, subject_geometry)
        if augmentation is None:
            augmentation = self.sample_absolute_augmentation(
                subject_geometry,
                generator=augmentation_generator,
                deterministic=deterministic,
            )
        return self._decode_bundled_from_features(
            head_features,
            queries,
            unique_electrodes,
            pair_index,
            augmentation=augmentation,
            deterministic=deterministic,
            dropout_generator=dropout_generator,
            anatomy_moments=anatomy_moments,
            require_unique_queries=require_unique_queries,
            validate_coordinates=validate_coordinates,
        )

    def forward_data_and_physics(
        self,
        head_volume: torch.Tensor,
        subject_geometry: SubjectGeometry,
        data_queries: QueryCoordinateBundle,
        physics_query_union: QueryCoordinateBundle,
        unique_electrodes: ElectrodeCoordinateBundle,
        pair_index: torch.Tensor,
        *,
        physics_pair_index: torch.Tensor | None = None,
        data_augmentation: AbsoluteCoordinateAugmentation | None = None,
        data_augmentation_generator: torch.Generator | None = None,
        data_dropout_generator: torch.Generator | None = None,
        anatomy_moments: torch.Tensor | None = None,
        validate_coordinates: bool = True,
        physics_requires_grad: bool | None = None,
    ) -> DataPhysicsOutput:
        """Encode the head once, then run stochastic data and deterministic physics branches.

        This method does not switch ``model.train()/eval()``. By default only
        P/GP builds a physics computation graph. C/G/G_LOCAL follows the same
        query schedule with ``grad_enabled=False``, avoiding the memory cost of
        retaining a graph and detaching it afterward. ``physics_requires_grad``
        permits gradient-free P/GP diagnostics but cannot add physics gradients
        to a data-only variant. The physics branch receives no generators and
        does not advance the data RNG sequence.
        """

        if physics_requires_grad is None:
            physics_requires_grad = self.variant_id in ("P", "GP")
        if physics_requires_grad and self.variant_id not in ("P", "GP"):
            raise ValueError(
                f"variant {self.variant_id} uses data supervision only and cannot enable physics gradients"
            )

        head_features = self.encode_head(head_volume, subject_geometry)
        if data_augmentation is None:
            data_augmentation = self.sample_absolute_augmentation(
                subject_geometry,
                generator=data_augmentation_generator,
                deterministic=False,
            )
        data_output = self._decode_bundled_from_features(
            head_features,
            data_queries,
            unique_electrodes,
            pair_index,
            augmentation=data_augmentation,
            deterministic=False,
            dropout_generator=data_dropout_generator,
            anatomy_moments=anatomy_moments,
            require_unique_queries=False,
            validate_coordinates=validate_coordinates,
        )

        identity = self.identity_absolute_augmentation(subject_geometry)
        build_physics_graph = torch.is_grad_enabled() and physics_requires_grad
        with torch.set_grad_enabled(build_physics_graph):
            physics_output = self._decode_bundled_from_features(
                head_features,
                physics_query_union,
                unique_electrodes,
                pair_index if physics_pair_index is None else physics_pair_index,
                augmentation=identity,
                deterministic=True,
                dropout_generator=None,
                anatomy_moments=anatomy_moments,
                require_unique_queries=True,
                validate_coordinates=validate_coordinates,
            )
        return DataPhysicsOutput(data=data_output, physics=physics_output)


__all__ = [
    "HEAD2FIELD_V1_CANONICAL_GRID_EXTENT_MM",
    "HEAD2FIELD_V1_COORDINATE_FRAME_ID",
    "HEAD2FIELD_V1_CORE_ARCHITECTURE_ID",
    "HEAD2FIELD_V1_DEFAULT_ENCODER_INPUT_SHAPE",
    "HEAD2FIELD_V1_GLOBAL_ARCHITECTURE_ID",
    "HEAD2FIELD_V1_LOCAL_ARCHITECTURE_ID",
    "HEAD2FIELD_V1_MODEL_VERSION",
    "MODEL_VARIANTS",
    "AbsoluteCoordinateAugmentation",
    "AnatomyPyramidEncoder",
    "AxialSelfAttention3d",
    "BottleneckResidual3d",
    "BranchOutput",
    "CoarseGlobalAnatomyMixer",
    "DataPhysicsOutput",
    "ElectrodeConditionedResidualBlock",
    "ElectrodeContext",
    "ElectrodeCoordinateBundle",
    "FourierEncoding3d",
    "GlobalMixerBlock",
    "Head2FieldV1",
    "HeadFeatureBundle",
    "QueryContext",
    "QueryCoordinateBundle",
    "ResidualSwiGLUBlock",
    "SubjectGeometry",
    "architecture_id_for_variant",
    "build_anatomy_statistics",
    "sample_volume_at_points",
    "stable_unique_electrodes",
]
