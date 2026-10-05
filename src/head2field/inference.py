"""Cached Head2Field v1 inference using input bundles only."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from .inputs import (
    ElectrodePairInputs,
    ModelInputSubject,
    build_electrode_pair_inputs,
    build_query_bundle,
    build_subject_geometry,
)
from .model import (
    ElectrodeContext,
    ElectrodeCoordinateBundle,
    Head2FieldV1,
    HeadFeatureBundle,
    QueryContext,
)

PrecisionContextFactory = Callable[[], AbstractContextManager[object]]


@dataclass(frozen=True)
class PreparedElectrodePairs:
    """Cached unique-electrode context and its source/reference pair indices."""

    inputs: ElectrodePairInputs
    context: ElectrodeContext
    pair_batch_size: int

    @property
    def pair_count(self) -> int:
        return self.inputs.pair_count


@dataclass(frozen=True)
class InferenceChunk:
    """An ordered query-row chunk and its ``[1,P,Q,3]`` prediction."""

    row_indices: np.ndarray
    prediction: torch.Tensor


def _slice_electrode_bundle(
    bundle: ElectrodeCoordinateBundle,
    start: int,
    end: int,
) -> ElectrodeCoordinateBundle:
    return ElectrodeCoordinateBundle(
        electrode_id=bundle.electrode_id[:, start:end],
        normalized_coord=bundle.normalized_coord[:, start:end],
        world_ras_mm=bundle.world_ras_mm[:, start:end],
    )


def _slice_electrode_context(
    context: ElectrodeContext,
    indices: torch.Tensor,
) -> ElectrodeContext:
    return ElectrodeContext(
        token=context.token.index_select(1, indices),
        electrode_id=context.electrode_id.index_select(1, indices),
        normalized_coord=context.normalized_coord.index_select(1, indices),
        world_ras_mm=context.world_ras_mm.index_select(1, indices),
    )


def _compact_pair_bank(
    pair_index: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap one subject's ``[1,P,2]`` pairs to a minimal electrode bank.

    Return global bank indices in first-occurrence order and local pair indices.
    Shared references appear once; requested pair order and direction are kept.
    The decoder can then allocate responses only for electrodes in this batch.
    """

    ordered_bank_indices: list[int] = []
    local_index_by_global: dict[int, int] = {}
    for raw_index in pair_index[0].reshape(-1).detach().cpu().tolist():
        global_index = int(raw_index)
        if global_index not in local_index_by_global:
            local_index_by_global[global_index] = len(ordered_bank_indices)
            ordered_bank_indices.append(global_index)
    gather_index = torch.tensor(
        ordered_bank_indices,
        dtype=torch.long,
        device=device,
    )
    local_pairs = torch.tensor(
        [
            [
                local_index_by_global[int(source)],
                local_index_by_global[int(reference)],
            ]
            for source, reference in pair_index[0].detach().cpu().tolist()
        ],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    return gather_index, local_pairs


def _strict_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_rows(
    values: Sequence[int] | np.ndarray,
    *,
    row_count: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(f"{name} must be a non-empty 1D integer sequence")
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= row_count):
        raise IndexError(f"{name} is out of range; valid range is [0,{row_count})")
    return result


class Head2FieldInferenceSession:
    """Cache head, global, electrode, and query computations for a read-only v1 subject.

    Electrode encoding is chunked by ``electrode_batch_size``. Pair decoding uses
    the same limit and remaps each chunk to the smallest local electrode context,
    avoiding a full ``U x Q`` hidden tensor for large source libraries. A small LRU
    cache holds query contexts shared by all pair batches.

    ``subject`` and ``model`` must stay unchanged for the session lifetime. Move
    the model to ``device`` before construction. Inference uses evaluation mode,
    identity augmentation, and FP32 unless ``precision_context`` is supplied.
    Create a new session after changing weights or subject inputs.
    """

    def __init__(
        self,
        model: Head2FieldV1,
        subject: ModelInputSubject,
        *,
        device: torch.device | str,
        precision_context: PrecisionContextFactory | None = None,
        query_cache_size: int = 1,
    ) -> None:
        self.model = model
        self.subject = subject
        self.device = torch.device(device)
        self.precision_context: PrecisionContextFactory = (
            precision_context if precision_context is not None else nullcontext
        )
        self.query_cache_size = _strict_positive_integer(
            query_cache_size,
            name="query_cache_size",
        )
        self.model.eval()
        self.geometry = build_subject_geometry(subject, device=self.device)
        self.head_volume = subject.head_volume.unsqueeze(0).to(self.device)
        self.augmentation = self.model.identity_absolute_augmentation(self.geometry)
        self._head_features: HeadFeatureBundle | None = None
        self._global_context: torch.Tensor | None = None
        self._electrode_cache: dict[tuple[object, ...], PreparedElectrodePairs] = {}
        self._query_cache: OrderedDict[tuple[int, ...], QueryContext] = OrderedDict()

    @property
    def head_features(self) -> HeadFeatureBundle:
        self._ensure_head_and_global()
        assert self._head_features is not None
        return self._head_features

    @property
    def global_context(self) -> torch.Tensor:
        self._ensure_head_and_global()
        assert self._global_context is not None
        return self._global_context

    @property
    def cached_electrode_count(self) -> int:
        return len(self._electrode_cache)

    @property
    def cached_query_count(self) -> int:
        return len(self._query_cache)

    def _ensure_head_and_global(self) -> None:
        if self._head_features is not None and self._global_context is not None:
            return
        with torch.inference_mode(), self.precision_context():
            if self._head_features is None:
                self._head_features = self.model.encode_head(
                    self.head_volume,
                    self.geometry,
                )
            if self._global_context is None:
                self._global_context = self.model.build_global_context(
                    self._head_features,
                    deterministic=True,
                )

    @staticmethod
    def _electrode_cache_key(
        inputs: ElectrodePairInputs,
        *,
        electrode_batch_size: int,
    ) -> tuple[object, ...]:
        electrodes = inputs.electrodes
        return (
            tuple(electrodes.electrode_id[0].detach().cpu().tolist()),
            electrodes.world_ras_mm.detach().float().cpu().numpy().tobytes(),
            inputs.pair_index.detach().cpu().numpy().tobytes(),
            electrode_batch_size,
        )

    def prepare_electrodes(
        self,
        source_world_ras_mm: np.ndarray | Sequence[Sequence[float]],
        *,
        source_electrode_ids: Sequence[int] | np.ndarray | None = None,
        reference_world_ras_mm: np.ndarray | Sequence[float] | None = None,
        reference_electrode_id: int | None = None,
        electrode_batch_size: int | None = None,
    ) -> PreparedElectrodePairs:
        """Prepare source-to-reference pairs in the supplied source order.

        Coordinates are world-RAS millimeters. The reference defaults to the
        subject's Oz coordinate. Repeated electrode IDs share an encoding and
        must have identical coordinates. ``electrode_batch_size`` bounds both
        electrode encoding and pair decoding; it does not reorder the pairs.
        """

        inputs = build_electrode_pair_inputs(
            self.subject,
            source_world_ras_mm,
            device=self.device,
            source_electrode_ids=source_electrode_ids,
            reference_world_ras_mm=reference_world_ras_mm,
            reference_electrode_id=reference_electrode_id,
            geometry=self.geometry,
        )
        pair_count = inputs.pair_count
        resolved_batch_size = (
            pair_count
            if electrode_batch_size is None
            else _strict_positive_integer(
                electrode_batch_size,
                name="electrode_batch_size",
            )
        )
        key = self._electrode_cache_key(
            inputs,
            electrode_batch_size=resolved_batch_size,
        )
        cached = self._electrode_cache.get(key)
        if cached is not None:
            return cached

        contexts: list[ElectrodeContext] = []
        bank = inputs.electrodes
        self._ensure_head_and_global()
        assert self._head_features is not None
        assert self._global_context is not None
        with torch.inference_mode():
            for start in range(
                0,
                bank.electrode_count,
                resolved_batch_size,
            ):
                end = min(
                    start + resolved_batch_size,
                    bank.electrode_count,
                )
                with self.precision_context():
                    contexts.append(
                        self.model.encode_unique_electrodes(
                            self._head_features,
                            self._global_context,
                            _slice_electrode_bundle(bank, start, end),
                            augmentation=self.augmentation,
                            deterministic=True,
                        )
                    )
        context = ElectrodeContext(
            token=torch.cat([value.token for value in contexts], dim=1),
            electrode_id=torch.cat(
                [value.electrode_id for value in contexts],
                dim=1,
            ),
            normalized_coord=torch.cat(
                [value.normalized_coord for value in contexts],
                dim=1,
            ),
            world_ras_mm=torch.cat(
                [value.world_ras_mm for value in contexts],
                dim=1,
            ),
        )
        prepared = PreparedElectrodePairs(
            inputs=inputs,
            context=context,
            pair_batch_size=resolved_batch_size,
        )
        self._electrode_cache[key] = prepared
        return prepared

    def _query_context(
        self,
        row_indices: Sequence[int] | np.ndarray,
    ) -> QueryContext:
        rows = _strict_rows(
            row_indices,
            row_count=len(self.subject.gmwm_ijk),
            name="row_indices",
        )
        key = tuple(int(value) for value in rows)
        cached = self._query_cache.pop(key, None)
        if cached is not None:
            self._query_cache[key] = cached
            return cached
        queries = build_query_bundle(
            self.subject,
            rows,
            device=self.device,
            geometry=self.geometry,
        )
        self._ensure_head_and_global()
        assert self._head_features is not None
        with torch.inference_mode(), self.precision_context():
            context = self.model.encode_queries(
                self._head_features,
                queries,
                augmentation=self.augmentation,
                deterministic=True,
            )
        self._query_cache[key] = context
        while len(self._query_cache) > self.query_cache_size:
            self._query_cache.popitem(last=False)
        return context

    def _decode_pair_batch(
        self,
        query_context: QueryContext,
        electrode_context: ElectrodeContext,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one pair batch with only its required electrode contexts."""

        gather_index, local_pairs = _compact_pair_bank(
            pair_index,
            device=self.device,
        )
        local_context = _slice_electrode_context(
            electrode_context,
            gather_index,
        )
        with torch.inference_mode(), self.precision_context():
            return self.model.decode_branch(
                query_context,
                local_context,
                local_pairs,
                deterministic=True,
            ).pair_field

    def predict_rows(
        self,
        row_indices: Sequence[int] | np.ndarray,
        prepared_electrodes: PreparedElectrodePairs,
        *,
        pair_indices: Sequence[int] | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Return ``[1,P,Q,3]`` fields in V/m for source +1 mA / reference -1 mA.

        ``row_indices`` select rows of ``subject.gmwm_ijk``; ``pair_indices``
        select prepared pairs. Both sequences retain their supplied order,
        including repeated selections. The result remains on the session device.
        """

        query_context = self._query_context(row_indices)
        if pair_indices is None:
            selected = np.arange(
                prepared_electrodes.pair_count,
                dtype=np.int64,
            )
        else:
            selected = _strict_rows(
                pair_indices,
                row_count=prepared_electrodes.pair_count,
                name="pair_indices",
            )
        fields: list[torch.Tensor] = []
        pair_index = prepared_electrodes.inputs.pair_index
        for start in range(
            0,
            len(selected),
            prepared_electrodes.pair_batch_size,
        ):
            batch_selection = torch.from_numpy(
                selected[start : start + prepared_electrodes.pair_batch_size]
            ).to(device=self.device)
            fields.append(
                self._decode_pair_batch(
                    query_context,
                    prepared_electrodes.context,
                    pair_index.index_select(1, batch_selection),
                )
            )
        return torch.cat(fields, dim=1)

    def iter_predictions(
        self,
        row_indices: Sequence[int] | np.ndarray,
        prepared_electrodes: PreparedElectrodePairs,
        *,
        query_chunk_size: int,
        pair_indices: Sequence[int] | np.ndarray | None = None,
    ) -> Iterator[InferenceChunk]:
        """Yield bounded chunks with their original GM/WM row identities.

        Concatenating predictions along axis 2 reconstructs the supplied query
        order. Consume each chunk before retaining the next to bound GPU memory;
        collecting all chunks keeps their tensors alive.
        """

        chunk_size = _strict_positive_integer(
            query_chunk_size,
            name="query_chunk_size",
        )
        rows = _strict_rows(
            row_indices,
            row_count=len(self.subject.gmwm_ijk),
            name="row_indices",
        )
        for start in range(0, len(rows), chunk_size):
            chunk_rows = rows[start : start + chunk_size]
            yield InferenceChunk(
                row_indices=chunk_rows,
                prediction=self.predict_rows(
                    chunk_rows,
                    prepared_electrodes,
                    pair_indices=pair_indices,
                ),
            )

    def clear_query_cache(self) -> None:
        """Release cached query contexts while keeping head and electrode features."""

        self._query_cache.clear()

    def clear_electrode_cache(self) -> None:
        """Release the session's electrode cache; caller-held pairs remain alive."""

        self._electrode_cache.clear()


__all__ = [
    "Head2FieldInferenceSession",
    "InferenceChunk",
    "PreparedElectrodePairs",
]
