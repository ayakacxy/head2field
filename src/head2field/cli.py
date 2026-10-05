"""Predict 75 source-to-Oz lead fields with Head2Field v1 from a prepared subject."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .inference import Head2FieldInferenceSession
from .metadata import file_records
from .inputs import (
    build_prepared_model_input_subject,
    ijk_to_world_mm,
)
from .model import (
    HEAD2FIELD_V1_DEFAULT_ENCODER_INPUT_SHAPE,
)
from .preprocessing.validation import (
    PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    PREPARED_SUBJECT_SHAPE,
    PREPARED_SUBJECT_MANIFEST_NAME,
    REQUIRED_PREPARED_SUBJECT_FILES,
    load_prepared_subject_v1,
    validate_prepared_subject_manifest,
)
from .weights import load_head2field_v1_weights, materialized_head2field_v1_weights

PREDICTION_SCHEMA_ID = "head2field_v1_source_to_oz_prediction"
PREDICTION_ARRAY_NAME = "lead_field_float32.npy"
QUERY_IJK_NAME = "query_ijk.npy"
QUERY_WORLD_RAS_NAME = "query_world_ras_mm.npy"
ELECTRODE_PAIRS_NAME = "electrode_pairs.json"
PREDICTION_MANIFEST_NAME = "prediction_manifest_v1.json"


def _is_within(path: Path, directory: Path) -> bool:
    """Check whether a resolved path lies within the specified directory."""

    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_new_output_target(path: Path, prepared_subject: Path) -> Path:
    """Audit the prediction destination without creating any files."""

    resolved = path.expanduser().resolve(strict=False)
    prepared_resolved = prepared_subject.expanduser().resolve(strict=True)
    if _is_within(resolved, prepared_resolved):
        raise ValueError(
            "Prediction output must be outside the prepared-subject input directory"
        )
    if resolved.exists():
        raise FileExistsError(f"Prediction output must be a new directory: {resolved}")
    return resolved


def _new_output_directory(path: Path, prepared_subject: Path) -> Path:
    """Exclusively create the prediction directory after read-only preflight."""

    resolved = _validate_new_output_target(path, prepared_subject)
    try:
        resolved.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Prediction output must be a new directory: {resolved}"
        ) from error
    return resolved


def _write_json_exclusive(path: Path, value: object) -> None:
    """Exclusively write, flush, and close a JSON file."""

    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _write_npy_exclusive(path: Path, value: np.ndarray) -> None:
    """Atomically write a pickle-free NPY file in the destination directory."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing array: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as file:
            np.save(file, value, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def predict_prepared_subject_v1(
    *,
    prepared_subject: Path,
    weights: Path | None = None,
    output_dir: Path,
    device: torch.device | str = "cpu",
    query_chunk_size: int = 32_768,
    electrode_batch_size: int = 76,
    expected_shape: tuple[int, int, int] = PREPARED_SUBJECT_SHAPE,
    expected_affine: Sequence[Sequence[float]] = PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    encoder_input_shape: tuple[
        int, int, int
    ] = HEAD2FIELD_V1_DEFAULT_ENCODER_INPUT_SHAPE,
) -> Path:
    """Write a validated subject's full 75-source basis and return its NPY path.

    ``weights=None`` selects the bundled pretrained model. ``output_dir`` must
    be new and outside the input directory. Chunk sizes control memory use;
    query/source identities and the +1 mA / -1 mA convention stay fixed.
    相邻清单记录单位、坐标系和数据顺序。
    Shape/affine overrides are intended for controlled synthetic fixtures.
    """

    with materialized_head2field_v1_weights(weights) as resolved:
        return _predict_prepared_subject_v1_with_weights(
            prepared_subject=prepared_subject,
            weights=resolved,
            output_dir=output_dir,
            device=device,
            query_chunk_size=query_chunk_size,
            electrode_batch_size=electrode_batch_size,
            expected_shape=expected_shape,
            expected_affine=expected_affine,
            encoder_input_shape=encoder_input_shape,
        )


def _predict_prepared_subject_v1_with_weights(
    *,
    prepared_subject: Path,
    weights: Path,
    output_dir: Path,
    device: torch.device | str,
    query_chunk_size: int,
    electrode_batch_size: int,
    expected_shape: tuple[int, int, int],
    expected_affine: Sequence[Sequence[float]],
    encoder_input_shape: tuple[int, int, int],
) -> Path:
    """Keep the resolved weight file available throughout read-only validation and inference."""

    if query_chunk_size <= 0 or electrode_batch_size <= 0:
        raise ValueError("query_chunk_size and electrode_batch_size must be positive")
    prepared_path = Path(prepared_subject)
    weight_path = Path(weights)
    _validate_new_output_target(output_dir, prepared_path)
    validate_prepared_subject_manifest(
        prepared_path,
        expected_shape=expected_shape,
        expected_affine=expected_affine,
    )
    weights_stat_before = (weight_path).stat()
    input_paths = [
        prepared_path / relative
        for relative in (
            *REQUIRED_PREPARED_SUBJECT_FILES,
            PREPARED_SUBJECT_MANIFEST_NAME,
        )
    ]
    input_records_before = file_records(input_paths, relative_to=prepared_path)
    prepared = load_prepared_subject_v1(
        prepared_path,
        expected_shape=expected_shape,
        expected_affine=expected_affine,
    )
    subject = build_prepared_model_input_subject(
        prepared,
        encoder_input_shape=encoder_input_shape,
    )
    model = load_head2field_v1_weights(
        weight_path,
        device=device,
    )
    output_directory = _new_output_directory(output_dir, prepared_path)
    array_path = output_directory / PREDICTION_ARRAY_NAME
    temporary_array_path = output_directory / (
        f".{PREDICTION_ARRAY_NAME}.tmp-{uuid.uuid4().hex}"
    )
    query_count = len(subject.gmwm_ijk)
    source_count = len(subject.source_names)
    output = np.lib.format.open_memmap(
        temporary_array_path,
        mode="w+",
        dtype=np.float32,
        shape=(source_count, query_count, 3),
    )
    session = Head2FieldInferenceSession(
        model,
        subject,
        device=device,
        query_cache_size=1,
    )
    prepared_electrodes = session.prepare_electrodes(
        subject.source_coords_mm,
        source_electrode_ids=np.arange(source_count, dtype=np.int64),
        reference_world_ras_mm=subject.reference_coord_mm,
        reference_electrode_id=source_count,
        electrode_batch_size=electrode_batch_size,
    )
    rows = np.arange(query_count, dtype=np.int64)
    for chunk in session.iter_predictions(
        rows,
        prepared_electrodes,
        query_chunk_size=query_chunk_size,
    ):
        prediction = chunk.prediction[0].detach().float().cpu().numpy()
        output[:, chunk.row_indices, :] = prediction
    output.flush()
    del output
    descriptor = os.open(temporary_array_path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if array_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing prediction: {array_path}"
        )
    os.replace(temporary_array_path, array_path)

    validate_prepared_subject_manifest(
        prepared_path,
        expected_shape=expected_shape,
        expected_affine=expected_affine,
    )
    weights_stat_after = (weight_path).stat()
    if weights_stat_after != weights_stat_before:
        raise RuntimeError("推理期间权重文件发生变化")
    if file_records(input_paths, relative_to=prepared_path) != input_records_before:
        raise RuntimeError("推理期间预处理输入文件发生变化")
    query_identity = np.ascontiguousarray(
        subject.gmwm_ijk,
        dtype=np.int32,
    )
    query_world_ras = np.ascontiguousarray(
        ijk_to_world_mm(
            query_identity.astype(np.float32),
            subject.affine,
        ),
        dtype=np.float32,
    )
    _write_npy_exclusive(output_directory / QUERY_IJK_NAME, query_identity)
    _write_npy_exclusive(
        output_directory / QUERY_WORLD_RAS_NAME,
        query_world_ras,
    )
    electrode_pairs = [
        {
            "source": {
                "name": name,
                "world_ras_mm": [float(value) for value in source_coord],
                "current_mA": 1.0,
            },
            "reference": {
                "name": "Oz",
                "world_ras_mm": [float(value) for value in subject.reference_coord_mm],
                "current_mA": -1.0,
            },
        }
        for name, source_coord in zip(
            subject.source_names,
            subject.source_coords_mm,
            strict=True,
        )
    ]
    _write_json_exclusive(
        output_directory / ELECTRODE_PAIRS_NAME,
        electrode_pairs,
    )
    manifest = {
        "schema_id": PREDICTION_SCHEMA_ID,
        "model_version": "v1",
        "array": {
            "relative_path": PREDICTION_ARRAY_NAME,
            "shape": [source_count, query_count, 3],
            "dtype": "float32",
            "unit": "V/m",
            "axis_order": ["source", "query", "world_ras_component"],
        },
        "query_ijk": {
            "relative_path": QUERY_IJK_NAME,
            "shape": [query_count, 3],
            "dtype": "int32",
        },
        "query_world_ras_mm": {
            "relative_path": QUERY_WORLD_RAS_NAME,
            "shape": [query_count, 3],
            "dtype": "float32",
            "unit": "mm",
            "component_order": ["R", "A", "S"],
        },
        "electrode_pairs": {
            "relative_path": ELECTRODE_PAIRS_NAME,
            "count": source_count,
            "direction": "source_to_Oz",
        },
        "source_names": list(subject.source_names),
        "reference_electrode": "Oz",
        "coordinate_frame": "world-RAS mm",
        "field_contract": {
            "source_current_mA": 1.0,
            "reference_current_mA": -1.0,
            "field_unit": "V/m",
            "equivalent_lead_field_coefficient_unit": "V/m/mA",
        },
    }
    _write_json_exclusive(
        output_directory / PREDICTION_MANIFEST_NAME,
        manifest,
    )
    directory_descriptor = os.open(output_directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return array_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public prediction command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-subject",
        type=Path,
        required=True,
        help="Validated prepared-subject v1 directory containing the input manifest.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Custom v1 weight file; defaults to the pretrained weights bundled with this installation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; must not already exist or be inside the prepared-subject input directory.",
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device, such as cpu, cuda, or cuda:0."
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=32_768,
        help="Maximum query rows per chunk (default: 32768); reduce for less memory use.",
    )
    parser.add_argument(
        "--electrode-batch-size",
        type=int,
        default=76,
        help="Electrode encoding and pair decoding batch limit (default: 76).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run public prediction and print the final array path."""

    args = parse_args(argv)
    path = predict_prepared_subject_v1(
        prepared_subject=args.prepared_subject,
        weights=args.weights,
        output_dir=args.output_dir,
        device=args.device,
        query_chunk_size=args.query_chunk_size,
        electrode_batch_size=args.electrode_batch_size,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "ELECTRODE_PAIRS_NAME",
    "PREDICTION_ARRAY_NAME",
    "PREDICTION_MANIFEST_NAME",
    "PREDICTION_SCHEMA_ID",
    "QUERY_IJK_NAME",
    "QUERY_WORLD_RAS_NAME",
    "main",
    "parse_args",
    "predict_prepared_subject_v1",
]
