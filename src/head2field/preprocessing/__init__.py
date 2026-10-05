"""Lightweight prepared-subject contracts and lazily loaded preprocessing APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .conductivity import (
    conductivity_table_metadata,
    labels_to_conductivity,
    normalize_conductivity_for_model,
    validate_conductivity_arrays,
)
from .electrodes import (
    DEFAULT_SIMNIBS_CAP_RELATIVE_PATH,
    STANDARD_76_ELECTRODE_NAMES,
    load_standard_cap_from_head_model,
    load_standard_electrode_inputs,
)
from .profiles import (
    HCP_ACPC_PROFILE,
    NON_HCP_RAW_PROFILE,
    T1_PREPROCESSING_PROFILES,
)
from .validation import (
    PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM,
    PREPARED_SUBJECT_MANIFEST_NAME,
    PREPARED_SUBJECT_SCHEMA_ID,
    PREPARED_SUBJECT_SHAPE,
    PREPARED_SUBJECT_VOXEL_SIZE_MM,
    PreparedSubjectV1,
    build_prepared_subject_manifest,
    load_prepared_subject_v1,
    prepared_subject_manifest_sha256,
    validate_prepared_subject_manifest,
)

# Load heavy SciPy/external SimNIBS paths only when preprocessing APIs are used.
_LAZY_EXPORTS = {
    "preprocess_t1": (".pipeline", "preprocess_t1"),
    "PreparationResult": (".assembly", "PreparationResult"),
    "assemble_prepared_subject": (".assembly", "assemble_prepared_subject"),
    "prepare_subject": (".assembly", "prepare_subject"),
    "SegmentationAdapter": (".charm_adapter", "SegmentationAdapter"),
    "SegmentationAdapterError": (".charm_adapter", "SegmentationAdapterError"),
    "SimNIBSCharmAdapter": (".charm_adapter", "SimNIBSCharmAdapter"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public symbols requiring SciPy or external runtimes."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_SIMNIBS_CAP_RELATIVE_PATH",
    "HCP_ACPC_PROFILE",
    "NON_HCP_RAW_PROFILE",
    "PREPARED_SUBJECT_AFFINE_IJK_TO_RAS_MM",
    "PREPARED_SUBJECT_MANIFEST_NAME",
    "PREPARED_SUBJECT_SCHEMA_ID",
    "PREPARED_SUBJECT_SHAPE",
    "PREPARED_SUBJECT_VOXEL_SIZE_MM",
    "STANDARD_76_ELECTRODE_NAMES",
    "T1_PREPROCESSING_PROFILES",
    "PreparationResult",
    "PreparedSubjectV1",
    "SegmentationAdapter",
    "SegmentationAdapterError",
    "SimNIBSCharmAdapter",
    "assemble_prepared_subject",
    "build_prepared_subject_manifest",
    "conductivity_table_metadata",
    "labels_to_conductivity",
    "load_prepared_subject_v1",
    "load_standard_cap_from_head_model",
    "load_standard_electrode_inputs",
    "normalize_conductivity_for_model",
    "prepare_subject",
    "prepared_subject_manifest_sha256",
    "preprocess_t1",
    "validate_conductivity_arrays",
    "validate_prepared_subject_manifest",
]
