"""Public APIs for generating, validating, and loading TES v1 FEM labels."""

from .contracts import (
    PRODUCTION_PROFILE,
    STANDARD_76_ELECTRODE_NAMES,
    FEMV1ContractError,
    FEMV1ProductionProfile,
    validate_leadfield_dataset_contract,
    validate_production_options,
)
from .generation import (
    FEM_LABELS_MANIFEST_NAME,
    FEMLabelGenerationPlan,
    FEMLabelsV1,
    build_fem_label_plan,
    dry_run_fem_generation,
    dry_run_fem_label_generation,
    export_fem_labels_from_hdf5,
    generate_fem_labels,
    load_fem_labels_v1,
    validate_fem_artifacts,
    validate_fem_labels_manifest,
)
from .leadfield_run import (
    LeadfieldRun,
    TDCSLeadfieldPlan,
    dry_run_tdcs_leadfield,
    run_tdcs_leadfield,
    validate_tdcs_leadfield_run,
)
from .simnibs_adapter import SimNIBSAdapterError
from .transaction import StagedOutputDirectory, staged_output_directory

__all__ = [
    "FEM_LABELS_MANIFEST_NAME",
    "PRODUCTION_PROFILE",
    "STANDARD_76_ELECTRODE_NAMES",
    "FEMLabelGenerationPlan",
    "FEMLabelsV1",
    "FEMV1ContractError",
    "FEMV1ProductionProfile",
    "LeadfieldRun",
    "SimNIBSAdapterError",
    "StagedOutputDirectory",
    "TDCSLeadfieldPlan",
    "build_fem_label_plan",
    "dry_run_fem_generation",
    "dry_run_fem_label_generation",
    "dry_run_tdcs_leadfield",
    "export_fem_labels_from_hdf5",
    "generate_fem_labels",
    "load_fem_labels_v1",
    "run_tdcs_leadfield",
    "staged_output_directory",
    "validate_fem_artifacts",
    "validate_fem_labels_manifest",
    "validate_leadfield_dataset_contract",
    "validate_production_options",
    "validate_tdcs_leadfield_run",
]
