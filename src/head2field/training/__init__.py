"""Explicit public API for the training subpackage."""

from .contracts import (
    TrainingConfig,
    TrainingContractError,
    TrainingDatasetManifest,
    build_run_contract,
    load_training_dataset_manifest,
)
from .dataset import (
    DeterministicHead2FieldSampler,
    Head2FieldTrainingDataset,
)
from .runner import LoopConfig, run_training_loop
from .steps import build_n128_gp_step_function
from .synthetic import (
    SyntheticHead2FieldDataset,
    create_synthetic_smoke_model,
    load_synthetic_training_dataset,
    write_synthetic_training_dataset,
)

__all__ = [
    "DeterministicHead2FieldSampler",
    "Head2FieldTrainingDataset",
    "LoopConfig",
    "SyntheticHead2FieldDataset",
    "TrainingConfig",
    "TrainingContractError",
    "TrainingDatasetManifest",
    "build_n128_gp_step_function",
    "build_run_contract",
    "create_synthetic_smoke_model",
    "load_synthetic_training_dataset",
    "load_training_dataset_manifest",
    "run_training_loop",
    "write_synthetic_training_dataset",
]
