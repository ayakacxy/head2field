# Source provenance

[English](SOURCE_PROVENANCE.md) · [简体中文](SOURCE_PROVENANCE_zh.md)

Head2Field extracts and adapts model, preprocessing, FEM-label,
and training components from its parent research project. The tables below
record repository-relative origins and the main adaptation boundaries.
Parent-project paths preserve the source names for traceability; the public
package, Python API, commands, and model identifiers use Head2Field naming.

The default training configs, statistics producer/consumer and synthetic
batches use 50/100 mm radial regions and per-step supervision scaling
`M / (75 * N_train)`. Statistics and run contracts bind these settings to each
training run. Inference uses the fixed v1 network and tensor-only weight schema.
See [reproducibility and evaluation](en/paper_reproduction.md) for the method
map, experimental scope, and weight/performance provenance.

The source and Python distributions contain project code, documentation, and
the verified tensor-only pretrained v1 weights.
They do not vendor external runtimes, native binaries, templates, atlases, or
research datasets. Dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Files with preserved executable logic

| Project file | Parent-project source | Scope |
| --- | --- | --- |
| `src/head2field/model.py` | `head2field/models/head2field.py` | Preserved numerical network; Head2Field v1 identifiers and English comments/docstrings |
| `src/head2field/fem_labels/transaction.py` | `head2field/fem_labels/transaction.py` | Atomic directory publication helper |

These files retain the numerical and transaction logic of their listed parent
sources. Comments and docstrings use English. The network class and contract
identifiers follow Head2Field naming; tensor keys and shapes retain the v1
weight format.

## Adapted inference core

| Project file | Parent-project source | Adaptation boundary |
| --- | --- | --- |
| `src/head2field/_spec.py` | `head2field/config/spec.py` | Package-local model import |
| `src/head2field/inputs.py` | `head2field/data/model_inputs.py` | Package-local model and preprocessing imports |
| `src/head2field/inference.py` | `head2field/evaluation/inference.py` | Package-local inputs and model imports |
| `src/head2field/weights.py` | `head2field/weights.py` | Package-local factory and model imports |
| `src/head2field/cli.py` | `head2field/cli/predict_lead_field.py` | Removed environment-specific project-root discovery; retained input-read-only and new-output-directory safeguards |

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.

## Adapted preprocessing implementation

| Project file | Parent-project source | Adaptation boundary |
| --- | --- | --- |
| `src/head2field/preprocessing/alignment.py` | `scripts/preprocessing/align_t1_rigid.py` | Parameterized HCP bypass and non-HCP 6-DOF runner |
| `src/head2field/preprocessing/pipeline.py` | `head2field/preprocessing/pipeline_v1.py` | Package-local adapters, explicit profiles, and transactional output |
| `src/head2field/preprocessing/conductivity.py` | `head2field/preprocessing/conductivity.py` | Lazy NIfTI dependency import for minimal inference installations |
| `src/head2field/preprocessing/validation.py` | `head2field/preprocessing/prepared_subject_v1.py` | Package layout and shared conductivity contract |
| `src/head2field/preprocessing/transaction.py` | `head2field/fem_labels/transaction.py` | Adapted transaction pattern for prepared-subject output |
| `src/head2field/preprocessing/charm_adapter.py` | `scripts/preprocessing/build_head_model.py` | External SimNIBS invocation isolated behind an adapter |

## Adapted FEM-label implementation

| Project file | Parent-project source | Adaptation boundary |
| --- | --- | --- |
| `src/head2field/fem_labels/contracts.py` | `head2field/fem_labels/contracts.py` | Public validation contract and package-local imports |
| `src/head2field/fem_labels/simnibs_adapter.py` | `head2field/fem_labels/simnibs_adapter.py` | External installation boundary and explicit failures |
| `src/head2field/fem_labels/leadfield_run.py` | `scripts/simulation/run_oz_reference_leadfield.py` | Parameterized source-to-Oz orchestration and provenance |
| `src/head2field/fem_labels/audit.py` | Project `leadfield_run.py` and `generation.py` validation code | Isolated schema, path-safety, and optional full-metadata audit |

## Adapted training implementation

| Project file | Parent-project source | Adaptation boundary |
| --- | --- | --- |
| `src/head2field/training/contracts.py` | `head2field/training/contracts.py` | Standalone public dataset and resume contracts |
| `src/head2field/training/losses.py` | `head2field/training/losses.py` | Package-local imports and frozen public objective |
| `src/head2field/training/physics.py` | `head2field/physics/losses.py` | Public N128 physics-loss implementation |
| `src/head2field/training/state.py` | `head2field/training/runner.py` | Extracted RNG and checkpoint semantics; AMP, best-model, and early-stopping state |
| `src/head2field/training/runner.py` | `head2field/training/runner.py` | CUDA FP16 public runner with epoch scheduling, early stopping, best-model test evaluation, and local resume |
| `src/head2field/training/dataset.py` | `head2field/data/head2field.py` | Prepared-subject and FEM-label consumer contract; complete epoch traversal |
| `src/head2field/training/steps.py` | Parent-project training flow | Public orchestration over package-local modules |
| `src/head2field/training/cache.py` | `scripts/training/prepare_training_cache.py` and the predictor loss-statistics script | Parameterized cache and statistics stages |
| `src/head2field/training/physics_data.py` | `head2field/data/physics_primitives.py` | Public physics-primitives and statistics contract |
| `src/head2field/training/validation.py` | Parent-project validation concept | Complete deterministic validation with a(t)=1 |
| `src/head2field/training/cli.py` | Parent-project train and preflight CLIs | Reimplemented public commands and strict argument separation |

## Project-authored implementation

The following source files were written for this package:

- package facade and helpers: `src/head2field/__init__.py`,
  `geometry.py`, and `resource_paths.py`;
- preprocessing facade and composition:
  `preprocessing/{__init__,__main__,assembly,cli,electrodes,synthetic}.py`;
- FEM-label facade and composition:
  `fem_labels/{__init__,__main__,cli,generation}.py`;
- training facade and synthetic support:
  `training/{__init__,__main__,synthetic}.py`.

All files under `tests/` and `examples/`, all package metadata, top-level
configs and schemas, package resource copies, and release documentation were
authored for the package. The tests and examples use only fictional
identifiers and generated arrays.

## Explicitly excluded

The whitelist excludes research datasets, full checkpoints and resume states,
FEM outputs, training caches, analysis results, optional performance backends,
stimulation-planning code, third-party binaries, templates, atlases, and
archived experiment material. The allowlisted pretrained v1 weights use a bare
`str -> Tensor` state dictionary in `src/head2field/resources/weights/` and are
included in the wheel and source distribution.
