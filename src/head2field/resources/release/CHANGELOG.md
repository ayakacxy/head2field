# Changelog

[English](CHANGELOG.md) · [简体中文](CHANGELOG_zh.md)

Software and model versions are tracked separately.

## Unreleased

- Organize user guides around installation, input preparation, training,
  inference, evaluation, and runtime measurement.
- Clarify weight compatibility, complete the workflow descriptions, and keep
  English, Chinese, and packaged documents consistent.

## 0.1.0

### Installation and distributions

- Python package `head2field`, nine workflow CLI entry points, and optional
  preprocessing/FEM dependencies.
- Installable wheel, source archive, and `SHA256SUMS` under tag `v0.1.0`.
- Paired English and Simplified Chinese READMEs, technical guides, model and
  data-use documentation, examples, and packaged configuration/schema resources.

### Bundled pretrained model

- Tensor-only Head2Field v1 weights and checksums are included in both
  distributions and loaded by default.
- `--weights` selects compatible custom weights.
- The fixed v1 factory and strict FP32 state-dictionary loader validate
  parameter keys, tensor shapes, and value counts.

### Workflows

- T1 preparation, external CHARM head modeling, scalar conductivity, GM/WM
  query assembly, and standard cap handling.
- Oz-referenced SimNIBS lead-field generation, solver-output validation, and
  ordered FEM label export.
- Chunked lead-field inference, subject-feature caching, current-pattern
  synthesis, and output manifests.
- Dataset, cache, statistics, and physics-pool preparation with validated
  coordinate, source-order, and unit checks.

### Training

- CUDA FP16 AMP and AdamW with initial learning rate `2e-4`, weight decay
  `1e-4`, eight source-to-Oz pairs, and 49,152 queries per step.
- Stable-energy supervision with 50/100 mm radial regions and
  `training_loss_statistics_v2`; global/radial/high-field weights 0.60/0.30/0.10.
- Integral physics losses with training-only calibration and four weights of 5.
- Complete epochs, a maximum 25-epoch budget, cosine scheduling, validation,
  five-epoch early stopping, and validation-best model selection.
- Complete local resume states and optional final test evaluation.
- A separate small CPU FP32 synthetic profile for installation and resume checks.

### Compatibility and project resources

- `Head2FieldV1`, `create_head2field_v1()`, `load_head2field_v1_weights()`, and
  `Head2FieldInferenceSession` provide the v1 model API.
- The model uses isotropic scalar conductivity and outputs `[75, Q, 3]`
  world-RAS vectors in V/m for source +1 mA and Oz −1 mA.
- Participant data, generated research arrays, custom checkpoints, and complete
  training-resume states are excluded from the software distributions.
