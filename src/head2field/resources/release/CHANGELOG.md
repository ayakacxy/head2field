# Changelog

[English](CHANGELOG.md) · [简体中文](CHANGELOG_zh.md)

Software and model versions are recorded separately.

## Unreleased

- Update the manuscript title in both languages, citation metadata, and the
  package description to match the current manuscript.
- Report cohort results, TIS FEM re-evaluation, electrode-position comparisons,
  and benchmark conditions; align the model card and packaged documentation.
- Reorganize the guides around installation, input preparation, training,
  inference, evaluation, and runtime measurement.
- Clarify weight compatibility and workflow instructions, and synchronize
  English, Chinese, and packaged documentation.

## 0.1.0

### Installation and distributions

- Added the `head2field` Python package, nine workflow CLI entry points,
  and optional preprocessing/FEM dependencies.
- Published an installable wheel, source archive, and `SHA256SUMS` under
  tag `v0.1.0`.
- Included paired English and Simplified Chinese READMEs, technical guides,
  model and data-use documentation, examples, and configuration/schema resources.

### Bundled pretrained model

- Included tensor-only Head2Field v1 weights and checksums in both
  distributions, with automatic loading by default.
- Added `--weights` for compatible custom weights.
- Added the fixed v1 factory and strict FP32 state-dictionary loading,
  checking parameter names, tensor shapes, and value counts.

### Workflows

- Added T1 preparation, external CHARM head modeling, scalar conductivity,
  GM/WM query assembly, and standard cap handling.
- Added Oz-referenced SimNIBS lead-field generation, solver-output validation,
  and ordered FEM label export.
- Added chunked inference, subject-feature caching, current-pattern synthesis,
  and output manifests.
- Added dataset, cache, statistics, and physics-pool preparation with checks
  for coordinates, source order, and units.

### Training

- Added CUDA FP16 AMP and AdamW with initial learning rate `2e-4`,
  weight decay `1e-4`, eight source-to-Oz pairs, and 49,152 queries per step.
- Added stable-energy supervision with 50/100 mm radial regions,
  `training_loss_statistics_v2`, and global/radial/high-field weights
  of 0.60/0.30/0.10.
- Added integral physics losses calibrated only on training data, with four
  loss weights of 5.
- Added complete epochs, a maximum 25-epoch budget, cosine scheduling,
  validation, five-epoch early stopping, and validation-best model selection.
- Added complete local resume states and optional final test evaluation.
- Included a small CPU FP32 synthetic profile for installation and resume checks.

### Compatibility and project resources

- Exposed `Head2FieldV1`, `create_head2field_v1()`,
  `load_head2field_v1_weights()`, and `Head2FieldInferenceSession`.
- Used isotropic scalar conductivity and `[75, Q, 3]` world-RAS output vectors
  in V/m for source +1 mA and Oz −1 mA.
- Excluded participant data, generated research arrays, custom checkpoints,
  and complete training-resume states from software distributions.
