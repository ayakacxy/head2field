# Changelog

[English](CHANGELOG.md) · [简体中文](CHANGELOG_zh.md)

Notable changes to the software are recorded here. The model-contract version
`v1` and software version are independent: a package update does not by itself
create a new trained model.

## 0.1.0rc1

### Documentation and code readability

- Provide paired English and Simplified Chinese versions of all public guides,
  with same-language navigation, per-page language switches, and bundled copies.
- Group technical guides in `docs/en/` and `docs/zh/` with matching filenames;
  route README reading links to the corresponding language guides.
- Move supporting project documents into `docs/`, retain the two root READMEs,
  and update navigation, package metadata, and bundled document copies.
- Document contributor attribution and the Codex co-author trailer.
- Add installed-wheel usage and code-reading guides, document chunking and
  source-current synthesis, and synchronize shipped technical documentation.
- Clarify core tensor/coordinate APIs and English diagnostics; isolate inference
  pair-bank remapping and clarify stable-energy loss variables without changing
  pretrained weights, model parameter names, or existing configuration values.

### Bundled pretrained model

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.

### Head2Field naming

- Package and import namespace: `head2field`; commands: `head2field-*`.
- Versioned model API: `Head2FieldV1`, `create_head2field_v1()`, and
  `load_head2field_v1_weights()` with `Head2FieldInferenceSession`.
- Model configuration: `configs/head2field_v1.json`; weight asset:
  `head2field_v1_weights.pth`. Profile, loss, architecture, operator, and
  prediction identifiers use the `head2field` prefix.

### Default training settings

- Default training profiles use 50/100 mm radial boundaries in configuration,
  statistics generation and batch consumers.
- Energy denominators use within-subject source medians, recorded in
  `training_loss_statistics_v2` manifests.
- Per-step supervision scaling is `M / (75 * N_train)`, including partial
  final batches, and is bound to the training run contract.
- Added method-to-code, experiment-scope and timing-provenance documentation;
  updated the manuscript title and citation metadata.

### Workflows

### Training

- CUDA FP16 AMP, AdamW with learning rate `2e-4` and weight decay `1e-4`,
  eight source-to-Oz pairs, and query chunks of 49,152 GM/WM locations.
- Complete source/query traversal per epoch with a maximum of 25 epochs and an
  epoch-based cosine schedule over 25 epochs.
- Epoch-end combined-objective validation with `a(t)=1`, five-epoch early
  stopping, best validation checkpoint export, and optional test evaluation.
- A separate CPU FP32 synthetic profile for installation and resume checks.

### Documentation and packaging

- English and Chinese READMEs, workflow guides, model/data-use documentation,
  contribution instructions, and a reusable release checklist.
- Deterministic synthetic examples, schemas, package resources, dependency
  notices, and implementation provenance.
- Bundled tensor-only pretrained weights; training-resume states and custom
  checkpoints excluded from Python distributions.
