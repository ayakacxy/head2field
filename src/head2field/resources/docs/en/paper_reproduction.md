# Reproducibility and evaluation

[English](paper_reproduction.md) · [简体中文](../zh/paper_reproduction.md)

Head2Field is described in:

> **Head2Field: A Continuous Neural Field for Subject-Specific Multi-Electrode
> Transcranial Electrical Stimulation Lead-Field Prediction from Structural
> MRI**

Publication identifiers should be added to `CITATION.cff` when available.

## Method-to-code map

| Manuscript | Implementation | Configuration / contract |
| --- | --- | --- |
| Section 2.2.1: 256-cube, 1 mm RAS T1 preparation | `preprocessing/`, `inputs.py` | `preprocessing_v1.json`; HCP ACPC inputs bypass rigid estimation |
| Sections 2.2.2–2.2.3: scalar conductivity and 75 source-to-Oz FEM fields | `preprocessing/conductivity.py`, `fem_labels/` | `fem_labels_v1.json`; 76 cap electrodes; 1 mA; world-RAS |
| Section 2.3: CNN/FPN, axial attention, query/electrode tokens and shared responses | `model.py`, `_spec.py` | `head2field_v1.json`; fixed v1 weight schema |
| Equations 7–9: energy-normalized supervision | `training/cache.py`, `dataset.py`, `losses.py` | Existing training configs; 50/100 mm radial boundaries; weights 0.60/0.30/0.10 |
| Equations 10–13: FEM-calibrated integral physics | `training/physics.py`, `physics_data.py`, `steps.py` | Four weights of 5; Huber delta 0.1; kappa 1; 10,000-step ramp |
| Section 2.5.2: complete epochs, validation and early stopping | `training/runner.py`, `validation.py`, `cli.py` | AdamW; 25 epochs; cosine horizon 25; patience 5; seed 42 |
| Cached complete-basis inference | `inference.py`, `cli.py` | Ordered `[75, Q, 3]` float32 output in V/m for 1 mA |

Paths in the implementation column are relative to `src/head2field/`.
Configuration files are in `configs/`; identical copies ship in the wheel.

## Default training configuration and derived assets

Use `configs/training_v1.json` for real-data training and
`configs/training_smoke_v1.json` for synthetic installation checks. Both use
the following near/middle/far regions:

```text
rho = min(distance(query, source), distance(query, Oz))
near: rho < 50 mm
middle: 50 mm <= rho < 100 mm
far: rho >= 100 mm
```

Statistics are computed from each subject's complete FEM fields, before query
chunking. The energy floor uses the median across that subject's valid source
fields, rather than fitting a cross-subject statistic. Physics calibration uses
training subjects only, then stays frozen for validation and test subjects.
The per-step data-loss scale is `M / (75 * N_train)`, where `M` includes every
source/query combination and partial final batches in a complete training epoch.
The same training-derived scale is used for validation and test objectives.

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.

Prepared subjects, FEM labels, target-array caches and physics pools carry
their own explicit contracts. Complete resume states require an identical
configuration and run contract. An inference-only state dict initializes the
fixed v1 network with a fresh optimizer and training state.

## Experimental reproduction scope

The manuscript uses 200 HCP subjects and a family-disjoint 128/32/40 split
(seed 42), with nested training sizes 10, 20, 40, 80 and 128. Recreate the split
from authorized data access; participant/family manifests are excluded from
the source release. The training command accepts a group-disjoint manifest,
but does not reconstruct the manuscript's private family assignments.

For Equations 14–16, compute relative L2 and mean direction cosine on matched
world-RAS vectors. Define high-field masks from FEM magnitudes, with inclusive
percentile thresholds. Average source errors within each subject before cohort
averaging. Amplitude retention uses within-subject source medians followed by
the cohort median and subject P10–P90. Chunkwise relative L2 means are training
diagnostics; they are not a substitute for these complete-field paper metrics.

The manuscript's experiments use different model assets: the tDCS comparison
uses the 80-subject model, while complete-basis, TIS and displaced-electrode
experiments use the 128-subject model. The single verified state dict does not
supply every training scale, the data-only ablation, or a comparator model.

The release provides the reusable basis and coordinate-query inference API.
Montage comparisons, the TIS optimizer, displaced-position FEM generation,
cohort statistical tests, and figure-generation pipelines are research
workflows outside this source package. Their results cannot be reproduced by
running the synthetic example or the training command alone.

## Weight and performance provenance

The manuscript reports 8.928 s including device-to-host transfer, and 12.073 s
including float32 output writing/flushing, at 1,392,017 GM/WM queries on one
representative subject. It used an RTX 2080 Ti, warmup and PyTorch compilation.
The approximate 100-fold speedup concerns lead-field generation; the estimated
4.7-fold end-to-end speedup also includes about 20 minutes for segmentation.
These are manuscript measurements, not benchmarks of the default eager CLI.
Record input loading, preprocessing, model loading, compilation, warmup,
transfer and output I/O separately when measuring another deployment.
