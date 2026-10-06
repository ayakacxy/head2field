<a id="reproducibility-and-evaluation"></a>

# Evaluation and reproducibility

[English](paper_reproduction.md) · [简体中文](../zh/paper_reproduction.md)

This guide explains how to prepare repeatable training runs, compare predicted
fields with FEM references, and measure runtime. Start with
[training](training.md) for commands and [inference](inference.md) for the
prediction API.

## Workflow and implementation

| Stage | Implementation | Configuration or output |
| --- | --- | --- |
| T1 preparation | `preprocessing/`, `inputs.py` | `preprocessing_v1.json`; 256³, 1 mm RAS grid |
| Conductivity and FEM labels | `preprocessing/conductivity.py`, `fem_labels/` | `fem_labels_v1.json`; 75 source-to-Oz fields for 1 mA |
| Anatomical encoding and field prediction | `model.py`, `_spec.py` | `head2field_v1.json`; fixed v1 architecture |
| Energy-normalized supervision | `training/cache.py`, `training/dataset.py`, `training/losses.py` | Full-field statistics; 50/100 mm regions; weights 0.60/0.30/0.10 |
| Integral physics losses | `training/physics.py`, `training/physics_data.py`, `training/steps.py` | Training-only calibration; four weights of 5; 10,000-step ramp |
| Epoch training and validation | `training/runner.py`, `training/validation.py`, `training/cli.py` | AdamW; up to 25 epochs; cosine scheduling; patience 5 |
| Complete-basis inference | `inference.py`, `cli.py` | Ordered `[75, Q, 3]` float32 output in world-RAS, V/m |

Implementation paths are relative to `src/head2field/`. Configuration files
are in `configs/`; matching copies are included in the wheel.

## Training configuration and derived data

Use `configs/training_v1.json` for real-data training and
`configs/training_smoke_v1.json` for synthetic installation checks. Both define
the following near/middle/far regions:

```text
rho = min(distance(query, source), distance(query, Oz))
near: rho < 50 mm
middle: 50 mm <= rho < 100 mm
far: rho >= 100 mm
```

Energy statistics use each subject's complete FEM fields before query chunking.
The denominator floor uses the median across that subject's valid source fields.
Fit physics calibration on training subjects only, then apply the same
calibration to validation and test subjects.

The per-step data-loss scale is `M / (75 * N_train)`, where `M` counts every
source/query combination in a complete training epoch, including partial final
batches. Validation and test use the scale derived from the training partition.
Regenerate statistics when changing the regional settings or training split.

Prepared subjects, FEM labels, target caches, and physics pools have explicit
validation contracts. Resume requires the saved configuration and dataset
contract to match the new run. Initializing from an inference state dictionary
starts a fresh optimizer and training state.

## Dataset splits and evaluation

Create a dataset manifest with disjoint training, validation, and optional test
groups. Put related participants in the same group when the study design
requires it. Keep group assignments, random seeds, preprocessing settings, and
weight checksums with the experiment record. Input datasets and participant
manifests are supplied by the user.

Compare prediction and FEM vectors at identical query rows, in the same
source order, coordinate frame, current convention, and units. For complete-field
evaluation:

- Compute relative L2 error and mean direction cosine on matched vectors.
- Define high-field masks from FEM magnitudes and record percentile thresholds,
  including the rule for values equal to the threshold.
- Average source errors within each subject before averaging across subjects.
- For amplitude retention, compute source medians within each subject, then
  report the cohort median and subject P10–P90.

Chunkwise training diagnostics and complete-field evaluation summarize different
quantities. Record the aggregation and masks used for each reported metric.

The package supplies one pretrained v1 state dictionary. Comparisons involving
other data splits, training sizes, seeds, ablations, or baseline models need
separately trained and evaluated weights. Synthetic examples check software
operation with a small test model; they do not measure pretrained-model accuracy.

## Measuring runtime

Record the hardware, software versions, query count, chunk sizes, and execution
mode. Measure input loading, preprocessing, model loading, compilation when
used, warmup, inference, device-to-host transfer, and output I/O separately.

Use CUDA synchronization at measurement boundaries for GPU execution. Distinguish
cold starts from repeated runs with cached model or subject features. State
whether output writing and flushing are included. End-to-end measurements should
include every stage needed by the intended workflow; report lead-field inference
separately when comparing that stage alone.

## Using the results

The reusable basis and Python query API support application-specific field
synthesis and evaluation. Current optimization, cohort statistical analysis,
and figure generation are separate application workflows. Preserve the basis
manifest, source currents, query coordinates, and weight identity with derived
results. See the [model card](../MODEL_CARD.md) for intended use and limitations,
and the [citation guide](../CITATION_GUIDE.md) for references.
