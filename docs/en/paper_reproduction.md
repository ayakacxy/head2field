<a id="reproducibility-and-evaluation"></a>

# Evaluation and reproducibility

[English](paper_reproduction.md) · [简体中文](../zh/paper_reproduction.md)

Use this guide when setting up an experiment or reporting its results. It
covers data splits, training-derived statistics, FEM comparisons, and runtime
measurements. Commands are in [training](training.md), and prediction APIs are
in [inference](inference.md).

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

The paths above are relative to `src/head2field/`. The configuration files
are in `configs/` and are also included in the wheel.

## Training configuration and derived data

The real-data configuration is `configs/training_v1.json`; the synthetic
installation configuration is `configs/training_smoke_v1.json`. Both use the
same near, middle, and far regions:

```text
rho = min(distance(query, source), distance(query, Oz))
near: rho < 50 mm
middle: 50 mm <= rho < 100 mm
far: rho >= 100 mm
```

Compute energy statistics from each subject's complete FEM fields, before
splitting queries into chunks. The denominator floor uses the median across
that subject's valid source fields. Fit physics calibration using training
subjects only, then apply it unchanged to validation and test subjects.

Each training step uses the scale `M / (75 * N_train)`. Here, `M` counts all
source/query combinations in a complete training epoch, including the last
partial batches. Validation and test use the training partition's scale.
Recompute statistics if the radial regions or training split change.

The loaders check prepared subjects, FEM labels, target caches, and physics
samples for matching formats and numerical values. Continuing a saved run
requires the same configuration and dataset records. Loading inference
weights instead starts a new optimizer and training state.

## Dataset splits and evaluation

Assign each group to training, validation, or the optional test partition,
without overlap. Where the study design requires it, keep related participants
in the same group. Save group assignments, random seeds, preprocessing settings,
and weight checksums with the experiment. Obtain input datasets and participant
manifests separately.

Match prediction and FEM vectors by query row and source. Check that their
coordinate frame, current convention, and units agree before computing
complete-field metrics:

- Compute relative L2 error and mean direction cosine on matched vectors.
- Define high-field masks from FEM magnitudes and record percentile thresholds,
  including the rule for values equal to the threshold.
- Average source errors within each subject before averaging across subjects.
- For amplitude retention, compute source medians within each subject, then
  report the cohort median and subject P10–P90.

Training diagnostics from individual chunks are not complete-field metrics.
For every reported metric, state how values were aggregated and which mask
was used.

The package includes one pretrained state dictionary. To compare data splits,
training sizes, random seeds, ablations, or baseline models, train and evaluate
weights for each setting. The synthetic examples use a small model to check
the software; they do not measure the pretrained model's accuracy.

## Measuring runtime

Report the hardware, software versions, query count, chunk sizes, and execution
mode with each timing result. Separate input loading, preprocessing, model
loading, any compilation, warmup, inference, device-to-host transfer, and
output I/O.

Synchronize CUDA before and after GPU timing. Report cold starts separately
from repeated runs that reuse model or subject caches, and say whether output
writing and flushing are included. An end-to-end time should cover every stage
needed by the stated workflow. If the comparison concerns lead-field inference
alone, report that stage separately.

## Using the results

Use the predicted basis or Python query API to build field-synthesis and
evaluation applications. Current optimization, cohort statistics, and plotting
are handled by those applications. Save the basis manifest, source currents,
query coordinates, and weight filename and checksum alongside derived results.
The [model card](../MODEL_CARD.md) describes intended use and limitations;
references are in the [citation guide](../CITATION_GUIDE.md).
