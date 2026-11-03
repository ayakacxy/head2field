<a id="reproducibility-and-evaluation"></a>

# Evaluation and reproducibility

[English](paper_reproduction.md) · [简体中文](../zh/paper_reproduction.md)

Use this guide when setting up an experiment or reporting its results. It
covers data splits, training-derived statistics, FEM comparisons, and runtime
measurements. Commands are in [training](training.md), and prediction APIs are
in [inference](inference.md).

## Manuscript results

The current [manuscript](../CITATION_GUIDE.md) evaluates 200 healthy adults from
WU-Minn HCP, aged 22–36 years, from 163 families. Family-grouped splitting
(seed 42) gives 128 training, 32 validation, and 40 test subjects. Nested
training subsets contain 10, 20, 40, 80, and 128 subjects, with fixed validation
and test sets. Each subject has 75 source-to-Oz fields evaluated over GM/WM.

| Experiment | Reported result |
| --- | --- |
| Training cohort size | Mean full-field relative L2 errors: 0.118, 0.098, 0.082, 0.078, and 0.070 for 10, 20, 40, 80, and 128 training subjects |
| Physics regularization | Full-field error: 0.078 without regularization and 0.070 with it; high-field error: 0.080 and 0.071; lower errors in 39/40 and 38/40 subjects, respectively |
| High-field amplitude | Cohort median retention: 0.992, 0.991, 0.986, 0.984, and 0.978 at FEM magnitude percentiles 90, 95, 99, 99.5, and 99.9 |
| tDCS synthesis | Mean full-field errors on trained/held-out montage sets: 0.077/0.079 for Head2Field and 0.243/0.818 for DeeptDCS |
| TIS envelopes | Mean relative L2 error 0.054 and voxelwise Pearson correlation 0.991 across 3,200 configurations |
| TIS planning | Under FEM evaluation, mean H-WMFI was 1.253 for FEM-optimized plans and 1.192 for Head2Field-optimized plans; geometric-mean focality retention was 94.9% |
| Unseen electrode positions | Mean full-field errors of 0.069–0.070 at 5, 10, 15, and 20 mm displacements, without fine-tuning |

### tDCS comparison

Both models used 80 training, 32 validation, and 40 test subjects. Head2Field
learned 75 source-to-reference fields per training subject; DeeptDCS learned
five montage-specific fields per subject. Trained and held-out refer to
inclusion in DeeptDCS supervision. Each set contains five montages, and neither
model underwent montage-specific fine-tuning for this comparison. The reported
errors describe the methods under their respective supervision and loss schemes.

### TIS synthesis and planning

Envelope synthesis used 80 configurations per test subject: 16 each with
4, 8, 16, 32, or 64 active electrodes and 1 mA total positive current per carrier.
Planning used a 5-mm-radius left internal globus pallidus target restricted
to GM/WM and a cortical/cerebellar GM/WM avoidance region excluding the target.
The optimizer selected 16 of 75 candidate electrodes, using 1,000 and
1,130 Hz carriers, independent current balance for each carrier, at most
2 mA absolute current per electrode, and at most 10 mA summed absolute current
across both carriers. Both bases used 3,000 optimization epochs and seed 42.

H-WMFI is the target-to-avoidance amplitude ratio. The target term averages
the highest decile of target envelope amplitudes. The avoidance term excludes
values above the avoidance region's 99.99th percentile, then averages retained
values at or above 0.9 times their remaining maximum.

Each optimized montage was evaluated on both bases with its electrodes and
currents fixed. Under FEM re-evaluation, Head2Field-optimized plans scored
0.061 lower on average than FEM-optimized plans (95% CI 0.031–0.091;
Holm-adjusted p = 0.001). The 94.9% result is the geometric mean of subject-level
H-WMFI ratios, with both plans evaluated using FEM. On their respective
optimization bases, the mean scores were 1.253 and 1.233 (p = 0.256).
These observations support rapid candidate search using the predicted basis
and FEM evaluation of selected plans.

### Electrode-position generalization

The 40 test subjects each contributed positions around five anchors
(F3, C4, P3, T8, and PO7), in four directions and at four distances, yielding
3,200 independent FEM configurations. Reusing the anchor prediction increased
mean full-field error from 0.095 at 5 mm to 0.273 at 20 mm. Querying the actual
coordinates gave errors of 0.069–0.070 and improved all 40 subjects at each
distance. Direct prediction also had lower errors than interpolation of
standard-position model predictions at all distances. Nearest-neighbor and
geodesic FEM-library baselines had lower errors at 5 mm; Head2Field had lower
mean errors than all three FEM-library baselines at 10–20 mm.

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

The manuscript's representative benchmark used one RTX 2080 Ti and the
128-subject model after warmup, with PyTorch compilation enabled. Predicting
the 75-column basis at 1,392,017 GM/WM voxels took 8.928 s including
device-to-host transfer, or 12.073 s including float32 output writing and
flushing (0.201 min).

| Timing scope | SimNIBS | Head2Field | Speedup |
| --- | --- | --- | --- |
| Lead-field generation, inputs prepared | Approximately 20 min | 0.201 min | Approximately 100 times |
| Estimated end-to-end workflow | Approximately 94 min | Approximately 20 min | Approximately 4.7 times |

The Head2Field end-to-end estimate adds approximately 20 min of segmentation
to basis generation, rounded to the nearest minute. The release prediction
CLI uses FP32 evaluation without enabling PyTorch compilation. Its preprocessing
command runs the full SimNIBS `charm` head-model workflow, including surface
reconstruction and meshing; prediction itself uses the prepared voxel inputs
and electrode coordinates. Record these stages separately when timing the
release commands.

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
