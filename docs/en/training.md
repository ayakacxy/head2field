# Training

[English](training.md) · [简体中文](../zh/training.md)

The training profile provides the data-loss and all-physics GP method used by
Head2Field v1, a generic explicit dataset contract, preparation
commands, CUDA FP16 epoch training, deterministic validation, and complete local
resume state. It does not include research data, cohort membership, prepared
subjects, FEM labels, caches, fitted statistics, physics pools, training logs,
or training-resume checkpoints.

## Execution profiles

The commands below use the existing configuration files in `configs/`. For a
wheel-only installation, copy the bundled defaults as shown in
[getting started](getting_started.md#configuration-files-after-wheel-installation).

| Configuration | Use | Execution |
| --- | --- | --- |
| [training_v1.json](../../configs/training_v1.json) | Real-data training with the fixed v1 network | CUDA FP16 AMP, complete epochs, validation, and early stopping |
| [training_smoke_v1.json](../../configs/training_smoke_v1.json) | Installation and resume checks with synthetic inputs | Small test-only model, CPU FP32, two epochs |

## Training settings

The default real-data profile specifies:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Initial learning rate | `2e-4` |
| Weight decay | `1e-4` |
| Subject batch | One subject per step |
| Electrode batch | Eight source-to-Oz pairs |
| Query chunk | 49,152 GM/WM locations |
| Precision | CUDA automatic mixed precision with FP16 |
| Gradient norm clipping | `1.0`, after unscaling gradients |
| Maximum training budget | 25 epochs |
| Learning-rate schedule | Epoch-based cosine annealing over 25 epochs, minimum LR `0` |
| Validation | At each epoch end, combined objective with `a(t)=1` |
| Early stopping | Five consecutive epochs without a lower validation loss |
| Test checkpoint | Best checkpoint selected by validation loss |

Each epoch visits every training subject and every combination of electrode
batches and query chunks. Subject order, source order, and query order are
deterministically shuffled per epoch. Partial final batches and query chunks
are included, so all 75 source pairs and all GM/WM queries are covered. The
epoch length is the sum, over subjects, of
`ceil(75 / 8) * ceil(number_of_queries / 49152)` steps. The learning rate stays
constant within an epoch and advances only after the epoch finishes.

AMP uses an initial gradient scale of 1024 and a growth interval of 2000.
The synthetic installation profile uses an 8-cube test-only model, CPU FP32,
and two compact epochs; it is separate from the real-data training settings.

## Data and physics objective

The v1 GP objective uses:

- the fixed N128 GP model factory and 75 ordered source-to-Oz pairs;
- raw scalar conductivity in `S/m` on disk and the shared float32
  `sigma / 1.654` network transform;
- stable energy data loss with global, radial, and high-field terms weighted
  `0.60 / 0.30 / 0.10`;
- denominator floor ratio `0.01`, radial edges 50 and 100 mm, source-pair
  weighting `(alpha=1, sigma=40 mm)`, and peak quantile 0.90;
- circulation, WM/GM normal-current continuity, WM/GM tangential-field
  continuity, and flux operators, with 512 primitives per operator family;
- `one_sided_excess_v1`, a train-fitted FEM scale floor and MAD, weight 5 for
  each physical term, and a 0-to-1 step curriculum over 10,000 steps; and
- world-RAS vector components in `V/m` for source `+1 mA` and Oz `-1 mA`.

The formulas are implemented in
[losses.py](../../src/head2field/training/losses.py) and
[physics.py](../../src/head2field/training/physics.py). The model prediction and FEM field are evaluated by the
same discrete physical operators. Prediction values cannot alter validity
masks or fitted thresholds.

The per-step data-loss scale is `M / (75 * N_train)`, fixed across source
batches and query chunks, including partial tails. Validation and test reuse
the training-derived scale. The energy denominator median is computed across
valid source fields within each subject. See [evaluation and reproducibility](paper_reproduction.md)
for dataset splits, complete-field metrics, and runtime measurements.

## Real-data manifest

The machine schema is
[`training_dataset_v1.schema.json`](../../schemas/training_dataset_v1.schema.json).
Every path is relative to the dataset manifest, and every referenced manifest
has an explicit relative path. Each anonymous `subject_key` references:

1. `prepared_subject_v1.json`;
2. `fem_labels_v1.json`;
3. `training_loss_statistics_v2.json`; and
4. `training_physics_pool_v1.json`.

The loader revalidates the underlying prepared-subject and all 75 FEM label
files. It cross-checks source order, query order, conductivity-table ID/values,
the `sigma / 1.654` transform ID, array shapes, dtypes, finite values, and all
manifest/file sizes. Groups, rather than individual rows, own the `train`,
`validation`, or optional `test` partition. A group cannot occur in more than
one partition, and both train and validation are mandatory.

Changing the canonical training config, group split, any referenced asset path,
or source order changes the run contract. Such a state cannot be resumed.

## Asset preparation

All destinations below must not exist. Commands use staging directories and
refuse to overwrite prior outputs.

An optional query-major target cache can be generated without changing any FEM
label:

```bash
head2field-prepare-cache \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/cache-v1
```

Generate stable-energy statistics from the verified labels. Record each
statistics manifest path and file size in the dataset manifest:

```bash
head2field-prepare-statistics \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/loss-statistics-v1
```

Physics preparation is intentionally three-stage so validation FEM values can
never fit thresholds:

```bash
# 对每个训练和验证被试生成几何及原始 FEM 残差/尺度，不拟合阈值。
head2field-prepare-physics raw \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/physics-raw-v1

# 仅传入训练分区的原始池；支持重复指定该选项。
head2field-prepare-physics fit \
  --train-raw-pool derived/subject-a/physics-raw-v1/training_physics_raw_pool_v1.json \
  --train-raw-pool derived/subject-b/physics-raw-v1/training_physics_raw_pool_v1.json \
  --output-dir derived/train-only-physics-statistics-v1

# 将同一份训练集拟合统计应用于每个训练或验证原始池。
head2field-prepare-physics finalize \
  --raw-pool derived/subject-a/physics-raw-v1/training_physics_raw_pool_v1.json \
  --statistics derived/train-only-physics-statistics-v1/training_physics_statistics_v1.json \
  --output-dir derived/subject-a/physics-pool-v1
```

`fit` stores the paths, source lists and array sizes of its explicitly supplied training raw pools. A
validation raw pool must never be passed to that command.

## Synthetic installation and resume smoke

Generate a compact dataset containing one fictional training subject and one
fictional validation subject. It retains the real 75-source order but contains
only analytic arrays on an 8³ grid:

```bash
head2field-make-synthetic-training \
  --output-dir synthetic-training-v1

head2field-train-preflight \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset synthetic-training-v1/synthetic_training_dataset_v1.json

head2field-train \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset synthetic-training-v1/synthetic_training_dataset_v1.json \
  --output-dir runs/synthetic-smoke
```

This runs up to two CPU FP32 epochs through the model forward, stable-energy
loss, four physical operators, train-only state machine, and deterministic
validation. The small model is explicitly test-only and cannot accept official
N128 initialization weights.

## Real-data training

Run preflight before allocating a new output directory:

```bash
head2field-train-preflight \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json

head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --output-dir runs/custom-cuda-run
```

Use `--init-weights` only for a bare `str -> FP32 Tensor` state dictionary. For
the fixed N128 model, the CLI additionally enforces the official 208-tensor and
2,607,870-value schema before initialization. Optimizer, scheduler, step,
sampler, and all RNGs start fresh.

Use `--resume-state` only for a complete state created by this training code.
Because PyTorch complete states use pickle, the CLI also requires
`--trust-local-resume-state`; never enable it for an untrusted download. Resume
always writes a new output directory:

```bash
head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --resume-state runs/part-a/train_state_step_000005000.pt \
  --trust-local-resume-state \
  --output-dir runs/part-b
```

The complete local state contains model, AdamW, cosine scheduler, AMP scaler,
global step, Python RNG, NumPy RNG, Torch CPU/CUDA RNGs, named
augmentation/dropout generators, sampler permutations/cursors, best validation
loss and model, early-stopping counter, the exact run contract, and all prior
log rows. State files are written to a same-directory temporary file, flushed, and atomically
promoted without overwriting an existing file. They are local artifacts and
must never be placed in `src/head2field/resources/weights/`.

The CPU deterministic regression test compares uninterrupted `N` steps with `K`
steps plus save/reload plus continuation to `N`. Model tensors, optimizer,
scheduler, every RNG, sampler, early-stopping state, and canonical JSONL log are
exactly equal. CUDA state is persisted as well; bitwise CUDA equivalence is not
claimed by the CPU test.

## Validation and outputs

Validation traverses all validation subjects and source/query combinations in
fixed order, with `model.eval()`, deterministic branches, and no training
generator. It averages the combined data and physics objective by subject,
always using `a(t)=1`. Epoch-end validation metrics, learning rates, and the
early-stopping counter are embedded in the step log. Strictly lower validation
loss resets the counter and selects a new best checkpoint; five consecutive
non-improving epochs stop training.

The training output is a new directory containing:

```text
training_log_v1.jsonl
best_model.pt
train_state_step_000000001.pt
train_state_step_000005000.pt
...
```

`best_model.pt` is a bare model state dictionary selected by validation loss,
and the returned in-memory model is restored to it for test evaluation. Resume
states retain the latest training model and complete optimizer/scheduler state.
An interrupted run exports the best checkpoint among its completed validation
epochs. If the real-data manifest contains a `test` partition, training evaluates
that partition once after reaching the epoch budget or early stopping, using
the restored best model, and writes `test_metrics_v1.json`. A deliberate
mid-training interruption does not trigger test evaluation.

These outputs are custom model artifacts. Record their configuration, data
split, and evaluation results before using them for inference. Participant data
are supplied separately.
