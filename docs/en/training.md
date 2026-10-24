# Training

[English](training.md) · [简体中文](../zh/training.md)

Training starts from prepared subjects and FEM labels. The steps below cover
loss statistics, physics samples, CUDA FP16 training, validation, and resuming an
interrupted run. The training objective combines field errors with physical
constraints; checkpoints and logs are saved in the output directory.

## Execution profiles

Choose the real-data configuration for a full model or the synthetic
configuration to check the installation. The examples use files in `configs/`.
After a wheel-only installation, [copy the bundled defaults](getting_started.md#configuration-files-after-wheel-installation)
into your working directory first.

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

An epoch covers every training subject, all 75 source pairs, and all GM/WM
query points. The sampler shuffles subjects, sources, and queries using the
epoch seed, and includes the final partial batches. For each subject, it takes
`ceil(75 / 8) * ceil(number_of_queries / 49152)` steps; the epoch length is the
sum across subjects. The learning rate stays constant during an epoch and
changes after that epoch finishes.

AMP starts with a gradient scale of 1024 and a growth interval of 2000.
The synthetic profile instead runs an 8-cube model on CPU in FP32 for two
short epochs. It is an installation check, not the full training configuration.

## Data and physics objective

The default GP configuration uses the following data and physics settings:

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

The data-loss formulas are in [losses.py](../../src/head2field/training/losses.py),
and the physical operators are in [physics.py](../../src/head2field/training/physics.py).
The same operators evaluate predicted and FEM fields. Sample-validity masks
and fitted thresholds come from the prepared data and stay fixed during training.

Each step uses the same data-loss scale, `M / (75 * N_train)`, including
partial source batches and query chunks. Validation and test use the scale
computed from the training partition. The energy denominator uses the median
across valid source fields within each subject. See [evaluation and reproducibility](paper_reproduction.md)
for data splits, complete-field metrics, and timing methods.

## Real-data manifest

Describe the dataset in a manifest that follows
[`training_dataset_v1.schema.json`](../../schemas/training_dataset_v1.schema.json).
Use paths relative to that manifest. Each anonymous `subject_key` points to
four files:

1. `prepared_subject_v1.json`;
2. `fem_labels_v1.json`;
3. `training_loss_statistics_v2.json`; and
4. `training_physics_pool_v1.json`.

When loading a dataset, the program checks the prepared subject and all 75
FEM label files. Source and query order, conductivity-table ID and values,
the `sigma / 1.654` transform ID, shapes, dtypes, finite values, and recorded
file sizes must agree. Assign each group to only one of `train`, `validation`,
or the optional `test` partition. Both training and validation groups are required.

To resume a run, keep its configuration, group assignments, file paths, and
source order unchanged. Start a new run if any of these settings change.

## Training data preparation

Use a new destination for each command below. Preparation writes to a staging
directory and refuses to overwrite existing results.

A query-major target cache is optional. It reorganizes reads for training and
leaves the FEM labels unchanged:

```bash
head2field-prepare-cache \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/cache-v1
```

Next, compute stable-energy statistics from the checked FEM labels. Add the
resulting statistics-manifest path and file size to the dataset manifest:

```bash
head2field-prepare-statistics \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/loss-statistics-v1
```

Prepare physics samples in three stages: build raw pools for each subject,
fit calibration on the training pools, then apply that calibration to every
pool. Keeping the fit step separate prevents validation FEM values from being
used to set thresholds:

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

Pass only training raw pools to `fit`. It records their paths, source lists,
and array sizes; validation pools belong only in the `raw` and `finalize` stages.

## Synthetic installation and resume smoke

For an installation check, generate one fictional training subject and one
fictional validation subject. These analytic 8³ arrays use the same 75-source
order as real inputs:

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

The example runs up to two CPU FP32 epochs, including forward evaluation,
stable-energy loss, four physical operators, training-state updates, and
validation. This small model cannot load the full N128 initialization weights.
Its results check software operation, not prediction accuracy.

## Real-data training

Check the configuration and dataset before starting training:

```bash
head2field-train-preflight \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json

head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --output-dir runs/custom-cuda-run
```

To start a new run from existing model weights, use `--init-weights` with a
bare `str -> FP32 Tensor` state dictionary. The N128 model requires exactly
208 tensors and 2,607,870 values with matching parameter names and shapes.
The optimizer, scheduler, step count, sampler, and all random-number generators
start from their initial states.

To continue an interrupted run, use `--resume-state` with a complete state
saved by this training code. These files use pickle, so the CLI also requires
`--trust-local-resume-state`; use it only for a state file you trust. Write the
continued run to a new output directory:

```bash
head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --resume-state runs/part-a/train_state_step_000005000.pt \
  --trust-local-resume-state \
  --output-dir runs/part-b
```

A resume state contains the model, AdamW optimizer, cosine scheduler, AMP
scaler, global step, and sampler permutations and cursors. It also saves the
Python, NumPy, Torch CPU/CUDA, and named augmentation/dropout random-number
states, along with the best validation loss and model, early-stopping counter,
run settings, and previous log rows.

The program flushes a temporary file in the destination directory before
atomically saving the state, without overwriting an existing file. Keep these
local training files out of `src/head2field/resources/weights/`.

The CPU resume regression compares an uninterrupted `N`-step run with a run
saved after `K` steps and continued to `N`. Model tensors, optimizer, scheduler,
random-number states, sampler, early-stopping state, and canonical JSONL logs
match exactly. CUDA state is saved too, but the CPU check does not establish
bitwise reproducibility on a GPU.

## Validation and outputs

At the end of each epoch, validation visits every validation subject and all
source/query combinations in a fixed order. It uses `model.eval()`,
deterministic branches, and no training random-number generator. The combined
data and physics objective is averaged by subject with `a(t)=1`.

The step log records validation metrics, learning rates, and the early-stopping
counter. A strictly lower validation loss selects a new best checkpoint and
resets the counter. Training stops after five consecutive epochs without an
improvement.

The training output is a new directory containing:

```text
training_log_v1.jsonl
best_model.pt
train_state_step_000000001.pt
train_state_step_000005000.pt
...
```

`best_model.pt` contains the bare model state dictionary with the lowest
validation loss. Before test evaluation, the in-memory model is restored to
those weights. Resume files instead retain the latest training model and the
complete optimizer and scheduler state.

An interrupted run exports the best checkpoint from its completed validation
epochs. If the real-data manifest includes a `test` partition, the restored
best model evaluates it once after the epoch limit or early stopping and writes
`test_metrics_v1.json`. Stopping partway through training does not run test
evaluation.

Keep the configuration, data split, and evaluation results alongside weights
you train. Participant data are obtained separately.
