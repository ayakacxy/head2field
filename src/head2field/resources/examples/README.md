# Deterministic synthetic examples

[English](README.md) · [简体中文](README_zh.md)

These generators create small, non-anatomical inputs. They contain no real
participant image, identifier, coordinate, label, or derived data. Always write
generated files to a new directory excluded from version control. For local
development, use an ignored task directory under `.work/`.

## Preprocessing branch inputs

```bash
python examples/generate_synthetic_preprocessing_inputs.py \
  --output-dir generated/synthetic-preprocessing

python -m head2field.preprocessing canonicalize \
  --input-t1 generated/synthetic-preprocessing/synthetic_raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template generated/synthetic-preprocessing/synthetic_orientation_template.nii.gz \
  --output-dir generated/synthetic-canonical
```

The generator also writes a strict 76-name cap override for adapter tests. Its
coordinates lie on a synthetic sphere. Do not run CHARM/FEM on these files and
interpret the result as anatomy or scientific validation.

## Training installation and resume smoke

The CLI example below also works after installing the wheel. Make the bundled
configuration files available using the
[configuration access guide](../docs/en/getting_started.md#configuration-files-after-wheel-installation).
Source checkouts already include `configs/`.

```bash
head2field-make-synthetic-training \
  --output-dir generated/synthetic-training

head2field-train-preflight \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json

head2field-train \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json \
  --output-dir generated/synthetic-run
```

The dataset has one fictional training subject and one fictional validation
subject with disjoint groups, the fixed 75-source order, and analytic arrays on
an 8-cube grid. The runner uses a test-only small network and CPU FP32. It
exercises and checks the install, training-step, state, RNG, sampler, and resume
machinery. It does not train the full pretrained architecture or measure
field-prediction accuracy.

The standalone script
`generate_synthetic_training_dataset.py` creates the same dataset when the CLI
entry point has not yet been installed.
