# Deterministic synthetic examples

[English](README.md) · [简体中文](README_zh.md)

Use these small, non-anatomical datasets to check installation and software
workflows. They contain no real participant images, identifiers, coordinates,
labels, or derived data. Write generated files to a new directory excluded
from version control; local development can use a task directory under `.work/`.

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

The generator also creates a cap override with the required 76 names and
coordinates on a synthetic sphere for adapter tests. These inputs do not
represent a head and cannot establish the accuracy of CHARM segmentation or
FEM fields.

## Training installation and resume smoke

This example works from a source checkout or after a wheel installation.
The checkout includes `configs/`; for a wheel installation, first follow the
[configuration access guide](../docs/en/getting_started.md#configuration-files-after-wheel-installation).

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

The dataset contains one fictional training subject and one fictional
validation subject in disjoint groups. It uses the fixed 75-source order
and analytic arrays on an 8-cube grid.

The runner uses a small test-only CPU FP32 network to check installation,
training steps, saved state, RNG state, sampling, and resume.
It does not train the full pretrained architecture or measure field-prediction
accuracy.

If the CLI is not installed, `generate_synthetic_training_dataset.py` creates
the same dataset.
