# Getting started

[English](getting_started.md) · [简体中文](../zh/getting_started.md)

This guide covers installation, prediction, and a small training check. The
wheel includes the pretrained Head2Field v1 model, default configurations, and
technical documentation. Your input images and prepared subjects stay local.

Installed documentation is also readable with
`head2field.resource_paths.read_packaged_text("docs", "en/getting_started.md")`.

## Installation

Use Python 3.11 or newer. From the directory containing the downloaded wheel:

```bash
python -m pip install ./head2field-0.1.0-py3-none-any.whl
head2field-predict --help
```

For preprocessing and FEM workflows, install the wheel with its optional extra:

```bash
python -m pip install './head2field-0.1.0-py3-none-any.whl[full]'
```

`full` adds SciPy and h5py. SimNIBS/CHARM 4.6.0 is installed separately and must
be available in the Python environment running preprocessing or FEM commands.
For a source checkout, use `python -m pip install '.[full]'` instead. See
[preprocessing](preprocessing.md) and [FEM label generation](fem_label_generation.md)
for the external head-modeling and solver requirements.

CPU inference works with the base installation. To use a GPU, select a PyTorch
installation compatible with your CUDA driver and check it in the same
environment as Head2Field:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Use `--device cuda` or `--device cuda:0` when CUDA is available. Real-data
training uses CUDA FP16; the synthetic training check below uses CPU FP32.

## Prepare inputs

Inference consumes a directory containing `prepared_subject_v1.json` and its
referenced files. It needs anatomy, conductivity, GM/WM query rows, and the
standard 76-electrode cap. The exact files, coordinates, and tissue labels are
listed in the [input contract](input_contract.md).

Starting with a raw T1 image:

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

`non_hcp_raw` estimates one rigid alignment; inspect the resulting QC image.
For already ACPC-aligned, distortion-corrected HCP T1 data, choose `hcp_acpc`
and omit `--template`. Each output directory must be new. The alignment
template is supplied by the user under its applicable distribution terms.

## Predict and inspect fields

```bash
head2field-predict \
  --prepared-subject prepared_subject \
  --output-dir prediction \
  --device cpu
```

The command verifies the prepared inputs and bundled weights before prediction.
It writes a `[75,Q,3]` float32 field basis, query coordinates, electrode pairs,
and a explicit manifest. `Q` is the number of prepared GM/WM query rows.
Each source-to-Oz map represents source `+1 mA` and Oz `-1 mA`; components are
world-RAS `[Ex,Ey,Ez]` in `V/m`.

Read the basis without loading the entire array into RAM:

```python
import json
from pathlib import Path

import numpy as np

output = Path("prediction")
manifest = json.loads((output / "prediction_manifest_v1.json").read_text())
basis = np.load(output / "lead_field_float32.npy", mmap_mode="r", allow_pickle=False)
queries = np.load(output / "query_world_ras_mm.npy", allow_pickle=False)
print(basis.shape, manifest["source_names"], queries.shape)
```

Preserve all five output files together. Use the manifest's `source_names` to
interpret the electrode axis and the query arrays to interpret the query axis.
For current combinations, chunking, and Python inference, see
[inference](inference.md).

## Configuration files after wheel installation

Prediction selects its bundled weights and fixed v1 architecture automatically.
Training and preflight accept an explicit `--config` path. A source checkout
already contains `configs/`. After installing only the wheel, copy the existing
default configuration resources into your working directory:

```python
from pathlib import Path

from head2field.resource_paths import CONFIG_FILES, read_packaged_text

destination = Path("configs")
destination.mkdir(exist_ok=True)
for name in CONFIG_FILES:
    with (destination / name).open("x", encoding="utf-8") as file:
        file.write(read_packaged_text("configs", name))
```

Run this once in a new working directory. Exclusive file creation preserves
existing local configuration files. These are copies of the shipped defaults;
no settings are inferred from the machine or input data.

## Small CPU training check

With the default configuration files available:

```bash
head2field-make-synthetic-training --output-dir generated/synthetic-training

head2field-train-preflight \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json

head2field-train \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json \
  --output-dir generated/synthetic-run
```

The check uses an 8-cube test model and analytic data, with one fictional
training subject and one fictional validation subject. It exercises the
training machinery; it does not measure pretrained-model accuracy. Real-data
preparation, training, and resume are described in [training](training.md).

## Common setup issues

| Symptom | Check or action |
| --- | --- |
| `head2field-predict` is unavailable | Activate the environment used for installation. `python -m head2field.cli --help` runs the same prediction parser. |
| `configs/training_smoke_v1.json` is missing | Copy the bundled configuration resources using the snippet above, or run from the source checkout. |
| Output directory already exists | Choose a new destination; commands preserve prior results. |
| Prepared-subject metadata or coordinate validation fails | Check the input manifest and its original referenced files; see the input contract. |
| CUDA is unavailable | Check the PyTorch build and driver in the active environment; use CPU for inference. |
| GPU memory is insufficient | Reduce `--query-chunk-size` first, then `--electrode-batch-size`; pair/query ordering stays fixed. |
| Custom weights fail strict validation | Use a bare FP32 v1 state dictionary with the expected keys and shapes; see the weight format below. |

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.
