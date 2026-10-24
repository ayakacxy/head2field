# Getting started

[English](getting_started.md) · [简体中文](../zh/getting_started.md)

Install Head2Field, prepare a subject, and predict the subject's 75 lead-field
maps. The wheel includes pretrained weights, default configurations, and the
technical guides. Supply your own T1 image and, when required, an alignment
template.

You can also read this guide after installation with
`head2field.resource_paths.read_packaged_text("docs", "en/getting_started.md")`.

## Installation

Use Python 3.11 or newer. In the directory containing the downloaded wheel, run:

```bash
python -m pip install ./head2field-0.1.0-py3-none-any.whl
head2field-predict --help
```

For preprocessing and FEM label generation, install the additional dependencies:

```bash
python -m pip install './head2field-0.1.0-py3-none-any.whl[full]'
```

The `full` extra adds SciPy and h5py. Install SimNIBS/CHARM 4.6.0 separately in
the same Python environment used for preprocessing or FEM. From a source
checkout, use `python -m pip install '.[full]'`. See
[preprocessing](preprocessing.md) and [FEM label generation](fem_label_generation.md)
for details.

The base installation supports CPU inference. For GPU inference, install a
PyTorch build compatible with your CUDA driver, then check the active environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is available, use `--device cuda` or `--device cuda:0`. Real-data
training uses CUDA FP16; the small training example below runs on CPU in FP32.

## Prepare inputs

Inference reads a subject directory containing `prepared_subject_v1.json`,
preprocessed images, conductivity, GM/WM query points, and the standard
76-electrode cap. The [input guide](input_contract.md) describes the files,
tissue labels, and coordinate conventions.

To prepare a raw T1 image:

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

The `non_hcp_raw` profile performs one rigid alignment. Inspect the QC image
before using the result. For HCP T1 images that already have ACPC alignment and
distortion correction, select `hcp_acpc` and omit `--template`. All output
directories must be new. Obtain the alignment template under its applicable
terms of use.

## Predict and inspect fields

```bash
head2field-predict \
  --prepared-subject prepared_subject \
  --output-dir prediction \
  --device cpu
```

After checking the inputs and weights, the command writes a float32
`[75,Q,3]` field basis, query coordinates, electrode pairs, and a JSON manifest.
`Q` is the number of GM/WM query points. Each map represents `+1 mA` at its
source and `-1 mA` at Oz, with world-RAS components `[Ex,Ey,Ez]` in `V/m`.

Use a memory map to inspect the basis without loading the complete array into RAM:

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

Keep all five output files together. The manifest's `source_names` gives the
order of the electrode axis; the query arrays identify the locations along the
query axis. See [inference](inference.md) for current combinations, chunk sizes,
and the Python API.

## Configuration files after wheel installation

Prediction automatically loads the bundled weights and their matching model
architecture. Training and preflight read the file given by `--config`. A source
checkout contains `configs/`; after a wheel-only installation, copy the bundled
defaults into your working directory:

```python
from pathlib import Path

from head2field.resource_paths import CONFIG_FILES, read_packaged_text

destination = Path("configs")
destination.mkdir(exist_ok=True)
for name in CONFIG_FILES:
    with (destination / name).open("x", encoding="utf-8") as file:
        file.write(read_packaged_text("configs", name))
```

Run this once in a new working directory. The snippet refuses to replace an
existing configuration file. It copies the shipped defaults without changing
settings for the local machine or input data.

## Small CPU training check

Once the configuration files are available, run:

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

This example uses an 8-cube test model and analytic data for one fictional
training subject and one fictional validation subject. It checks that the
training workflow runs; it does not evaluate the pretrained model. For real
data and checkpoint recovery, see [training](training.md).

## Common setup issues

| Symptom | Check or action |
| --- | --- |
| `head2field-predict` is unavailable | Activate the environment used for installation. `python -m head2field.cli --help` opens the same prediction command. |
| `configs/training_smoke_v1.json` is missing | Copy the bundled defaults as shown above, or run from a source checkout. |
| Output directory already exists | Choose a new destination; existing results are not overwritten. |
| Input or coordinate validation fails | Check `prepared_subject_v1.json` and the files it references against the input guide. |
| CUDA is unavailable | Check the PyTorch build and driver in the active environment, or use CPU inference. |
| GPU memory is insufficient | Reduce `--query-chunk-size`, then `--electrode-batch-size`. Electrode and query order remain unchanged. |
| Custom weights fail validation | Supply a bare FP32 v1 state dictionary with the required parameter names and shapes. |

See [weights](../../weights/README.md) for checksums and custom-weight loading,
and the [code guide](code_guide.md) for implementation details.
