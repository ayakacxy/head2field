<div align="center">

# Head2Field

[English](README.md) · [简体中文](README_zh.md)

**A structural MRI-based neural field for subject-specific TES lead-field prediction.**

[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-orange?style=flat)](https://github.com/ayakacxy/head2field/releases/tag/v0.1.0)
[![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6.0](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF?style=flat)](docs/en/preprocessing.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue?style=flat)](docs/LICENSE_GUIDE.md)

[Installation](#installation) · [Quick start](#quick-start) · [Documentation](#documentation) · [Training and evaluation](#reproduction) · [Citation](#citation)

🧠 **Structural MRI** · ⚡ **Neural field prediction** · 🧭 **World-RAS vectors** · 🔬 **Physics regularization**

</div>

Head2Field predicts subject-specific electric fields for transcranial electrical
stimulation. It uses T1-derived head anatomy, tissue conductivity, and electrode
coordinates to build a reusable field basis. You can then combine these fields
with different electrode currents without retraining the model for each montage.

<div align="center">

[![Head2Field overview: MRI-derived head inputs, FEM supervision during training, and a multiscale neural field producing a 75-column lead-field basis](docs/assets/head2field-overview.png)](docs/assets/head2field-overview.pdf)

**Method overview.** (A) Subject-specific lead-field prediction and field
synthesis. (B) Multiscale anatomical encoding, electrode-conditioned decoding,
and shared-response differences. Click the figure for the vector PDF.

</div>

---

## 📚 Explore Head2Field

[🚀 Install](#installation) · [⚡ Predict fields](#quick-start) · [🧪 Train a model](#reproduction) · [📖 Read the docs](#documentation)

## 📄 Paper

This repository contains the **official PyTorch implementation** of:

> **Head2Field: A Structural MRI-Based Neural Field for Subject-Specific
> Lead-Field Prediction in Transcranial Electrical Stimulation**

The package includes **Head2Field v1** (`Head2FieldV1`), pretrained weights, and
commands for preprocessing, FEM label generation, training, and inference. The
model configuration is [head2field_v1.json](configs/head2field_v1.json).

For each subject, the model predicts 75 source-to-Oz fields at `Q` gray- and
white-matter query locations. The output has shape `[75, Q, 3]`, with world-RAS
components `[Ex, Ey, Ez]` in `V/m`. Each field corresponds to `+1 mA` at the source
and `-1 mA` at Oz. The [model card](docs/MODEL_CARD.md) describes the training
settings, intended use, and limitations.

In the manuscript's 40-subject test cohort, increasing training from 10 to
128 subjects reduced mean full-field relative L2 error from 0.118 to 0.070;
omitting physics regularization increased it to 0.078. TIS plans optimized
on the predicted basis achieved 94.9% of FEM-optimized focality under FEM
re-evaluation. With inputs prepared, lead-field generation was approximately
100 times faster in the representative benchmark; the estimated end-to-end
speedup was 4.7 times. Experimental conditions and downstream results are in
[evaluation and reproducibility](docs/en/paper_reproduction.md#manuscript-results).

## ✨ Features

| Workflow | Capability | Requirements |
| --- | --- | --- |
| Inference | Prepared head inputs and model weights to 75 lead-field maps; no FEM labels required | Base installation |
| Preprocessing | T1 canonicalization, external CHARM head modeling, tissue conductivity, GM/WM queries, and standard electrode cap | `preprocess` extra and SimNIBS 4.6.0 |
| FEM labels | Oz-referenced lead-field generation, HDF5 validation, and voxel-label export | `fem` extra and SimNIBS 4.6.0 |
| Training | Data and physics losses, complete epochs, validation, early stopping, and local resume | `training` extra and CUDA-enabled PyTorch |

<a id="installation"></a>

## 🚀 Installation

Use Python 3.11 or newer. Download the wheel, including the pretrained weights,
from [Release v0.1.0](https://github.com/ayakacxy/head2field/releases/tag/v0.1.0)
and install it:

```bash
python -m pip install head2field-0.1.0-py3-none-any.whl
head2field-predict --help
```

To install from source, run this in the repository root:

```bash
python -m pip install .
```

For optional preprocessing and FEM dependencies, install the local wheel with:

```bash
python -m pip install './head2field-0.1.0-py3-none-any.whl[full]'
```

From a source checkout, select the workflows as needed:

```bash
python -m pip install '.[preprocess]'
python -m pip install '.[fem]'
python -m pip install '.[training]'
python -m pip install '.[full]'
```

Preprocessing and FEM label generation also need a separate installation of
SimNIBS/CHARM **4.6.0**. Real-data training needs CUDA-enabled PyTorch. The
`training` extra is a convenience name for the training workflow; its Python
dependencies are already included in the base package. The `full` extra adds
SciPy and h5py.

See [getting started](docs/en/getting_started.md) for CUDA setup, copying the
bundled configuration files, and troubleshooting installation.

<a id="quick-start"></a>

## ⚡ Quick start

### Predict lead fields

Start with a [prepared subject directory](docs/en/input_contract.md). The command
uses the [bundled pretrained weights](weights/README.md) by default:

```bash
head2field-predict \
  --prepared-subject prepared_subject \
  --output-dir prediction \
  --device cpu
```

Set `--device cuda` to use a GPU. The output directory contains
`lead_field_float32.npy` (`[75, Q, 3]`), the query coordinates and electrode-pair
records, and a manifest describing the inputs and outputs. The fields use the
world-RAS components and 1 mA convention described above. See
[inference](docs/en/inference.md) for query chunking and the Python API.

Use `--weights /path/to/custom_v1.pth` to select your own compatible weights.

### Prepare a raw T1 image

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

Use `non_hcp_raw` for an unaligned T1 image. It performs one 6-DOF rigid
alignment; inspect the QC image before using the result. For HCP images that
already have ACPC alignment and distortion correction, use `hcp_acpc` and omit
`--template` to skip rigid estimation. Choose new paths for all output directories. The
[preprocessing guide](docs/en/preprocessing.md) explains both profiles.

### Check the training installation with synthetic data

This example runs a small CPU FP32 training job. It needs neither participant
data nor SimNIBS.

Run the commands from the source checkout, or
[copy the bundled configuration files](docs/en/getting_started.md#configuration-files-after-wheel-installation)
into your working directory after installing the wheel.

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

The generated arrays are useful for checking that training runs, but they do
not represent head anatomy or measure prediction accuracy. See
[examples](examples/README.md) for the available checks.

<a id="reproduction"></a>

## 🧪 Training and evaluation

To train on real data, first prepare the subject inputs and FEM labels. The
default [training_v1.json](configs/training_v1.json) uses AdamW with learning
rate `2e-4` and weight decay `1e-4`, CUDA FP16, eight source pairs and 49,152
queries per step. It trains complete epochs for up to 25 epochs, with a
25-epoch cosine schedule and validation at the end of each epoch. Training
stops after five epochs without improvement; testing uses the best validation
checkpoint. The data loss uses 50/100 mm radial regions.

Follow [training and resume](docs/en/training.md) for data preparation and
checkpoint handling. [Evaluation and reproducibility](docs/en/paper_reproduction.md)
explains the data splits, FEM comparisons, and runtime measurements. Participant
data and cohort splits are obtained separately.

<a id="documentation"></a>

## 📖 Documentation

- [Documentation index](docs/en/README.md)
- [Getting started: installation to prediction](docs/en/getting_started.md)
- [Input format and coordinate conventions](docs/en/input_contract.md)
- [T1 preprocessing and CHARM integration](docs/en/preprocessing.md)
- [FEM label generation and validation](docs/en/fem_label_generation.md)
- [Training, validation, and resume](docs/en/training.md)
- [Inference and output format](docs/en/inference.md)
- [Model architecture](docs/en/model_architecture.md)
- [Evaluation and reproducibility](docs/en/paper_reproduction.md)
- [Machine-readable data formats](docs/en/data_contracts.md)
- [Code guide: modules, tensor shapes, and execution flow](docs/en/code_guide.md)
- [Model card](docs/MODEL_CARD.md), [data use](docs/DATA_USE.md), and [weight format](weights/README.md)

## 🤝 Contributing

See [Contributing](docs/CONTRIBUTING.md) for development setup and choosing
checks for a change. The [code and dependency overview](docs/SOURCE_PROVENANCE.md)
describes the package components, and the [changelog](docs/CHANGELOG.md) records
software changes. Maintainers can follow the [release checklist](docs/RELEASE_CHECKLIST.md)
when preparing a distribution.

English is the project's primary language. Simplified Chinese translations are
available for all public guides; use the language link at the top to switch.

<a id="citation"></a>

## 📝 Citation and license

The source code is licensed under [GPL-3.0-only](docs/LICENSE_GUIDE.md). For third-party
software and bundled model-weight provenance, see
[THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) and [DATA_USE.md](docs/DATA_USE.md).

If you use Head2Field in research, cite the manuscript with this BibTeX entry:

```bibtex
@misc{chen_head2field,
  author = {Chen, Xiayu and Wang, Xiaoxiao and Cao, Zixin and
            Liu, Senlin and Liu, Zhichen and Wang, Zhengyang and
            Du, Huiyu and Qiu, Bensheng and Wang, Yanming},
  title  = {{Head2Field}: A Structural {MRI}-Based Neural Field for
            Subject-Specific Lead-Field Prediction in Transcranial
            Electrical Stimulation},
  note   = {Manuscript}
}
```

The manuscript title and author order follow [CITATION.cff](CITATION.cff). See
the [citation guide](docs/CITATION_GUIDE.md) for software citation and references
for the methods and data sources.

Model development benefited from WU-Minn Human Connectome Project data;
[DATA_USE.md](docs/DATA_USE.md) provides the acknowledgement and citation requirements.
