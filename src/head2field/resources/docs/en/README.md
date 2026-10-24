# Documentation

[English](README.md) · [简体中文](../zh/README.md)

Follow [getting started](getting_started.md) to install the package, prepare
the inputs, and run a prediction. The guides below cover each
workflow in more detail, from MRI preprocessing to model training and evaluation.

## Workflows and implementation

| Guide | Contents |
| --- | --- |
| [Getting started](getting_started.md) | Wheel installation, CUDA checks, prediction, and bundled defaults |
| [Input files and coordinates](input_contract.md) | Subject directory contents, tissue labels, coordinates, and units |
| [Preprocessing](preprocessing.md) | Raw T1 and HCP profiles, CHARM, conductivity, and query preparation |
| [FEM labels](fem_label_generation.md) | Oz-referenced simulations, validation, and label export |
| [Training](training.md) | Cache and statistics preparation, physics calibration data, validation, and resume |
| [Inference](inference.md) | Chunking, Python APIs, output files, and current-pattern synthesis |
| [Model architecture](model_architecture.md) | Anatomical encoders, shared electrode responses, and field decoding |
| [Evaluation and reproducibility](paper_reproduction.md) | Dataset splits, FEM comparisons, and runtime measurements |
| [Data formats and validation](data_contracts.md) | JSON fields, file layouts, and validation tools |
| [Code guide](code_guide.md) | Module responsibilities, tensor shapes, and execution flow |
| [Examples](../../examples/README.md) | Synthetic inputs and installation checks |
| [Weights](../../weights/README.md) | Bundled model, tensor format, and custom-weight loading |

## Project and release information

| Document | Contents |
| --- | --- |
| [Project overview](../../README.md) | Features, installation, quick start, and overview figure |
| [Model card](../MODEL_CARD.md) | Model inputs and outputs, intended use, assumptions, and limitations |
| [Data use](../DATA_USE.md) | Dataset acknowledgements, methods citations, and distribution scope |
| [Contributing](../CONTRIBUTING.md) | Development setup, compatibility, and focused verification |
| [Release checklist](../RELEASE_CHECKLIST.md) | Preparing source and wheel distributions for release |
| [Implementation and dependencies](../SOURCE_PROVENANCE.md) | Package components and external integrations |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | External software and bundled-weight notices |
| [Changelog](../CHANGELOG.md) | Software changes and model versioning |
| [Citation guide](../CITATION_GUIDE.md) | Manuscript and software citation metadata |
| [License guide](../LICENSE_GUIDE.md) | Source license and related distribution notices |
