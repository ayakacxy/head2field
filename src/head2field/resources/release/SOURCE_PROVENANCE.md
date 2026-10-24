<a id="source-provenance"></a>

# Implementation and dependencies

[English](SOURCE_PROVENANCE.md) · [简体中文](SOURCE_PROVENANCE_zh.md)

Head2Field is maintained as a standalone Python package developed from the
Head2Field research codebase. The table below maps its components to the
source tree. Tensor dimensions and execution flow are described in the
[code guide](en/code_guide.md).

## Package components

All paths are relative to the repository root.

| Component | Main source paths | Responsibility |
| --- | --- | --- |
| Model and architecture | `src/head2field/model.py`, `src/head2field/_spec.py` | Anatomical encoding, query/electrode tokens, shared responses, and the fixed v1 factory |
| Inputs and geometry | `src/head2field/inputs.py`, `src/head2field/geometry.py` | Model tensors, voxel/world transforms, and query-row indices and electrode order |
| Weight loading | `src/head2field/weights.py` | Bundled model access and strict FP32 state-dictionary validation |
| Inference | `src/head2field/inference.py`, `src/head2field/cli.py` | Subject caches, pair/query batching, field arrays, and prediction manifests |
| Preprocessing | `src/head2field/preprocessing/` | T1 preparation, external CHARM invocation, conductivity, and prepared-subject assembly |
| FEM labels | `src/head2field/fem_labels/` | SimNIBS execution, solver-output validation, interpolation, and label manifests |
| Training | `src/head2field/training/` | Dataset loading, statistics, physics pools, losses, validation, and local resume |
| Resources | `src/head2field/resource_paths.py`, `src/head2field/resources/` | Packaged configurations, schemas, documents, and pretrained weights |

## Validation and outputs

Loaders check arrays, geometry, tissue conductivities, source/query order,
file sizes, and units. Each workflow records its inputs and outputs in a
manifest and writes results to a new directory. Resuming training also checks
the saved configuration and dataset records.

The model factory and weight loader use the same v1 parameter names and tensor
shapes. Software, model, and data-schema versions are recorded separately.
See [data formats](en/data_contracts.md) for schema fields,
[inference](en/inference.md) for outputs, and the [model card](MODEL_CARD.md)
and [weights](../weights/README.md) for compatibility and intended use.

## External integrations

The base runtime uses PyTorch, NumPy, and NiBabel. SciPy and h5py provide
optional preprocessing and FEM data support. SimNIBS/CHARM 4.6.0 is installed
separately and called through the preprocessing and FEM adapters.

External runtimes, native libraries, templates, atlases, and participant
datasets are not bundled. The [third-party notices](THIRD_PARTY_NOTICES.md)
list dependency licenses; [data use](DATA_USE.md) covers dataset terms and
acknowledgements.

## Distributed materials

The wheel includes project code, configuration/schema resources, documentation,
and tensor-only pretrained weights. The source archive also includes tests and
example generators. Participant data, generated FEM and prediction arrays,
training caches, logs, and complete resume states are stored separately.

Tests and examples use fictional identifiers and generated arrays.
See [Contributing](CONTRIBUTING.md) for development and the
[release checklist](RELEASE_CHECKLIST.md) for packaging.
