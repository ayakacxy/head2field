<a id="source-provenance"></a>

# Implementation and dependencies

[English](SOURCE_PROVENANCE.md) · [简体中文](SOURCE_PROVENANCE_zh.md)

The model, preprocessing, FEM-label, and training components originate from
the Head2Field research codebase and are maintained here as a standalone Python
package. This guide describes the current module responsibilities and external
integrations. For tensor dimensions and execution flow, see the
[code guide](en/code_guide.md).

## Package components

Paths below are relative to the repository root.

| Component | Main source paths | Responsibility |
| --- | --- | --- |
| Model and architecture | `src/head2field/model.py`, `src/head2field/_spec.py` | Anatomical encoding, query/electrode tokens, shared responses, and the fixed v1 factory |
| Inputs and geometry | `src/head2field/inputs.py`, `src/head2field/geometry.py` | Model tensors, voxel/world transforms, and ordered query/electrode identities |
| Weight loading | `src/head2field/weights.py` | Bundled model access and strict FP32 state-dictionary validation |
| Inference | `src/head2field/inference.py`, `src/head2field/cli.py` | Subject caches, pair/query batching, field arrays, and prediction manifests |
| Preprocessing | `src/head2field/preprocessing/` | T1 preparation, external CHARM invocation, conductivity, and prepared-subject assembly |
| FEM labels | `src/head2field/fem_labels/` | SimNIBS execution, solver-output validation, interpolation, and label manifests |
| Training | `src/head2field/training/` | Dataset loading, statistics, physics pools, losses, validation, and local resume |
| Resources | `src/head2field/resource_paths.py`, `src/head2field/resources/` | Packaged configurations, schemas, documents, and pretrained weights |

## Validation and outputs

Loaders validate the actual arrays, geometry, tissue conductivities, source/query
order, file sizes, and units. Each workflow has a defined manifest and produces
new output directories. Training resume verifies the saved configuration and
dataset contract. See [data contracts](en/data_contracts.md) for the schemas and
[inference](en/inference.md) for output interpretation.

The model factory and weight loader share the v1 parameter keys and tensor
shapes. Software, model, and data-schema versions are tracked independently.
See the [model card](MODEL_CARD.md) and [weights](../weights/README.md) for
compatibility and intended use.

## External integrations

PyTorch, NumPy, and NiBabel provide the base tensor and imaging runtime. SciPy
and h5py are optional dependencies for preprocessing and FEM data handling.
SimNIBS/CHARM 4.6.0 is installed separately and accessed through the
preprocessing and FEM adapters.

The package does not embed external runtimes, native libraries, templates,
atlases, or participant datasets. Dependencies and their license notices are
listed in [third-party notices](THIRD_PARTY_NOTICES.md); dataset terms and
acknowledgements are in [data use](DATA_USE.md).

## Distributed materials

The wheel contains project code, configuration/schema resources, documentation,
and the bundled tensor-only v1 weights. The source archive also includes tests
and example generators. Participant data, generated FEM or prediction arrays,
training caches, logs, and complete resume states are stored separately.

Tests and examples use fictional identifiers and generated arrays. Development
and packaging instructions are in [Contributing](CONTRIBUTING.md) and the
[release checklist](RELEASE_CHECKLIST.md).
