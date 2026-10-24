<a id="third-party-notices-and-dependency-boundary"></a>

# Third-party software and licenses

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES_zh.md)

The wheel and source archive contain Head2Field Python code, package resources
(including pretrained weights), documentation, and license files.
Third-party Python wheels, native libraries, datasets, templates, atlases,
and SimNIBS source/binaries are installed or obtained separately.

## Runtime and optional dependencies

| Component | Project role | Upstream license and redistribution notes |
| --- | --- | --- |
| PyTorch | tensor runtime, model execution, and training | The top-level upstream license is BSD-3-Clause, while the repository and binary distributions contain or bundle components covered by additional notices and licenses. The exact installed artifact's `LICENSE`/`NOTICE` files are authoritative for redistribution. |
| NumPy | array processing and file formats | NumPy's top-level license is BSD-3-Clause, but its source and binary artifacts can contain bundled or generated components under additional terms. Retain and review the exact artifact's license inventory. |
| NiBabel | NIfTI input/output and spatial metadata | The top-level NiBabel license is MIT; the exact artifact's upstream `COPYING` file also records third-party code/data terms and is authoritative for redistribution. |
| SciPy | rigid-alignment and numerical preprocessing support | The upstream project is BSD-3-Clause. Binary wheels can bundle OpenBLAS and compiler runtime libraries under additional licenses. |
| h5py | FEM HDF5 validation and label export | BSD-3-Clause for the upstream project; review the exact binary wheel and linked HDF5/runtime notices before redistributing an environment. |
| SimNIBS | external CHARM/head-model and TDCS lead-field backend | GPL-3.0-only for the upstream project, with its own third-party notices and scientific assets. Users install it separately; this project does not redistribute it. |

## Development-only tools

`pytest`, `jsonschema`, `Ruff`, `build`, `setuptools`, and `wheel` are used
for testing and packaging and are not imported by the installed runtime.
Record their versions and applicable artifact license inventories when
preparing a release.

## What a release artifact contains

- The wheel contains Python code, resources, and the bundled pretrained v1
  `.pth` file with its checksum. Other checkpoints, training states, NIfTI,
  meshes, HDF5, NPY/NPZ, native libraries, and third-party datasets are excluded.
- The source archive includes the same project materials and pretrained weights.
- The bundled weights are a bare state dictionary with no optimizer, scheduler,
  participant identifier, or training-resume state.
- Containers, Conda environments, wheelhouses, and standalone binaries have
  their own dependency inventories. Review the license notices of the
  components they actually include.

## Authoritative upstream references

- [SimNIBS 4.6.0 package metadata](https://github.com/simnibs/simnibs/blob/v4.6.0/pyproject.toml)
- [PyTorch package metadata](https://github.com/pytorch/pytorch/blob/main/pyproject.toml)
- [NumPy package metadata](https://github.com/numpy/numpy/blob/main/pyproject.toml)
- [NiBabel repository and license](https://github.com/nipy/nibabel)
- [SciPy repository](https://github.com/scipy/scipy)
- [h5py repository](https://github.com/h5py/h5py)

Before release, compare this list with the actual wheel, source archive,
bundled weights, and build/install environment records. Review the inventory
again when adding copied or vendored third-party materials.
