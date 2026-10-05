# Third-party notices and dependency boundary

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES_zh.md)

This document records upstream dependencies and the materials included in
software distributions. The project wheel and source distribution contain only this
project's Python source, package resources (including pretrained v1 weights),
documentation, and license files.
They do not vendor third-party Python wheels, native libraries, datasets,
templates, atlases, or SimNIBS source/binaries.

## Runtime and optional dependencies

| Component | Project role | Upstream license boundary |
| --- | --- | --- |
| PyTorch | tensor runtime, model execution, and training | The top-level upstream license is BSD-3-Clause, while the repository and binary distributions contain or bundle components covered by additional notices and licenses. The exact installed artifact's `LICENSE`/`NOTICE` files are authoritative for redistribution. |
| NumPy | array processing and file formats | NumPy's top-level license is BSD-3-Clause, but its source and binary artifacts can contain bundled or generated components under additional terms. Retain and review the exact artifact's license inventory. |
| NiBabel | NIfTI input/output and spatial metadata | The top-level NiBabel license is MIT; the exact artifact's upstream `COPYING` file also records third-party code/data terms and is authoritative for redistribution. |
| SciPy | rigid-alignment and numerical preprocessing support | The upstream project is BSD-3-Clause. Binary wheels can bundle OpenBLAS and compiler runtime libraries under additional licenses. |
| h5py | FEM HDF5 validation and label export | BSD-3-Clause for the upstream project; review the exact binary wheel and linked HDF5/runtime notices before redistributing an environment. |
| SimNIBS | external CHARM/head-model and TDCS lead-field backend | GPL-3.0-only for the upstream project, with its own third-party notices and scientific assets. Users install it separately; this project does not redistribute it. |

## Development-only tools

`pytest`, `jsonschema`, `Ruff`, `build`, `setuptools`, and `wheel` are used to
test and package the software. They are not imported by the installed runtime.
Record their versions and the applicable artifact license inventories when
preparing a release.

## What a release artifact contains

Runtime checks compare parameters, groups, relative paths, file sizes, coordinate order, shapes, dtypes and finite values directly. Resume compares the full configuration and dataset manifest. Regenerate statistics for the current 50/100 mm regions and training split.

## Authoritative upstream references

- [SimNIBS 4.6.0 package metadata](https://github.com/simnibs/simnibs/blob/v4.6.0/pyproject.toml)
- [PyTorch package metadata](https://github.com/pytorch/pytorch/blob/main/pyproject.toml)
- [NumPy package metadata](https://github.com/numpy/numpy/blob/main/pyproject.toml)
- [NiBabel repository and license](https://github.com/nipy/nibabel)
- [SciPy repository](https://github.com/scipy/scipy)
- [h5py repository](https://github.com/h5py/h5py)

The final release operator must compare this inventory with the actual wheel,
source archive, bundled pretrained weights, and recorded build/install
environment inventory. Any vendored or copied third-party material changes the
conclusion and requires a new review.
