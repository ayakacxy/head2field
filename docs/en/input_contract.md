<a id="prepared-subject-input-contract"></a>

# Input files and coordinates

[English](input_contract.md) · [简体中文](../zh/input_contract.md)

Inference reads a directory of preprocessed subject data. Run
`head2field-prepare` to create the following files:

```text
prepared-subject/
├── prepared_subject_v1.json
├── images/
│   ├── T1_1mm.nii.gz
│   ├── final_tissues.nii.gz
│   └── sigma_1mm.nii.gz
├── voxels/gmwm_ijk.npy
└── electrodes/
    ├── electrode_names_all.json
    ├── electrode_names_no_oz.json
    └── electrode_coords_mm.npy
```

## Images and conductivity

T1, tissue labels, and conductivity must use the same grid:

- 256³ voxels, 1 mm resolution, and RAS orientation.
- Affine diagonal `[1,1,1,1]` and translation `[-128,-128,-128]`.
- Float32 T1 and conductivity arrays.
- Integer tissue labels from 0 to 10, with 1=WM and 2=GM.

Conductivity is in S/m. Each voxel takes the value at its tissue-label index in
`[0,0.126,0.275,1.654,0.010,0.465,0.500,0.008,0.025,0.600,0.160]`.
The model receives float32 `sigma/1.654`.

## Query points and electrodes

`gmwm_ijk.npy` is a non-empty integer `[Q,3]` array containing all GM/WM voxel
locations. Its contents and order must match the C-order result of
`np.argwhere(np.isin(tissues, (1, 2)))`. Predictions use the same query-row order.

The electrode files contain 76 unique names, with Oz appearing once. Coordinates
have shape `[76,3]`, use world-RAS millimetres, and cannot contain NaN or Inf.
Removing Oz leaves 75 sources in their original order.

## File checks

`prepared_subject_v1.json` records relative paths, file sizes, spatial metadata,
the conductivity table, model input transform, and SimNIBS 4.6.0 head-model
information. Loading also checks the actual arrays. Before FEM execution,
preflight checks the CHARM mesh, software version, and agreement between the
prepared and CHARM tissue images.

Inference keeps inputs read-only and checks input and weight file sizes and
modification times before and after execution. Optional digest fields in older
manifests do not participate in these checks.

See the [schema](../../schemas/prepared_subject_v1.schema.json) for JSON fields
and [preprocessing](preprocessing.md) for data preparation.
