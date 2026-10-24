<a id="prepared-subject-input-contract"></a>

# Input files and coordinates

[English](input_contract.md) · [简体中文](../zh/input_contract.md)

Head2Field reads preprocessed subject data from the directory created by
`head2field-prepare`. Keep the manifest and the files it references together:

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

T1, tissue labels, and conductivity must share the following grid and data types:

- 256³ voxels at 1 mm resolution, in RAS orientation.
- Affine diagonal `[1,1,1,1]` and translation `[-128,-128,-128]`.
- Float32 T1 and conductivity arrays.
- Integer tissue labels from 0 to 10, with 1=WM and 2=GM.

Conductivity values are in S/m. The tissue label selects an entry from
`[0,0.126,0.275,1.654,0.010,0.465,0.500,0.008,0.025,0.600,0.160]`.
The model uses float32 `sigma/1.654` as its conductivity input.

## Query points and electrodes

`gmwm_ijk.npy` contains all GM/WM voxel locations in a non-empty integer
`[Q,3]` array. Its rows must match
`np.argwhere(np.isin(tissues, (1, 2)))` in C-order, including their order.
The prediction's query axis follows these rows.

The electrode files contain 76 unique names, including Oz exactly once.
Their coordinates form a finite `[76,3]` array in world-RAS millimetres.
Removing Oz gives the 75 source electrodes without changing their relative order.

## File checks

`prepared_subject_v1.json` records relative file paths, sizes, spatial metadata,
the conductivity table, the model input transform, and the SimNIBS 4.6.0 head
model. The loader checks the arrays as well as the manifest. FEM preflight also
checks the CHARM mesh and software version, and compares the prepared tissue
image with the one in the CHARM directory.

Inference reads inputs without modifying them. It checks input and weight file
sizes and modification times before and after the run. Optional digest fields
in older manifests are not used for these checks.

See the [schema](../../schemas/prepared_subject_v1.schema.json) for individual
JSON fields and [preprocessing](preprocessing.md) for the preparation steps.
