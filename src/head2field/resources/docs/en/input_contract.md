# Prepared-subject input contract

[English](input_contract.md) · [简体中文](../zh/input_contract.md)

Inference reads a prepared directory. Use `head2field-prepare` to produce it.

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

Volumes are 256³, 1 mm, RAS, with affine diagonal `[1,1,1,1]` and spatial
translation `[-128,-128,-128]`. T1 and sigma are float32. Tissue labels are
integers 0–10, with 1=WM and 2=GM. Conductivity in S/m equals the exact lookup
`[0,0.126,0.275,1.654,0.010,0.465,0.500,0.008,0.025,0.600,0.160]` at each voxel;
the network uses float32 `sigma/1.654`.

`gmwm_ijk.npy` is a non-empty integer `[Q,3]` array equal to the C-order
`np.argwhere(tissues in {1,2})`. The 76 electrode names are unique, include Oz
once, and match finite world-RAS `[76,3]` coordinates in millimetres. The 75
source names retain their order after excluding Oz.

The manifest records required relative paths, file sizes, spatial metadata,
the full conductivity table and input transform, plus the SimNIBS 4.6.0
head-model provenance. The loader checks actual arrays and geometry. FEM
preflight additionally checks the unique CHARM mesh, runtime version and
prepared/CHARM tissue equality. Prediction keeps inputs read-only and observes
input and weight file size/mtime before and after execution.

See [schema](../../schemas/prepared_subject_v1.schema.json) and
[preprocessing](preprocessing.md). Optional digest fields in older manifests
are accepted as unused provenance.
