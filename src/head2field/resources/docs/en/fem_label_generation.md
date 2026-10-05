# FEM label generation

[English](fem_label_generation.md) · [简体中文](../zh/fem_label_generation.md)

This workflow uses external SimNIBS 4.6.0 and prepared head inputs. Every
destination must be new. The solver produces 75 source-to-Oz fields with the
standard 76-electrode cap, circular electrodes of diameter 10 mm and thickness
4 mm, and the complete isotropic tissue conductivity table.

```bash
head2field-fem plan --prepared-subject-dir prepared_subject --head-mesh head_model/subject.msh

head2field-fem generate   --prepared-subject-dir prepared_subject   --head-mesh head_model/subject.msh   --leadfield-output-dir leadfield_run   --labels-output-dir fem_labels

head2field-fem validate   --prepared-subject-dir prepared_subject   --fem-labels-dir fem_labels
```

`export` can interpolate an existing validated solver run; inspect
`head2field-fem export --help` for its HDF5, log, runtime-manifest and output
arguments. Linear interpolation uses GM/WM queries. SimNIBS 1 A fields are
scaled by 0.001 to source +1 mA and Oz −1 mA, with float32 world-RAS vectors
in V/m.

The run retains solver log, HDF5 and runtime MAT. Configured tissue
conductivities are compared with the runtime MAT readback. Labels retain
ordered source names, relative paths, file sizes, `[Q,3]` shapes, float32 dtype
and finite values. New exports also store `query_ijk.npy` and compare its
ordered coordinates directly with the prepared subject.

Validation checks current, reference, field, units, tissues, source order,
coordinates, conductivity, numerical arrays and safe paths. There is one
validation mode. Older optional digest fields do not determine acceptance.
Outputs publish atomically from sibling staging and never overwrite existing
results. See [contracts](data_contracts.md) and
[FEM schema](../../schemas/fem_labels_v1.schema.json).
