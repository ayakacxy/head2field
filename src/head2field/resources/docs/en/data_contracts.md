<a id="data-contracts"></a>

# Data formats and validation

[English](data_contracts.md) · [简体中文](../zh/data_contracts.md)

JSON schemas define the required manifest fields. File loading also checks array
shapes, coordinates, conductivity, electrode order, and units, and rejects
numerical arrays containing NaN or Inf.

| Format | Purpose |
| --- | --- |
| `prepared_subject_v1` | Prepared inputs for inference, FEM, and training |
| `fem_labels_v1` | FEM labels for all 75 sources, used in training |
| `training_loss_statistics_v2` | Energy statistics for the 50/100 mm regions |
| `training_dataset_v1` | Training file paths and train/validation/test groups |

See [schemas](../../schemas/) for field definitions and
[input format](input_contract.md) for the directory layout. Paths are relative
to their manifests; loading checks path safety, file existence, and file sizes.

Predictions, FEM labels, and training statistics must use the same query points
and source order:

- Query rows follow `gmwm_ijk`; sources follow the electrode list with Oz removed.
- FEM export saves `query_ijk.npy` and compares it with the prepared query rows.
- Energy-statistics NPZ files store query rows, the affine, and electrode
  coordinates, which are compared with the inputs when loaded.

Checks cover every conductivity-table value and its units. FEM validation also
checks current, vector components, HDF5 metadata, and conductivity read back from
the solver. Training resume requires the saved configuration, groups, and data
paths to match. Optional digest fields in older manifests are not used for
validation.
