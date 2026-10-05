# Data contracts

[English](data_contracts.md) · [简体中文](../zh/data_contracts.md)

JSON schemas describe manifest fields. Runtime validators directly check the
arrays, coordinates, conductivity, source order, finite values, and units.

| Contract | Consumer |
| --- | --- |
| `prepared_subject_v1` | Preprocessing, inference, FEM and training |
| `fem_labels_v1` | Complete 75-source FEM export and training |
| `training_loss_statistics_v2` | 50/100 mm regional energy supervision |
| `training_dataset_v1` | Group-disjoint training and local resume |

See [schemas](../../schemas/) and the [input contract](input_contract.md).
Required input files and FEM/training assets use relative paths. Loaders reject
unsafe paths, missing files, inconsistent byte sizes, shapes, and scientific
parameters. Query identity is the exact ordered `gmwm_ijk` array; source identity
is the ordered list excluding Oz. New FEM exports store `query_ijk.npy` and
compare it directly with prepared queries. Energy-statistics NPZ files store
the query, affine, and electrode coordinates and compare them directly.

The conductivity table is checked by its complete values and units. FEM checks
also include current, vector components, HDF5 metadata, and solver conductivity
readback. Training resume compares the saved full configuration and dataset
group/path records. Optional digest fields in older manifests are ignored.
