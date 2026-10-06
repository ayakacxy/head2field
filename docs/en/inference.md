# Inference

[English](inference.md) · [简体中文](../zh/inference.md)

For installation and input preparation, start with [getting started](getting_started.md).

## Requirements

Inference uses preprocessed subject data and loads the bundled Head2Field v1
weights by default. The base installation is sufficient and does not load the
preprocessing, FEM, or training pipelines, SimNIBS, SciPy, or h5py. FEM labels
are not needed.

## CLI

```bash
head2field-predict \
  --prepared-subject <prepared-subject-dir> \
  --output-dir <new-output-dir> \
  --device cpu \
  --query-chunk-size 32768 \
  --electrode-batch-size 76
```

Add `--weights /path/to/custom_v1.pth` to use a custom compatible bare state
dictionary. The prepared-subject directory and any custom weight file must
exist. The output path must not exist and must not be inside the prepared-subject
directory. The command never overwrites an existing output directory.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--prepared-subject` | Required | Directory containing `prepared_subject_v1.json` and its declared files |
| `--output-dir` | Required | New directory outside the input directory |
| `--weights` | Bundled pretrained v1 | Optional compatible bare FP32 state dictionary |
| `--device` | `cpu` | PyTorch device, including `cuda` and `cuda:0` |
| `--query-chunk-size` | `32768` | Maximum GM/WM query rows per inference chunk |
| `--electrode-batch-size` | `76` | Limit for electrode encoding and source/reference pair decoding |

For a smaller GPU, start with `--query-chunk-size 8192 --electrode-batch-size 8`.
These settings bound network working memory; the full output still contains all
75 sources and every prepared query. The CLI writes the field array through a
NumPy memory map. Inference uses FP32 and evaluation mode.

## Processing sequence

1. Validate prepared arrays, spatial metadata, and electrode/query order.
2. Build the v1 input tensors and strictly load the bundled or custom weights.
3. Cache anatomical and electrode features in an inference session.
4. Evaluate ordered query chunks and write all 75 pair fields to the memory map.
5. Verify input files are unchanged, then write coordinates, pair records, and
   the prediction manifest.

Chunk sizes are resource controls, not scientific profile changes.

## Output tree

```text
new-output-dir/
├── lead_field_float32.npy
├── query_ijk.npy
├── query_world_ras_mm.npy
├── electrode_pairs.json
└── prediction_manifest_v1.json
```

`lead_field_float32.npy` has shape `[75, Q, 3]`, dtype float32, and axis order
`[source, query, world_ras_component]`. Components follow world-RAS. The field
unit is `V/m` for the `+1 mA` source and `-1 mA` Oz reference convention.

| File | Interpretation |
| --- | --- |
| `lead_field_float32.npy` | `[75,Q,3]` world-RAS vectors in the manifest's source order |
| `query_ijk.npy` | `[Q,3]` int32 canonical-grid indices matching the query axis |
| `query_world_ras_mm.npy` | `[Q,3]` float32 world-RAS locations in millimeters |
| `electrode_pairs.json` | Ordered source/Oz names, coordinates, and +1/-1 mA currents |
| `prediction_manifest_v1.json` | Input/weight/output metadata, axes, units, and `source_names` |

The query axis contains the prepared GM/WM rows. It is not a flattened whole
head grid; use `query_ijk.npy` when scattering values into a voxel volume.

## Combine source currents

For source currents `I_s` expressed in **mA**, the reusable basis gives:

```text
E(query) = sum_s I_s * basis[s, query]
I_Oz = -sum_s I_s
```

The numerical coefficient uses the 1 mA basis, so currents in amperes must be
converted to mA before multiplication. Source current order follows the
manifest's `source_names`; the reference is Oz. For example, F3 +2 mA and
F4 -2 mA gives zero net Oz current:

```python
import json
from pathlib import Path

import numpy as np

output = Path("prediction")
manifest = json.loads((output / "prediction_manifest_v1.json").read_text())
basis = np.load(output / "lead_field_float32.npy", mmap_mode="r", allow_pickle=False)
source_names = manifest["source_names"]
currents_mA = np.zeros(len(source_names), dtype=np.float32)
currents_mA[source_names.index("F3")] = 2.0
currents_mA[source_names.index("F4")] = -2.0

# 计算一个查询分块，后续分块执行相同操作。
field_V_per_m = np.einsum("p,pqc->qc", currents_mA, basis[:, :32768, :])
print(field_V_per_m.shape, "Oz current (mA):", -currents_mA.sum())
```

Combine vector components before computing field magnitudes. Preserve the
current vector, source order, and basis manifest with any derived field.

## Python API

To create the same files and manifest as the CLI:

```python
from pathlib import Path

from head2field.cli import predict_prepared_subject_v1

array_path = predict_prepared_subject_v1(
    prepared_subject=Path("prepared_subject"),
    output_dir=Path("prediction"),
    device="cpu",
)
```

To stream selected queries without writing the complete basis:

```python
from pathlib import Path

import numpy as np

from head2field import (
    Head2FieldInferenceSession,
    load_head2field_v1_weights,
)
from head2field.inputs import build_prepared_model_input_subject
from head2field.preprocessing.validation import load_prepared_subject_v1

model = load_head2field_v1_weights(device="cpu")
prepared = load_prepared_subject_v1(Path("prepared_subject"))
subject = build_prepared_model_input_subject(
    prepared, encoder_input_shape=(128, 128, 128)
)
session = Head2FieldInferenceSession(model, subject, device="cpu")
electrodes = session.prepare_electrodes(
    subject.source_coords_mm, electrode_batch_size=8
)
rows = np.arange(len(subject.gmwm_ijk), dtype=np.int64)

for chunk in session.iter_predictions(rows, electrodes, query_chunk_size=32768):
    field = chunk.prediction.squeeze(0).detach().cpu().numpy()
    # field 的形状为 [75, 当前分块查询数量, 3]。
    # chunk.row_indices 标识这些查询在 subject.gmwm_ijk 中的行。
```

`Head2FieldInferenceSession` and the weight-loading factory are stable top-level
exports. Prepared-subject construction and lower-level chunk orchestration
are available through the corresponding implementation modules.

Move the model to the session device when loading it. Both `row_indices` and
optional `pair_indices` preserve the supplied order. A chunk's prediction stays
on that device until transferred to CPU. Consume or save chunks as they arrive;
collecting GPU tensors retains their memory. Recreate the session after changing
the model weights or subject data because its caches assume immutable inputs.

## Weights and reproducibility

Use the bundled v1 state dictionary or compatible custom FP32 weights. Record
its checksum together with the software version, prepared-input manifest, and
chunk settings. See [weights](../../weights/README.md) for loading and format
requirements, and [evaluation](paper_reproduction.md) for FEM comparisons and
runtime measurements.
