# Inference

[English](inference.md) · [简体中文](../zh/inference.md)

Predict a subject's lead-field basis, then combine its source currents to
calculate a stimulation field. For installation and input preparation, see
[getting started](getting_started.md).

## Requirements

Supply a prepared subject directory. The base installation includes the
pretrained weights and supports inference without FEM labels. The prediction
command does not import the preprocessing, FEM, or training pipelines, or
require SimNIBS, SciPy, or h5py.

## CLI

```bash
head2field-predict \
  --prepared-subject <prepared-subject-dir> \
  --output-dir <new-output-dir> \
  --device cpu \
  --query-chunk-size 32768 \
  --electrode-batch-size 76
```

The command loads the bundled weights unless `--weights /path/to/custom_v1.pth`
is supplied. Custom weights must be a compatible bare FP32 state dictionary.
Inputs must already exist. Choose a new output directory outside the subject
directory; existing outputs are not overwritten.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--prepared-subject` | Required | Directory containing `prepared_subject_v1.json` and the files it references |
| `--output-dir` | Required | New directory outside the input directory |
| `--weights` | Bundled pretrained v1 | Compatible bare FP32 state dictionary for a custom model |
| `--device` | `cpu` | PyTorch device, including `cuda` and `cuda:0` |
| `--query-chunk-size` | `32768` | Maximum number of GM/WM query points in each chunk |
| `--electrode-batch-size` | `76` | Batch limit for electrode encoding and source/reference decoding |

If GPU memory is limited, start with
`--query-chunk-size 8192 --electrode-batch-size 8`. Smaller chunks reduce
working memory; the output still covers all 75 sources and all prepared query
points. The command uses FP32 evaluation and writes the field array through a
NumPy memory map.

## Processing sequence

1. Check the prepared arrays, spatial metadata, and electrode and query order.
2. Build the model inputs and load the weights with strict key and shape checks.
3. Cache anatomy and electrode features for this subject.
4. Calculate the query chunks in order and write all 75 pair fields.
5. Check that input files are unchanged, then save coordinates, electrode pairs,
   and the prediction manifest.

Chunk sizes affect memory use, not the electrode order, query order,
coordinates, or units.

## Output tree

```text
new-output-dir/
├── lead_field_float32.npy
├── query_ijk.npy
├── query_world_ras_mm.npy
├── electrode_pairs.json
└── prediction_manifest_v1.json
```

`lead_field_float32.npy` is a float32 `[75, Q, 3]` array. Its axes are
`[source, query, world_ras_component]`, with components `[Ex,Ey,Ez]` in
world-RAS. Values are in `V/m` for source `+1 mA` and Oz `-1 mA`.

| File | Interpretation |
| --- | --- |
| `lead_field_float32.npy` | `[75,Q,3]` world-RAS vectors in the manifest's source order |
| `query_ijk.npy` | `[Q,3]` int32 canonical-grid indices matching the query axis |
| `query_world_ras_mm.npy` | `[Q,3]` float32 world-RAS positions in millimetres |
| `electrode_pairs.json` | Ordered source/Oz names, coordinates, and +1/-1 mA currents |
| `prediction_manifest_v1.json` | Input, weight, and output metadata, axes, units, and `source_names` |

The query axis contains GM/WM points only. Use `query_ijk.npy` to place these
rows into a voxel volume; do not reshape the array as a full head grid.

## Combine source currents

Express source currents `I_s` in **mA** and combine the basis as follows:

```text
E(query) = sum_s I_s * basis[s, query]
I_Oz = -sum_s I_s
```

The basis is calculated for 1 mA, so convert currents in amperes to mA before
multiplication. Follow the manifest's `source_names` order. Oz carries the
negative sum of the source currents; F3 +2 mA and F4 -2 mA therefore gives zero
net Oz current:

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

Sum the vector components before calculating field magnitudes. Save the
current vector, source order, and basis manifest with any derived field.

## Python API

To write the same files as the CLI:

```python
from pathlib import Path

from head2field.cli import predict_prepared_subject_v1

array_path = predict_prepared_subject_v1(
    prepared_subject=Path("prepared_subject"),
    output_dir=Path("prediction"),
    device="cpu",
)
```

To process query chunks without writing the complete basis:

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

`Head2FieldInferenceSession` and the weight loader are top-level package
exports. Input construction and the lower-level chunk functions are available
from the modules shown above.

Load the model on the same device as the session. `row_indices` and optional
`pair_indices` keep their supplied order. Predictions stay on that device
until you transfer them to CPU. Process or save each chunk as it arrives;
keeping references to GPU tensors also keeps their memory allocated.
Create a new session whenever the subject or model weights change, so that
cached features match the data being predicted.

## Weights and reproducibility

Use the bundled weights or a compatible custom FP32 state dictionary.
Record the weight checksum, software version, input manifest, and chunk
settings with the result. See [weights](../../weights/README.md) for loading
requirements and [evaluation](paper_reproduction.md) for FEM comparisons and
runtime measurements.
