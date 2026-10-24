<a id="inference"></a>

# 推理

[English](../en/inference.md) · [简体中文](inference.md)

先预测被试的先导场基底，再按源电极电流组合成刺激电场。
安装和数据准备步骤见[上手指南](getting_started.md)。

<a id="requirements"></a>

## 使用条件

准备好预处理后的被试目录即可运行。基础安装已包含预训练权重，推理不需要 FEM 标签。
预测命令不导入预处理、FEM 或训练流程，也不需要 SimNIBS、SciPy 或 h5py。

<a id="cli"></a>

## CLI

```bash
head2field-predict \
  --prepared-subject <prepared-subject-dir> \
  --output-dir <new-output-dir> \
  --device cpu \
  --query-chunk-size 32768 \
  --electrode-batch-size 76
```

命令默认加载内置权重，也可通过 `--weights /path/to/custom_v1.pth` 指定兼容的自定义权重。
自定义权重文件只能包含 FP32 模型参数（state dictionary）。输入文件必须存在，输出目录必须尚不存在，
且不能位于被试目录内部；已有输出不会被覆盖。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--prepared-subject` | 必需 | 包含 `prepared_subject_v1.json` 及其引用文件的目录 |
| `--output-dir` | 必需 | 位于输入目录之外的新目录 |
| `--weights` | 内置预训练 v1 | 自定义模型的 FP32 参数文件，名称和形状须兼容 |
| `--device` | `cpu` | PyTorch 设备，支持 `cuda` 和 `cuda:0` |
| `--query-chunk-size` | `32768` | 每块最多处理的 GM/WM 查询点数量 |
| `--electrode-batch-size` | `76` | 电极编码和源/参考电极对解码的批次上限 |

显存较小时，可从 `--query-chunk-size 8192 --electrode-batch-size 8` 开始。
减小分块可以降低计算时的内存需求，完整输出仍覆盖全部 75 个源电极和所有查询点。
命令以 FP32、评估模式运行，通过 NumPy 内存映射写入电场数组。

<a id="processing-sequence"></a>

## 处理顺序

1. 检查输入数组、空间信息，以及电极和查询点顺序。
2. 构造模型输入，按参数名和形状严格加载权重。
3. 缓存该被试的解剖特征和电极特征。
4. 按顺序分块计算，写出全部 75 个电极对的电场。
5. 确认输入文件未改变，保存坐标、电极对和预测清单。

分块大小影响内存使用，不改变电极顺序、查询点顺序、坐标或单位。

<a id="output-tree"></a>

## 输出结构

```text
new-output-dir/
├── lead_field_float32.npy
├── query_ijk.npy
├── query_world_ras_mm.npy
├── electrode_pairs.json
└── prediction_manifest_v1.json
```

`lead_field_float32.npy` 为 float32 `[75, Q, 3]` 数组，
轴顺序为 `[source, query, world_ras_component]`。
向量分量是 world-RAS `[Ex,Ey,Ez]`，单位为 `V/m`，
对应源电极 `+1 mA`、Oz `-1 mA`。

| 文件 | 含义 |
| --- | --- |
| `lead_field_float32.npy` | `[75,Q,3]` world-RAS 向量，源轴顺序见清单 |
| `query_ijk.npy` | `[Q,3]` int32 网格索引，与查询轴对应 |
| `query_world_ras_mm.npy` | `[Q,3]` float32 world-RAS 毫米坐标 |
| `electrode_pairs.json` | 有序源/Oz 名称、坐标及 +1/-1 mA 电流 |
| `prediction_manifest_v1.json` | 输入、权重、输出信息，以及轴、单位和 `source_names` |

查询轴只包含 GM/WM 点。写回体素体积时，请用 `query_ijk.npy` 放置各行，
不要将数组直接变形为整个头部网格。

<a id="combine-source-currents"></a>

## 组合源电流

源电流 `I_s` 以 **mA** 表示，按下式组合基底：

```text
E(query) = sum_s I_s * basis[s, query]
I_Oz = -sum_s I_s
```

基底按 1 mA 计算，因此用安培表示的电流需要先换算成 mA。
电流顺序必须与清单中的 `source_names` 相同，Oz 电流为各源电流之和的相反数。
例如，F3 +2 mA、F4 -2 mA 时，Oz 的净电流为零：

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

先对向量分量求和，再计算电场幅值。保存组合后的电场时，请同时记录电流向量、
源电极顺序和基底清单。

<a id="python-api"></a>

## Python API

写出与 CLI 相同的文件：

```python
from pathlib import Path

from head2field.cli import predict_prepared_subject_v1

array_path = predict_prepared_subject_v1(
    prepared_subject=Path("prepared_subject"),
    output_dir=Path("prediction"),
    device="cpu",
)
```

不写入完整基底，逐块处理查询点：

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

`Head2FieldInferenceSession` 和权重加载函数可从包的顶层导入。
输入构造与底层分块函数可从示例所示模块导入。

模型和推理会话应位于同一设备。`row_indices` 与可选的 `pair_indices` 保持传入顺序。
预测张量留在计算设备上，直到显式传输到 CPU。建议逐块处理或保存结果；
保留 GPU 张量引用也会持续占用显存。
被试或权重变化后，请新建推理会话，使缓存特征与当前数据一致。

<a id="weights-and-reproducibility"></a>

## 权重与复现

可以使用内置权重，或参数名称和形状兼容的自定义 FP32 权重。
请将权重校验值、软件版本、输入清单和分块设置与预测结果一同记录。
加载要求见[权重说明](../../weights/README_zh.md)，
FEM 对比和运行时间测量见[评估说明](paper_reproduction.md)。
