<a id="inference"></a>

# 推理

[English](../en/inference.md) · [简体中文](inference.md)

安装与输入准备从[上手指南](getting_started.md)开始。

<a id="requirements"></a>

## 使用条件

推理使用预处理后的被试数据，默认加载安装包内的 Head2Field v1 权重。

基础安装即可运行推理，不需要 FEM 标签。运行时不会导入预处理、FEM 或训练模块，也不会加载 SimNIBS、SciPy 或 h5py。

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

添加 `--weights /path/to/custom_v1.pth` 可选择兼容的自定义裸 state dictionary。prepared subject 目录及自定义权重文件必须存在；输出目录必须尚不存在，且不能位于 prepared subject 目录内部。命令不覆盖已有输出目录。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--prepared-subject` | 必需 | 包含 `prepared_subject_v1.json` 及声明文件的目录 |
| `--output-dir` | 必需 | 位于输入目录之外的新目录 |
| `--weights` | 内置预训练 v1 | 可选的兼容裸 FP32 state dictionary |
| `--device` | `cpu` | PyTorch 设备，支持 `cuda` 和 `cuda:0` |
| `--query-chunk-size` | `32768` | 每个推理分块最多包含的 GM/WM 查询行数 |
| `--electrode-batch-size` | `76` | 电极编码与源/参考电极对解码的批次上限 |

较小显存的 GPU 可从 `--query-chunk-size 8192 --electrode-batch-size 8` 开始。这些设置限制网络工作内存，完整输出仍包含全部 75 个源电极和每个 prepared query。CLI 通过 NumPy 内存映射写入电场数组，推理采用 FP32 和评估模式。

<a id="processing-sequence"></a>

## 处理顺序

1. 验证 prepared 数组、空间元数据及电极/查询顺序。
2. 构造 v1 输入张量，严格加载内置或自定义权重。
3. 在推理 session 中缓存解剖与电极特征。
4. 按查询顺序分块计算，将全部 75 个电极对电场写入内存映射数组。
5. 检查输入文件未发生变化，写入查询坐标、电极对记录和预测清单。

调整分块大小会影响内存使用，但不改变电极或查询点的顺序、坐标系和输出单位。

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

`lead_field_float32.npy` 形状为 `[75, Q, 3]`，dtype 为 float32，轴顺序是 `[source, query, world_ras_component]`。分量采用 world-RAS，电场单位为 `V/m`，对应固定的源电极 `+1 mA`、Oz 参考 `-1 mA`。

| 文件 | 含义 |
| --- | --- |
| `lead_field_float32.npy` | `[75,Q,3]` world-RAS 向量，源轴顺序由清单规定 |
| `query_ijk.npy` | `[Q,3]` int32 canonical 网格索引，与查询轴对应 |
| `query_world_ras_mm.npy` | `[Q,3]` float32 world-RAS 毫米位置 |
| `electrode_pairs.json` | 有序源/Oz 名称、坐标及 +1/-1 mA 电流 |
| `prediction_manifest_v1.json` | 输入信息、轴、单位和 `source_names` |

查询轴仅包含 prepared GM/WM 行，不是整个头部网格的展开数组。将结果写回体素体积时应使用 `query_ijk.npy`。

<a id="combine-source-currents"></a>

## 组合源电流

源电流 `I_s` 以 **mA** 表示时，可复用基底满足：

```text
E(query) = sum_s I_s * basis[s, query]
I_Oz = -sum_s I_s
```

数值系数基于 1 mA 基底，使用安培电流时应先换算为 mA。电流顺序遵循清单的 `source_names`，参考为 Oz。例如，F3 +2 mA 与 F4 -2 mA 对应 Oz 净电流为零：

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

应先组合向量分量，再计算电场幅值。保存衍生电场时，也应保留电流向量、源顺序及基底清单。

<a id="python-api"></a>

## Python API

生成与 CLI 相同的文件和清单：

```python
from pathlib import Path

from head2field.cli import predict_prepared_subject_v1

array_path = predict_prepared_subject_v1(
    prepared_subject=Path("prepared_subject"),
    output_dir=Path("prediction"),
    device="cpu",
)
```

不写入完整基底、流式处理选定查询：

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

`Head2FieldInferenceSession` 和权重加载工厂是稳定的顶层导出。prepared subject 构建与底层分块流程可通过相应实现模块调用。

模型与推理会话应使用同一设备。`row_indices` 和可选 `pair_indices` 保持传入顺序。每块预测最初保存在推理设备上，需要 CPU 数组时再传输。建议边计算边处理或保存结果，避免长期保留 GPU 张量引用而占用显存。换用其他权重或被试数据时，请新建推理会话，避免复用旧缓存。

<a id="weights-and-reproducibility"></a>

## 权重与复现

使用内置 v1 state dictionary 或兼容的自定义 FP32 权重，将权重校验值与软件版本、prepared 输入清单及分块设置一同记录。加载和格式要求见[权重说明](../../weights/README_zh.md)，FEM 对比和运行时间测量见[评估说明](paper_reproduction.md)。
