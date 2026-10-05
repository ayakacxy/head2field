<a id="getting-started"></a>

# 上手指南

[English](../en/getting_started.md) · [简体中文](getting_started.md)

本指南介绍安装、预测和小型训练检查。wheel 内置预训练 Head2Field v1 模型、默认配置和技术文档；用户的输入图像与 prepared subject 保存在本地。

安装后的中文文档也可通过 `head2field.resource_paths.read_packaged_text("docs", "zh/getting_started.md")` 读取。

<a id="installation"></a>

## 安装

使用 Python 3.11 或更新版本，在下载的 wheel 所在目录执行：

```bash
python -m pip install ./head2field-0.1.0-py3-none-any.whl
head2field-predict --help
```

预处理和 FEM 流程需要安装 wheel 的可选 extra：

```bash
python -m pip install './head2field-0.1.0-py3-none-any.whl[full]'
```

`full` 增加 SciPy 和 h5py。SimNIBS/CHARM 4.6.0 单独安装，并应在执行预处理或 FEM 命令的 Python 环境中可用。源码目录中可执行 `python -m pip install '.[full]'`。外部头模型和求解器要求见[预处理](preprocessing.md)与[FEM 标签生成](fem_label_generation.md)。

基础安装支持 CPU 推理。使用 GPU 时，选择与 CUDA 驱动兼容的 PyTorch 安装，并在运行 Head2Field 的同一环境中检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA 可用时选择 `--device cuda` 或 `--device cuda:0`。真实数据训练使用 CUDA FP16；下方合成训练检查使用 CPU FP32。

<a id="prepare-inputs"></a>

## 准备输入

推理读取包含 `prepared_subject_v1.json` 及其引用文件的目录，需要解剖、电导率、GM/WM 查询行和标准 76 电极 cap。精确文件、坐标和组织标签见[输入合同](input_contract.md)。

从原始 T1 图像开始：

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

`non_hcp_raw` 执行一次刚体配准，请检查生成的 QC 图。对于已经完成 ACPC 对齐与畸变校正的 HCP T1，选择 `hcp_acpc` 并省略 `--template`。每个输出目录都必须是新目录。配准模板由用户按适用条款提供。

<a id="predict-and-inspect-fields"></a>

## 预测与查看电场

```bash
head2field-predict \
  --prepared-subject prepared_subject \
  --output-dir prediction \
  --device cpu
```

运行时直接核对参数、分组、相对路径、文件大小、坐标顺序、shape、dtype 和有限值。训练恢复比较完整配置与数据清单；统计必须按当前 50/100 mm 设置和训练划分生成。

无需将整个数组加载到 RAM 即可读取：

```python
import json
from pathlib import Path

import numpy as np

output = Path("prediction")
manifest = json.loads((output / "prediction_manifest_v1.json").read_text())
basis = np.load(output / "lead_field_float32.npy", mmap_mode="r", allow_pickle=False)
queries = np.load(output / "query_world_ras_mm.npy", allow_pickle=False)
print(basis.shape, manifest["source_names"], queries.shape)
```

请一同保留全部五个输出文件。通过清单的 `source_names` 解释电极轴，通过查询数组解释查询轴。电流组合、分块和 Python 推理见[推理说明](inference.md)。

<a id="configuration-files-after-wheel-installation"></a>

## wheel 安装后的配置文件

预测自动选择内置权重与固定 v1 结构。训练与预检通过 `--config` 指定配置路径。源码目录已提供 `configs/`；仅安装 wheel 时，可将现有默认配置资源复制到工作目录：

```python
from pathlib import Path

from head2field.resource_paths import CONFIG_FILES, read_packaged_text

destination = Path("configs")
destination.mkdir(exist_ok=True)
for name in CONFIG_FILES:
    with (destination / name).open("x", encoding="utf-8") as file:
        file.write(read_packaged_text("configs", name))
```

在新工作目录中执行一次即可。独占创建文件可保护已有本地配置。这些文件是随包分发的默认配置副本，不根据机器或输入数据推断设置。

<a id="small-cpu-training-check"></a>

## 小型 CPU 训练检查

准备好默认配置文件后执行：

```bash
head2field-make-synthetic-training --output-dir generated/synthetic-training

head2field-train-preflight \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json

head2field-train \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json \
  --output-dir generated/synthetic-run
```

检查使用 8³ 测试模型和解析数据，包含一个虚构训练被试和一个虚构验证被试，用于检查训练机制，不测量预训练模型精度。真实数据准备、训练和恢复见[训练说明](training.md)。

<a id="common-setup-issues"></a>

## 常见安装问题

| 现象 | 检查或处理 |
| --- | --- |
| 无法找到 `head2field-predict` | 激活安装软件时使用的环境；`python -m head2field.cli --help` 使用同一预测解析器。 |
| 缺少 `configs/training_smoke_v1.json` | 按上方示例复制内置配置，或在源码目录运行。 |
| 输出目录已存在 | 选择新目录，命令会保留已有结果。 |
| prepared subject 数组或坐标验证失败 | 核对输入清单与原始引用文件，参见输入合同。 |
| CUDA 不可用 | 检查活动环境的 PyTorch 构建与驱动；推理可选择 CPU。 |
| GPU 显存不足 | 先减小 `--query-chunk-size`，再减小 `--electrode-batch-size`；电极对与查询顺序保持固定。 |
| 自定义权重严格验证失败 | 使用参数名及形状正确的裸 FP32 v1 state dictionary，参见权重格式。 |

预训练校验值和自定义权重见[权重说明](../../weights/README_zh.md)；实现结构见[代码阅读指南](code_guide.md)。
