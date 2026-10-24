<a id="getting-started"></a>

# 上手指南

[English](../en/getting_started.md) · [简体中文](getting_started.md)

本指南介绍如何安装 Head2Field、准备被试数据，以及预测 75 个先导场。
wheel 已包含预训练权重、默认配置和技术文档；T1 影像和需要使用的配准模板由用户准备。

安装后，也可以通过
`head2field.resource_paths.read_packaged_text("docs", "zh/getting_started.md")`
读取本指南。

<a id="installation"></a>

## 安装

使用 Python 3.11 或更新版本，在下载的 wheel 所在目录执行：

```bash
python -m pip install ./head2field-0.1.0-py3-none-any.whl
head2field-predict --help
```

需要预处理或生成 FEM 标签时，安装相应的附加依赖：

```bash
python -m pip install './head2field-0.1.0-py3-none-any.whl[full]'
```

`full` 包含 SciPy 和 h5py。SimNIBS/CHARM 4.6.0 需要单独安装，并且与预处理或 FEM
命令使用同一个 Python 环境。从源码安装时，执行 `python -m pip install '.[full]'`。
详细要求见[预处理](preprocessing.md)和[FEM 标签生成](fem_label_generation.md)。

基础安装即可进行 CPU 推理。使用 GPU 时，安装与 CUDA 驱动兼容的 PyTorch，
然后在运行 Head2Field 的环境中检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA 可用时，使用 `--device cuda` 或 `--device cuda:0`。真实数据训练使用 CUDA FP16；
下方的小型训练示例在 CPU 上以 FP32 运行。

<a id="prepare-inputs"></a>

## 准备输入

推理读取包含 `prepared_subject_v1.json` 的被试目录，其中还需有预处理影像、电导率、
GM/WM 查询点和标准 76 电极布局。文件结构、组织标签和坐标要求见[输入指南](input_contract.md)。

从原始 T1 影像开始，可执行：

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

`non_hcp_raw` 进行一次刚体配准，使用结果前请检查 QC 图。对于已经完成 ACPC 对齐和
畸变校正的 HCP T1，选择 `hcp_acpc` 并省略 `--template`。所有输出目录必须尚不存在。
配准模板应按其适用的使用条款获取。

<a id="predict-and-inspect-fields"></a>

## 预测与查看电场

```bash
head2field-predict \
  --prepared-subject prepared_subject \
  --output-dir prediction \
  --device cpu
```

程序检查输入和权重后，写出 float32 `[75,Q,3]` 电场基底、查询坐标、电极对和 JSON 清单。
`Q` 为 GM/WM 查询点数量。每个电场对应源电极 `+1 mA`、Oz `-1 mA`，
向量分量为 world-RAS `[Ex,Ey,Ez]`，单位为 `V/m`。

使用内存映射可以查看基底，无需将整个数组加载到 RAM：

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

请将全部五个输出文件保存在一起。清单中的 `source_names` 给出电极轴的顺序，
查询数组给出查询轴上各行对应的位置。电流组合、分块设置和 Python API 见[推理说明](inference.md)。

<a id="configuration-files-after-wheel-installation"></a>

## wheel 安装后的配置文件

预测会自动加载内置权重和与其匹配的模型结构。训练和预检读取 `--config` 指定的配置文件。
源码目录中已提供 `configs/`；如果只安装了 wheel，先将内置默认配置复制到工作目录：

```python
from pathlib import Path

from head2field.resource_paths import CONFIG_FILES, read_packaged_text

destination = Path("configs")
destination.mkdir(exist_ok=True)
for name in CONFIG_FILES:
    with (destination / name).open("x", encoding="utf-8") as file:
        file.write(read_packaged_text("configs", name))
```

在新的工作目录中运行一次即可。示例遇到同名配置文件时会停止，不会覆盖已有配置；
复制出的文件与随包提供的默认配置相同，不会按本机或输入数据调整参数。

<a id="small-cpu-training-check"></a>

## 小型 CPU 训练检查

准备好配置文件后执行：

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

示例使用 8³ 测试模型和解析数据，包含一个虚构训练被试和一个虚构验证被试，
用于检查训练流程能否运行，不用于评估预训练模型精度。
真实数据训练和断点恢复见[训练说明](training.md)。

<a id="common-setup-issues"></a>

## 常见安装问题

| 现象 | 检查或处理 |
| --- | --- |
| 无法找到 `head2field-predict` | 激活安装软件时使用的环境；`python -m head2field.cli --help` 可打开同一预测命令。 |
| 缺少 `configs/training_smoke_v1.json` | 按上方示例复制内置默认配置，或从源码目录运行。 |
| 输出目录已存在 | 选择新目录，程序不会覆盖已有结果。 |
| 输入或坐标检查失败 | 按输入指南核对 `prepared_subject_v1.json` 及其引用文件。 |
| CUDA 不可用 | 检查当前环境中的 PyTorch 和驱动，或使用 CPU 推理。 |
| GPU 显存不足 | 先减小 `--query-chunk-size`，再减小 `--electrode-batch-size`；电极和查询点顺序不变。 |
| 自定义权重检查失败 | 使用参数名和形状符合 v1 模型要求、仅含模型参数的 FP32 权重文件。 |

权重校验值和自定义权重加载见[权重说明](../../weights/README_zh.md)，
实现细节见[代码阅读指南](code_guide.md)。
