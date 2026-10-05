<div align="center">

# Head2Field

[English](README.md) · [简体中文](README_zh.md)

从结构 MRI 预测个体特异性多电极 TES 先导场的连续神经场。

[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-orange?style=flat)](https://github.com/ayakacxy/head2field/releases/tag/v0.1.0)
[![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](pyproject.toml)
[![SimNIBS 4.6.0](https://img.shields.io/badge/SimNIBS-4.6.0-6D4AFF?style=flat)](docs/zh/preprocessing.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue?style=flat)](docs/LICENSE_GUIDE_zh.md)

[快速上手](#quick-start) · [文档](#documentation) · [论文复现](#reproduction) · [引用](#citation)

🧠 **结构 MRI** · ⚡ **连续神经场** · 🧭 **World-RAS** · 🔬 **物理引导训练**

[![Head2Field 项目概览](docs/assets/head2field-overview.png)](docs/assets/head2field-overview.pdf)

*图 1：A，先导场预测流程；B，模型结构。点击主图查看 PDF。*

</div>

Head2Field 根据 T1 提取的头部解剖、组织电导率和电极坐标预测可复用的多电极电场基底。
共享电极响应可用于不同电流模式的电场合成，无需为每种刺激组合重新训练。

## 📚 目录

[论文](#paper) · [主要功能](#features) · [安装](#installation) · [快速上手](#quick-start) · [训练与复现](#reproduction) · [文档导航](#documentation) · [贡献](#contributing) · [引用与许可证](#citation)

<a id="paper"></a>

## 📄 论文

本项目是以下论文的**官方实现**：

> **Head2Field: A Continuous Neural Field for Subject-Specific Multi-Electrode
> Transcranial Electrical Stimulation Lead-Field Prediction from Structural
> MRI**

中文参考译名：**Head2Field：从结构 MRI 预测个体特异性多电极 tES 先导场的连续神经场**。

当前模型为 **Head2Field v1**（`Head2FieldV1`），模型配置见
[head2field_v1.json](configs/head2field_v1.json)，Python 包名为 `head2field`。
论文方法、模型权重和复现实验的对应关系见[论文复现说明](docs/zh/paper_reproduction.md)。

<a id="features"></a>

## 🧩 主要功能

项目提供基于 PyTorch 的预处理、FEM 标签生成、物理引导训练和推理流程。v1 模型输出
75 个 source→Oz 电场图，向量分量采用 world-RAS 坐标；每个电场图对应源电极 `+1 mA`、
Oz `-1 mA`。本项目用于非临床科研；适用条件与限制见[模型说明](docs/MODEL_CARD_zh.md)。

| 工作流程 | 功能 | 依赖 |
| --- | --- | --- |
| 推理 | prepared subject 和模型权重生成 75 个先导场，无需 FEM 标签 | 基础安装 |
| 预处理 | T1 canonicalization、外部 CHARM 头模型、组织电导率、GM/WM 查询和标准电极 cap | `preprocess` extra 和 SimNIBS 4.6.0 |
| FEM 标签 | Oz 参考先导场生成、HDF5 验证和体素标签导出 | `fem` extra 和 SimNIBS 4.6.0 |
| 训练 | 数据与物理损失、完整 epoch、验证、早停和本地恢复 | `training` extra 和 CUDA 版 PyTorch |

<a id="installation"></a>

## 🐍 安装

要求 Python 3.11 或更新版本。从 [Release v0.1.0](https://github.com/ayakacxy/head2field/releases/tag/v0.1.0) 下载 wheel；其中已包含预训练 Head2Field v1 权重，安装后即可用于推理：

```bash
python -m pip install head2field-0.1.0-py3-none-any.whl
head2field-predict --help
```

也可以在源码仓库根目录安装：

```bash
python -m pip install .
```

通过 wheel 安装预处理和 FEM 所需的可选依赖：

```bash
python -m pip install './head2field-0.1.0-py3-none-any.whl[full]'
```

从源码安装时，根据需要选择可选流程：

```bash
python -m pip install '.[preprocess]'
python -m pip install '.[fem]'
python -m pip install '.[training]'
python -m pip install '.[full]'
```

真实数据训练需要 CUDA 版 PyTorch。SimNIBS/CHARM 需单独安装，预处理与 FEM adapter 要求
当前环境中的 SimNIBS 版本为 **4.6.0**。`training` extra 作为流程标识保留，其 Python 依赖
已包含在基础安装中；`full` extra 加入预处理和 FEM 所需的 SciPy 与 h5py。

从安装到推理的完整步骤、CUDA 检查、内置配置获取和常见问题见
[上手指南](docs/zh/getting_started.md)。

<a id="quick-start"></a>

## 🚀 快速上手

### 预测先导场

准备符合[输入合同](docs/zh/input_contract.md)的 prepared subject。
命令会自动加载[内置的预训练 v1 模型权重](weights/README_zh.md)，无需单独下载权重。

```bash
head2field-predict \
  --prepared-subject prepared_subject \
  --output-dir prediction \
  --device cpu
```

GPU 推理可使用 `--device cuda`。输出包含形状为 `[75, Q, 3]` 的
`lead_field_float32.npy`、精确查询坐标、电极对记录，以及包含输入输出元数据 的 manifest。
向量分量为 world-RAS `[Ex, Ey, Ez]`，电场在固定 1 mA 电流约定下以 `V/m` 表示。
分块设置与 Python API 见[推理说明](docs/zh/inference.md)。

使用自定义权重时，加上 `--weights /path/to/custom_v1.pth`。

### 预处理原始 T1

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

`non_hcp_raw` 执行一次 6-DOF 刚体配准，使用前需检查 QC 图。对于已经完成 ACPC 对齐和畸变
校正的 HCP 输入，使用 `hcp_acpc`，该流程跳过刚体估计。所有输出目录必须尚不存在。
详细说明见[预处理](docs/zh/preprocessing.md)。

### 用合成数据验证训练安装

下面的例子使用小型 CPU FP32 模型，不需要参与者数据或 SimNIBS。

命令使用源码目录中的 `configs/`。如果只安装了 wheel，先按照
[内置配置获取说明](docs/zh/getting_started.md#configuration-files-after-wheel-installation)
将默认配置复制到当前工作目录。

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

这些非解剖的解析合成输入用于验证软件流程，不用于衡量预测精度。
更多示例见[examples](examples/README_zh.md)。

<a id="reproduction"></a>

## 🔬 训练与论文复现

默认训练配置为 [training_v1.json](configs/training_v1.json)，径向分区阈值为 **50/100 mm**。
使用 AdamW（学习率 `2e-4`、weight decay `1e-4`）和 CUDA FP16，每步包含八个
source→Oz 电极对及 49,152 个查询。完整 epoch 训练最多运行 25 轮，采用 25 轮余弦调度，
每轮结束验证；连续五轮验证损失未降低时早停，测试使用验证最优模型。

真实数据训练需要 prepared subjects、FEM 标签和 CUDA 版 PyTorch。数据准备、训练配置、
验证与 checkpoint 恢复见[训练说明](docs/zh/training.md)。

模型与权重的对应关系、缓存合同、评估口径和性能计时范围见
[论文复现说明](docs/zh/paper_reproduction.md)。

<a id="documentation"></a>

## 🗂️ 文档导航

- [完整文档目录](docs/zh/README.md)
- [上手指南：安装、输入准备与推理](docs/zh/getting_started.md)
- [代码阅读指南：模块职责、张量形状与执行流程](docs/zh/code_guide.md)
- [输入格式与坐标约定](docs/zh/input_contract.md)
- [T1 预处理与 CHARM 集成](docs/zh/preprocessing.md)
- [FEM 标签生成与验证](docs/zh/fem_label_generation.md)
- [训练、验证与恢复](docs/zh/training.md)
- [推理与输出格式](docs/zh/inference.md)
- [模型结构](docs/zh/model_architecture.md)
- [论文与代码对应及复现范围](docs/zh/paper_reproduction.md)
- [机器可读数据合同](docs/zh/data_contracts.md)
- [模型说明](docs/MODEL_CARD_zh.md)、[数据使用](docs/DATA_USE_zh.md)和[权重格式](weights/README_zh.md)

<a id="contributing"></a>

## 🤝 贡献

开发环境与验证命令见[贡献指南](docs/CONTRIBUTING_zh.md)。实现来源、软件变更和发布要求分别见
[SOURCE_PROVENANCE_zh.md](docs/SOURCE_PROVENANCE_zh.md)、[CHANGELOG_zh.md](docs/CHANGELOG_zh.md) 和
[发布清单](docs/RELEASE_CHECKLIST_zh.md)。

公开 API 名称和 CLI 帮助使用英文，新增源码注释与 docstring 使用中文。全部公开指南均提供英文与简体中文版本，
每页顶部可切换同一篇文档的语言。

<a id="citation"></a>

## 🙏 许可证、引用与致谢

源码采用 [GPL-3.0-only](docs/LICENSE_GUIDE_zh.md) 许可证。第三方软件和模型权重遵循各自的分发说明，详见
[THIRD_PARTY_NOTICES_zh.md](docs/THIRD_PARTY_NOTICES_zh.md) 和 [DATA_USE_zh.md](docs/DATA_USE_zh.md)。

在科研中使用 Head2Field 时，可直接使用以下 BibTeX 引用论文：

```bibtex
@misc{chen_head2field,
  author = {Chen, Xiayu and Wang, Xiaoxiao and Cao, Zixin and
            Liu, Senlin and Liu, Zhichen and Wang, Zhengyang and
            Du, Huiyu and Qiu, Bensheng and Wang, Yanming},
  title  = {{Head2Field}: A Continuous Neural Field for Subject-Specific
            Multi-Electrode Transcranial Electrical Stimulation
            Lead-Field Prediction from Structural {MRI}},
  note   = {Manuscript}
}
```

题目与作者顺序对应 [CITATION.cff](CITATION.cff)。该文件尚未记录发表年份、期刊和 DOI。
软件、相关方法与数据来源的引用要求见[引用指南](docs/CITATION_GUIDE_zh.md)。

模型开发受益于 WU-Minn Human Connectome Project 数据，[数据使用说明](docs/DATA_USE_zh.md)
提供 HCP 致谢和方法引用要求的官方链接。源码与 Python 分发包不包含参与者影像、mesh、
FEM 输出、cohort manifest 或训练恢复状态。
