<a id="deterministic-synthetic-examples"></a>

# 合成数据示例

[English](README.md) · [简体中文](README_zh.md)

这些小型非解剖数据用于检查安装和软件流程，不含真实参与者影像、标识、坐标、标签或衍生数据。
生成文件应写入版本控制忽略的新目录；本地开发可使用 `.work/` 下的任务目录。

<a id="preprocessing-branch-inputs"></a>

## 预处理分支输入

```bash
python examples/generate_synthetic_preprocessing_inputs.py \
  --output-dir generated/synthetic-preprocessing

python -m head2field.preprocessing canonicalize \
  --input-t1 generated/synthetic-preprocessing/synthetic_raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template generated/synthetic-preprocessing/synthetic_orientation_template.nii.gz \
  --output-dir generated/synthetic-canonical
```

生成器还会为 adapter 测试创建电极布局覆盖文件，包含要求的 76 个名称，坐标位于合成球面上。
这些输入不代表真实头部，不能用来判断 CHARM 分割或 FEM 电场的精度。

<a id="training-installation-and-resume-smoke"></a>

## 训练安装与恢复检查

以下示例适用于源码安装和 wheel 安装。
源码目录已包含 `configs/`；wheel 用户请先按照
[内置配置获取说明](../docs/zh/getting_started.md#configuration-files-after-wheel-installation)
准备配置文件。

```bash
head2field-make-synthetic-training \
  --output-dir generated/synthetic-training

head2field-train-preflight \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json

head2field-train \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset generated/synthetic-training/synthetic_training_dataset_v1.json \
  --output-dir generated/synthetic-run
```

数据集包含一个虚构训练被试和一个虚构验证被试，两者分组互斥。
数据保留固定的 75 源电极顺序，并在 8³ 网格上使用解析数组。

训练器采用仅供测试的小型 CPU FP32 网络，检查安装、训练步、保存状态、RNG 状态、采样和恢复。
它不训练完整预训练结构，也不测量电场预测精度。

尚未安装 CLI 时，可通过 `generate_synthetic_training_dataset.py` 生成相同的数据集。
