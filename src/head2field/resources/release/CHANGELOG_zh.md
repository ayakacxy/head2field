<a id="changelog"></a>

# 更新日志

[English](CHANGELOG.md) · [简体中文](CHANGELOG_zh.md)

本文件记录软件的重要变化。模型合同版本 `v1` 与软件版本分别管理；更新软件包本身并不意味着生成了新的训练模型。

<a id="010rc1"></a>

## 0.1.0rc1

<a id="documentation-and-code-readability"></a>

### 文档与代码可读性

- 所有公开指南提供英文与简体中文版本，正文保持当前语言跳转，每页可切换对应语言，并随安装包提供副本。
- 技术指南按 `docs/en/` 和 `docs/zh/` 分类，两边使用相同文件名；README 阅读入口跳转到对应语言的指南。

- 新增安装后使用指南和代码阅读指南，补充推理分块与源电流组合说明，并同步安装包内的技术文档。
- 明确核心张量与坐标 API，完善英文报错；将推理中的电极对索引重映射提取为独立函数，改进稳定能量损失变量命名，保留预训练权重、模型参数名和现有配置值。

<a id="bundled-pretrained-model"></a>

### 内置预训练模型

- wheel 和源码分发包包含已验证的 Head2Field v1 权重及其校验文件。
- 推理默认使用内置预训练模型；`--weights` 可指定兼容的自定义权重。
- `load_head2field_v1_weights()` 在未指定路径时加载内置模型。

<a id="head2field-naming"></a>

### Head2Field 命名

- 软件包与导入命名空间为 `head2field`，命令采用 `head2field-*`。
- 版本化模型 API 为 `Head2FieldV1`、`create_head2field_v1()`、`load_head2field_v1_weights()` 和 `Head2FieldInferenceSession`。
- 模型配置为 `configs/head2field_v1.json`，权重文件为 `head2field_v1_weights.pth`。执行配置、损失、模型结构、算子及预测标识采用 `head2field` 前缀。

<a id="default-training-settings"></a>

### 默认训练设置

- 默认训练配置、统计生成与批次读取使用 50/100 mm 径向分区。
- 能量分母采用每个被试内部各源电场的中位数，记录在 `training_loss_statistics_v2` 清单中。
- 每步监督损失缩放为 `M / (75 * N_train)`，包含末尾不完整批次，并绑定到训练运行合同。
- 补充方法与代码对应、实验范围、计时来源说明，更新论文标题与引用元数据。

<a id="workflows"></a>

### 工作流程

运行时直接核对参数、分组、相对路径、文件大小、坐标顺序、shape、dtype 和有限值。训练恢复比较完整配置与数据清单；统计必须按当前 50/100 mm 设置和训练划分生成。

<a id="training"></a>

### 训练

- 使用 CUDA FP16 AMP 和 AdamW，学习率为 `2e-4`，weight decay 为 `1e-4`；每步八个 source→Oz 电极对，查询分块包含 49,152 个 GM/WM 位置。
- 每个 epoch 完整遍历源电极与查询组合，最多运行 25 个 epoch，采用覆盖 25 个 epoch 的余弦调度。
- 每轮结束时以 `a(t)=1` 验证组合目标；连续五轮无改善时早停，导出验证最优 checkpoint，并支持可选的测试集评估。
- 提供独立的 CPU FP32 合成数据配置，用于安装与恢复检查。

<a id="documentation-and-packaging"></a>

### 文档与打包

- 提供英文和中文 README、流程指南、模型与数据使用说明、贡献指南和可复用发布清单。
- 包含确定性合成示例、schema、包资源、依赖声明和实现来源记录。
- 内置仅含张量的预训练权重；Python 分发包排除训练恢复状态和自定义 checkpoint。
