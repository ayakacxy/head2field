<a id="model-card-head2field-v1"></a>

# 模型说明：Head2Field v1

[English](MODEL_CARD.md) · [简体中文](MODEL_CARD_zh.md)

本文介绍 Head2Field v1 的输入输出、默认训练配置、内置权重及预期用途。方法参考文献见[引用指南](CITATION_GUIDE_zh.md)。

<a id="model-summary"></a>

## 模型概览

`Head2FieldV1` 是用于个体特异性 TES 电场先导场预测的 PyTorch 模型，将 T1 衍生头部表示与电极、GM/WM 查询坐标相结合。v1 使用各向同性标量电导率及以下关系：

```text
E(x; source -> reference) = G(x, source) - G(x, reference)
```

模型预测 75 个 source→Oz 电场图，输出为 float32 `[75, Q, 3]` 数组，分量采用 world-RAS `[Ex, Ey, Ez]`。每个图对应源电极 `+1 mA`、Oz `-1 mA`，电场单位为 `V/m`。推理需要 prepared inputs 和兼容权重；监督训练需要 FEM 标签，预测不需要。

<a id="versioning-and-input-contract"></a>

## 模型版本与输入要求

模型版本为 `v1`；软件版本记录在 `pyproject.toml` 和 `CITATION.cff` 中。更新软件不会自动重新训练模型或改变输入输出格式。

Head2Field v1 使用以下输入和输出格式：

- T1 是唯一原始影像模态；
- prepared input 采用 `256 x 256 x 256`、1 mm、RAS 网格；
- 十种前景组织及背景，采用以 `S/m` 为单位的 `simnibs-4.6-isotropic-v1` 标量电导率表；
- 十二个模型输入通道：十种组织指示通道、以 float32 计算的 `sigma_s_per_m / 1.654` 电导率，以及归一化 T1 强度；
- 使用 v1 模型创建函数，编码器输入大小为 128³；
- 76 个有序 cap 电极，Oz 为参考，另有 75 个有序源电极；
- 与输入文件一致的 GM/WM 查询顺序，以及通过 affine 进行的体素/世界坐标转换；
- 上述电流约定与输出单位；
- 固定参数名和张量形状的严格 FP32 权重加载。

文件结构、检查方法和网络结构分别见[输入格式说明](zh/input_contract.md)、[模型结构](zh/model_architecture.md)和[数据格式与检查](zh/data_contracts.md)。

<a id="training"></a>

## 训练

默认真实数据训练配置采用 AdamW，初始学习率 `2e-4`，weight decay `1e-4`。每步处理一个被试、八个 source→Oz 电极对，以及 49,152 个 GM/WM 查询位置，使用 CUDA FP16 AMP 和 1.0 的梯度范数裁剪。

每个 epoch 完整覆盖所有训练被试的源电极批次与查询分块组合。最多运行 25 个 epoch，余弦调度覆盖 25 个 epoch。每轮结束时以数据与物理组合目标验证，`a(t)=1`。连续五轮验证无改善时停止，测试评估使用验证最优 checkpoint。

数据目标采用 50/100 mm 径向分区和固定的每步缩放 `M / (75 * N_train)`，统计量保存在 `training_loss_statistics_v2`。

训练目标包含稳定能量数据损失、环流、WM/GM 法向电流连续性、切向电场连续性和通量项。物理阈值由训练数据拟合。训练、验证和测试采用分组互斥划分，训练记录同时保存数据文件路径，以便检查恢复时的数据是否一致。损失设置、数据准备与恢复步骤见[训练说明](zh/training.md)。

<a id="verified-weight-format"></a>

## 内置权重格式

内置 v1 权重文件包含：

| 字段 | 内容 |
| --- | --- |
| State dictionary | 裸字符串到张量的映射 |
| 张量 | 208 个连续 float32 张量 |
| 标量总数 | 2,607,870 |
| 文件大小 | 10,497,106 bytes |

文件不包含优化器、调度器、scaler、步数、epoch、指标、manifest、参与者标识、路径、采样器或 RNG 状态。张量名和形状匹配 v1 工厂。wheel 和源码包内置这些预训练权重，并默认加载。参见[权重说明](../weights/README_zh.md)和[数据使用说明](DATA_USE_zh.md)。

<a id="intended-use"></a>

## 预期用途

Head2Field 用于科研中的 TES 电场预测、方法比较和软件测试。使用前应按输入说明准备数据，确认影像坐标、电极顺序、电流和单位，并在所需应用条件下与 FEM 结果比较。

软件包预测先导场基底，不执行刺激电流优化、治疗规划或不确定性量化。

<a id="limitations"></a>

## 局限性

- 即使张量形状正确，组织标签、电导率、仿射、电极顺序、查询顺序或单位错误也可能使预测失效。
- 模型使用固定的各向同性标量电导率；使用各向异性电导率，或改变输入通道和组织标签时，需要相应的新模型与权重。
- 训练数据不意味着模型已适用于所有扫描仪、成像协议、年龄群体、病理、病灶、植入物或特殊解剖。
- 输出没有经过校准的不确定性估计。
- 预测尚未用于诊断、治疗选择、患者安全决策或自主临床控制的验证。

实验若依赖既有评估条件之外的精度，仍需针对相应被试与 FEM 进行比较。

<a id="reproducibility-and-evaluation"></a>

## 复现与评估

仓库包含合成测试，覆盖权重与 schema 加载、坐标和源电极和查询点顺序、预处理与 FEM adapter、完整 epoch 覆盖、学习率调度、验证、早停及本地恢复。CUDA FP16 操作也已用合成输入运行验证。

这些测试用于检查程序是否正确运行。模型精度、泛化能力和临床适用性需要另外评估；与 FEM 比较时，应使用对应的数据划分、预处理方法、权重和评价指标，并遵守数据访问条款。

<a id="data-and-acknowledgements"></a>

## 数据与致谢

源码及 Python 安装包不包含参与者级数据和训练恢复状态。模型开发受益于 WU-Minn HCP 数据；[数据使用说明](DATA_USE_zh.md)提供上游条款与致谢参考。代码许可证、数据集条款及权重使用说明分别适用于各自材料。
