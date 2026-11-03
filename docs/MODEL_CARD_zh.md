<a id="model-card-head2field-v1"></a>

# 模型说明：Head2Field v1

[English](MODEL_CARD.md) · [简体中文](MODEL_CARD_zh.md)

本页介绍模型输入输出、训练设置、权重和适用范围。相关方法见[引用指南](CITATION_GUIDE_zh.md)。

<a id="model-summary"></a>

## 模型概览

`Head2FieldV1` 根据 T1 衍生的头部表示、电极位置和 GM/WM 查询坐标，预测个体特异性 TES
先导场。模型使用各向同性标量电导率，并将源电极到参考电极的电场表示为两个共享电极响应之差：

```text
E(x; source -> reference) = G(x, source) - G(x, reference)
```

输出是 float32 `[75, Q, 3]` 数组，即 `Q` 个查询位置上的 75 个 source→Oz 电场图。
向量分量为 world-RAS `[Ex, Ey, Ez]`；源电极电流为 `+1 mA`、Oz 为 `-1 mA`，
电场单位为 `V/m`。预测需要预处理后的输入和兼容权重，FEM 标签用于监督训练。

<a id="versioning-and-input-contract"></a>

## 模型版本与输入要求

模型版本为 `v1`，软件版本单独记录在 `pyproject.toml` 和 `CITATION.cff` 中。
更新软件不会自动重新训练模型，也不会自动改变输入输出格式。

模型要求：

- T1 是唯一原始影像模态；
- 预处理后的输入位于 `256 x 256 x 256`、1 mm、RAS 网格；
- 十种前景组织及背景，电导率采用以 `S/m` 为单位的 `simnibs-4.6-isotropic-v1` 标量表；
- 十二个输入通道：十种组织指示通道、以 float32 计算的 `sigma_s_per_m / 1.654` 电导率，
  以及归一化 T1 强度；
- 使用固定的 v1 模型创建函数，编码器输入大小为 128³；
- 76 个按固定顺序排列的 cap 电极，其中 Oz 为参考，另有 75 个源电极；
- GM/WM 查询顺序与输入记录一致，体素坐标和世界坐标通过影像 affine 转换；
- 电流约定和输出单位与上文一致；
- FP32 权重的参数名和张量形状符合模型要求。

文件结构与检查细节见[输入格式说明](zh/input_contract.md)和[数据格式与检查](zh/data_contracts.md)，
网络结构见[模型结构](zh/model_architecture.md)。

<a id="training"></a>

## 训练

默认真实数据配置采用 AdamW，初始学习率为 `2e-4`，weight decay 为 `1e-4`，
使用 CUDA FP16 AMP，并将梯度范数裁剪到 1.0。每步处理一个被试、八个 source→Oz
电极对，以及一块包含 49,152 个位置的 GM/WM 查询。

每个 epoch 覆盖所有训练被试的全部源电极批次与查询分块组合。训练最多进行 25 个 epoch，
余弦调度覆盖 25 个 epoch。每轮结束时，用数据与物理组合目标进行验证，`a(t)=1`。
连续五轮没有改善时停止训练，测试评估使用验证结果最好的 checkpoint。

数据损失采用 50/100 mm 径向分区、每步缩放 `M / (75 * N_train)`，以及
`training_loss_statistics_v2` 格式的统计量，并与环流、WM/GM 法向电流连续性、
切向电场连续性和通量损失结合。物理阈值只用训练数据拟合。
运行配置记录分组互斥的训练、验证和测试划分，以及对应的数据路径。

准备数据、设置损失和恢复训练的步骤见[训练说明](zh/training.md)。

<a id="verified-weight-format"></a>

## 内置权重格式

| 字段 | 内容 |
| --- | --- |
| State dictionary | 仅含模型参数名与张量的字典 |
| 张量 | 208 个连续 float32 张量 |
| 标量总数 | 2,607,870 |
| 文件大小 | 10,497,106 bytes |

文件只保存模型张量，不含优化器、调度器、scaler、步数、epoch、指标、manifest、
参与者标识、路径、采样器或 RNG 状态。参数名和张量形状匹配 v1 模型创建函数。
wheel 和源码包均包含这些权重，并默认加载。
详细说明见[权重说明](../weights/README_zh.md)和[数据使用说明](DATA_USE_zh.md)。

<a id="intended-use"></a>

## 预期用途

Head2Field 支持 TES 电场建模、方法比较和个体化刺激规划研究。预测导联场基可复用于
tDCS 电场合成和 TIS 包络预测。论文中的 TIS 优化实验将该基底用于独立的下游优化器；
软件包提供基底预测和 Python 查询接口，供这些应用调用。

<a id="limitations"></a>

## 局限性

- 评估队列包含 WU-Minn HCP 的 200 名健康成人，年龄为 22–36 岁。
  训练、验证和测试分别为 128、32 和 40 人，同一家系分配到同一分区。
- 组织电导率固定，采用各向同性设置。
- 电极采用固定圆柱形几何，直径 10 mm、厚度 4 mm；位置泛化实验测试了 5–20 mm 位移。
- 模型输出电场向量，未包含不确定性估计。

论文将更广年龄范围和患者队列、各向异性电导率、可变电极几何列为后续研究方向。

<a id="reproducibility-and-evaluation"></a>

## 复现与评估

论文中，128 人训练模型的平均全场相对 L2 误差为 0.070，去除物理正则化后为 0.078。
在测试的未见电极位置上，无需微调的误差为 0.069–0.070。用预测基底优化的 TIS 方案经 FEM
评价后，达到 FEM 优化方案聚焦性得分的 94.9%。队列、比较结果和计时条件见
[评估指南](zh/paper_reproduction.md#manuscript-results)。

合成测试覆盖权重与 schema 加载、坐标、源电极和查询顺序、预处理与 FEM adapter、
完整 epoch、学习率调度、验证、早停及本地恢复。CUDA FP16 操作也已使用合成输入运行验证。

合成测试检查软件行为。论文中的电场精度结果使用评估指南所述的队列数据、训练模型和 FEM 比较。

<a id="data-and-acknowledgements"></a>

## 数据与致谢

源码及 Python 安装包不包含参与者级数据和完整训练恢复状态。模型开发受益于 WU-Minn HCP 数据。
[数据使用说明](DATA_USE_zh.md)列出了上游条款及致谢参考。代码许可证、数据集条款和权重声明
分别适用于各自材料。
