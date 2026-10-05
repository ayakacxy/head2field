<a id="model-card-head2field-v1"></a>

# 模型说明：Head2Field v1

[English](MODEL_CARD.md) · [简体中文](MODEL_CARD_zh.md)

本软件包提供 Head2Field 的**官方实现**，对应论文：

> **Head2Field: A Continuous Neural Field for Subject-Specific Multi-Electrode
> Transcranial Electrical Stimulation Lead-Field Prediction from Structural
> MRI**

<a id="model-summary"></a>

## 模型概览

`Head2FieldV1` 是用于个体特异性 TES 电场先导场预测的 PyTorch 模型，将 T1 衍生头部表示与电极、GM/WM 查询坐标相结合。v1 使用各向同性标量电导率及以下关系：

```text
E(x; source -> reference) = G(x, source) - G(x, reference)
```

模型预测 75 个 source→Oz 电场图，输出为 float32 `[75, Q, 3]` 数组，分量采用 world-RAS `[Ex, Ey, Ez]`。每个图对应源电极 `+1 mA`、Oz `-1 mA`，电场单位为 `V/m`。推理需要 prepared inputs 和兼容权重；监督训练需要 FEM 标签，预测不需要。

<a id="versioning-and-input-contract"></a>

## 版本与输入合同

模型合同版本为 `v1`；软件版本记录在 `pyproject.toml` 和 `CITATION.cff` 中。软件补丁本身不产生新的训练模型，也不改变输入输出合同。

v1 合同固定以下内容：

- T1 是唯一原始影像模态；
- prepared input 采用 `256 x 256 x 256`、1 mm、RAS 网格；
- 十种前景组织及背景，采用以 `S/m` 为单位的 `simnibs-4.6-isotropic-v1` 标量电导率表；
- 十二个模型输入通道：十种组织指示通道、以 float32 计算的 `sigma_s_per_m / 1.654` 电导率，以及归一化 T1 强度；
- 使用固定的 v1 模型工厂和 128³ 编码器输入；
- 76 个有序 cap 电极，Oz 为参考，另有 75 个有序源电极；
- 精确的 GM/WM 查询顺序和显式体素/世界坐标转换；
- 上述电流约定与输出单位；
- 固定参数名和张量形状的严格 FP32 权重加载。

机器可读细节见[输入合同](zh/input_contract.md)、[模型结构](zh/model_architecture.md)和[数据合同](zh/data_contracts.md)。

<a id="training"></a>

## 训练

可执行的真实数据训练配置采用 AdamW，初始学习率 `2e-4`，weight decay `1e-4`。每步处理一个被试、八个 source→Oz 电极对，以及 49,152 个 GM/WM 查询位置，使用 CUDA FP16 AMP 和 1.0 的梯度范数裁剪。

每个 epoch 完整覆盖所有训练被试的源电极批次与查询分块组合。最多运行 25 个 epoch，余弦调度覆盖 25 个 epoch。每轮结束时以数据与物理组合目标验证，`a(t)=1`。连续五轮验证无改善时停止，测试评估使用验证最优 checkpoint。

数据目标采用 50/100 mm 径向分区和固定的每步缩放 `M / (75 * N_train)`，统计合同为 `training_loss_statistics_v2`。

运行时直接核对参数、分组、相对路径、文件大小、坐标顺序、shape、dtype 和有限值。训练恢复比较完整配置与数据清单；统计必须按当前 50/100 mm 设置和训练划分生成。

配置还保留了已验证权重原始训练运行的历史元数据。该记录与当前可执行训练设置分别管理，不控制训练器。

<a id="verified-weight-format"></a>

## 已验证权重格式

v1 参考权重文件包含：

| 字段 | 内容 |
| --- | --- |
| State dictionary | 裸字符串到张量的映射 |
| 张量 | 208 个连续 float32 张量 |
| 标量总数 | 2,607,870 |
| 文件大小 | 10,497,106 bytes |

文件不包含优化器、调度器、scaler、步数、epoch、指标、manifest、参与者标识、路径、采样器或 RNG 状态。张量名和形状匹配 v1 工厂。wheel 和源码分发包内置这些预训练权重，并默认加载。参见[权重说明](../weights/README_zh.md)和[数据使用说明](DATA_USE_zh.md)。

<a id="intended-use"></a>

## 预期用途

模型支持满足 v1 合同输入上的非临床研究、软件验证、方法比较及可复现 TES 电场建模实验。用户应了解神经影像空间元数据、电极与电流约定，以及学习型代理模型的数值验证。

软件包预测先导场基底，不执行刺激电流优化、治疗规划或不确定性量化。

<a id="limitations"></a>

## 局限性

- 即使张量形状正确，组织标签、电导率、仿射、电极顺序、查询顺序或单位错误也可能使预测失效。
- 模型使用固定的各向同性标量电导率；各向异性或模型/输入语义变化需要不同的模型合同。
- 训练数据不意味着模型已适用于所有扫描仪、成像协议、年龄群体、病理、病灶、植入物或特殊解剖。
- 输出没有经过校准的不确定性估计。
- 预测尚未用于诊断、治疗选择、患者安全决策或自主临床控制的验证。

实验若依赖既有评估条件之外的精度，仍需针对相应被试与 FEM 进行比较。

<a id="reproducibility-and-evaluation"></a>

## 复现与评估

仓库包含合成测试，覆盖权重与 schema 加载、坐标和源/查询合同、预处理与 FEM adapter、完整 epoch 覆盖、学习率调度、验证、早停及本地恢复。CUDA FP16 操作也已用合成输入运行验证。

这些检查用于确认软件行为与兼容性，不测量队列精度、泛化、临床有效性或与 FEM 的端到端等价性。复现科学结果需要相应的数据划分、输入准备、模型资产、评估协议及合法数据访问。

<a id="data-and-acknowledgements"></a>

## 数据与致谢

源码及 Python 分发包不包含参与者级数据和训练恢复状态。模型开发受益于 WU-Minn HCP 数据；[数据使用说明](DATA_USE_zh.md)提供上游条款与致谢参考。代码许可证、数据集条款及权重资产声明分别适用于各自材料。
