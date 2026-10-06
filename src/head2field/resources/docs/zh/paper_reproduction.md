<a id="reproducibility-and-evaluation"></a>

# 评估与可重复实验

[English](../en/paper_reproduction.md) · [简体中文](paper_reproduction.md)

本指南说明如何准备可重复的训练运行、将预测电场与 FEM 参考比较，以及测量运行时间。运行命令见[训练说明](training.md)，预测 API 见[推理说明](inference.md)。

<a id="method-to-code-map"></a>

## 工作流程与实现

| 阶段 | 实现 | 配置或输出 |
| --- | --- | --- |
| T1 准备 | `preprocessing/`、`inputs.py` | `preprocessing_v1.json`；256³、1 mm RAS 网格 |
| 电导率与 FEM 标签 | `preprocessing/conductivity.py`、`fem_labels/` | `fem_labels_v1.json`；1 mA 下的 75 个 source→Oz 电场 |
| 解剖编码与电场预测 | `model.py`、`_spec.py` | `head2field_v1.json`；固定 v1 结构 |
| 能量归一化监督 | `training/cache.py`、`training/dataset.py`、`training/losses.py` | 完整电场统计；50/100 mm 分区；权重 0.60/0.30/0.10 |
| 积分物理损失 | `training/physics.py`、`training/physics_data.py`、`training/steps.py` | 仅使用训练集校准；四项权重均为 5；10,000 步递增 |
| epoch 训练与验证 | `training/runner.py`、`training/validation.py`、`training/cli.py` | AdamW；最多 25 个 epoch；余弦调度；patience 5 |
| 完整基底推理 | `inference.py`、`cli.py` | 有序 `[75, Q, 3]` float32 输出，world-RAS，单位 V/m |

实现路径相对于 `src/head2field/`。配置文件位于 `configs/`，相同副本随 wheel 提供。

<a id="default-training-configuration-and-derived-assets"></a>

## 训练配置与衍生数据

真实数据训练使用 `configs/training_v1.json`，合成安装检查使用 `configs/training_smoke_v1.json`。两者均定义以下近/中/远分区：

```text
rho = min(distance(query, source), distance(query, Oz))
near: rho < 50 mm
middle: 50 mm <= rho < 100 mm
far: rho >= 100 mm
```

能量统计使用查询分块之前的被试完整 FEM 电场。分母 floor 采用该被试内部有效源电场的中位数。物理校准仅在训练被试上拟合，再将同一校准应用于验证和测试被试。

每步数据损失缩放为 `M / (75 * N_train)`，其中 `M` 统计完整训练 epoch 内全部源/查询组合，包含末尾不完整批次。验证和测试使用训练分区确定的缩放。径向设置或训练划分变化时，应重新生成统计量。

prepared subject、FEM 标签、目标缓存和物理池具有明确的验证合同。恢复训练要求保存的配置及数据集合同与新运行一致。由推理 state dictionary 初始化时，优化器与训练状态重新创建。

<a id="experimental-reproduction-scope"></a>

## 数据划分与评估

使用训练、验证和可选测试分组互斥的数据集清单。研究设计要求时，将有关联的参与者放入同一组。随实验记录保存分组、随机种子、预处理设置和权重校验值。输入数据集与参与者清单由用户提供。

预测与 FEM 向量应在相同查询行上比较，并保持源顺序、坐标系、电流约定和单位一致。完整电场评估应明确以下定义：

- 在匹配向量上计算相对 L2 误差和平均方向余弦。
- 根据 FEM 幅值确定高场掩膜，记录分位数阈值及等于阈值时的处理规则。
- 先在每个被试内平均源误差，再计算被试之间的平均值。
- 幅值保留率先计算各被试内部的源中位数，再报告队列中位数和被试 P10–P90。

分块训练诊断与完整电场评估汇总的量不同，每个报告指标均应记录其聚合方式和掩膜。

安装包提供一份预训练 v1 state dictionary。比较不同数据划分、训练规模、随机种子、消融或基线模型时，需要分别训练和评估对应权重。合成示例使用小型测试模型检查软件流程，不测量预训练模型精度。

<a id="weight-and-performance-provenance"></a>

## 运行时间测量

记录硬件、软件版本、查询数量、分块大小及执行模式，分别测量输入加载、预处理、模型加载、实际使用的编译步骤、warmup、推理、设备到主机传输和输出 I/O。

GPU 执行时在计时边界进行 CUDA 同步。区分冷启动和复用模型或被试缓存的重复运行，并说明是否计入输出写入与刷新。端到端测量应包含目标流程所需的全部阶段；只比较先导场推理时，应单独报告该阶段。

## 结果使用

可复用基底与 Python 查询 API 支持应用层电场合成和评估。电流优化、队列统计分析与制图属于独立的应用流程。保存衍生结果时，同时保留基底清单、源电流、查询坐标和权重身份。用途与局限见[模型说明](../MODEL_CARD_zh.md)，参考文献见[引用指南](../CITATION_GUIDE_zh.md)。
