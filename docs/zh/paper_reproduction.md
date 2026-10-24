<a id="reproducibility-and-evaluation"></a>

# 评估与可重复实验

[English](../en/paper_reproduction.md) · [简体中文](paper_reproduction.md)

设计实验或报告结果时，可按本指南记录数据划分、训练统计、FEM 对比和运行时间。操作命令见[训练说明](training.md)，预测 API 见[推理说明](inference.md)。

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

上表的源码路径相对于 `src/head2field/`。配置文件位于 `configs/`，wheel 中也包含相同文件。

<a id="default-training-configuration-and-derived-assets"></a>

## 训练配置与衍生数据

真实数据配置为 `configs/training_v1.json`，合成安装配置为 `configs/training_smoke_v1.json`。两者使用相同的近、中、远分区：

```text
rho = min(distance(query, source), distance(query, Oz))
near: rho < 50 mm
middle: 50 mm <= rho < 100 mm
far: rho >= 100 mm
```

能量统计应在查询分块前，根据每个被试的完整 FEM 电场计算。分母下限使用该被试内部有效源电场的中位数。物理校准只用训练被试拟合，再原样用于验证和测试被试。

每步数据损失使用缩放 `M / (75 * N_train)`，其中 `M` 为一个完整训练 epoch 的全部源/查询组合数，也包含末尾不完整批次。验证和测试沿用训练分区的缩放。修改径向分区或训练划分后，应重新计算统计量。

加载器检查预处理数据、FEM 标签、目标缓存和物理采样数据的格式与数值。继续已保存的训练时，需要相同的配置和数据集记录；只加载推理权重则会重新初始化优化器和训练状态。

<a id="experimental-reproduction-scope"></a>

## 数据划分与评估

将各组分别分配给训练、验证或可选的测试分区，分区之间不能重叠。研究设计有此要求时，将有关联的参与者放入同一组。实验记录应保存分组、随机种子、预处理设置和权重校验值。输入数据集及参与者清单需另行获取。

按查询行和源电极匹配预测与 FEM 向量，确认坐标系、电流约定和单位一致后，再计算完整电场指标：

- 在匹配向量上计算相对 L2 误差和平均方向余弦。
- 根据 FEM 幅值确定高场掩膜，记录分位数阈值及等于阈值时的处理规则。
- 先在每个被试内平均源误差，再计算被试之间的平均值。
- 幅值保留率先计算各被试内部的源中位数，再报告队列中位数和被试 P10–P90。

单个分块上的训练诊断不能直接作为完整电场指标。报告每项指标时，应说明使用的聚合方式和掩膜。

安装包包含一份预训练 state dictionary。比较数据划分、训练规模、随机种子、消融或基线模型时，应为每种设置分别训练并评估权重。合成示例只用小模型检查软件流程，不测量预训练模型的精度。

<a id="weight-and-performance-provenance"></a>

## 运行时间测量

每项计时结果应附上硬件、软件版本、查询数量、分块大小和执行模式。分别记录输入加载、预处理、模型加载、实际使用的编译步骤、预热、推理、设备到主机传输和输出 I/O。

GPU 计时前后应进行 CUDA 同步。冷启动与复用模型或被试缓存的重复运行分别报告，并注明是否计入输出写入和刷新。端到端时间应覆盖所述流程需要的全部阶段；比较先导场推理本身时，单独报告该阶段。

## 结果使用

可以使用预测基底或 Python 查询接口编写电场合成与评估程序。电流优化、队列统计和绘图由应用程序处理。派生结果应与基底清单、源电流、查询坐标、权重文件名和校验值一同保存。用途与局限见[模型说明](../MODEL_CARD_zh.md)，参考文献见[引用指南](../CITATION_GUIDE_zh.md)。
