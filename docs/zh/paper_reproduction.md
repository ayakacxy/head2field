<a id="reproducibility-and-evaluation"></a>

# 复现与评估

[English](../en/paper_reproduction.md) · [简体中文](paper_reproduction.md)

Head2Field 对应论文：

> **Head2Field: A Continuous Neural Field for Subject-Specific Multi-Electrode
> Transcranial Electrical Stimulation Lead-Field Prediction from Structural
> MRI**

出版标识可用时，应添加到 `CITATION.cff`。

<a id="method-to-code-map"></a>

## 方法与代码对应

| 论文内容 | 实现 | 配置或合同 |
| --- | --- | --- |
| 第 2.2.1 节：256³、1 mm RAS T1 准备 | `preprocessing/`、`inputs.py` | `preprocessing_v1.json`；HCP ACPC 输入跳过刚体估计 |
| 第 2.2.2–2.2.3 节：标量电导率及 75 个 source→Oz FEM 电场 | `preprocessing/conductivity.py`、`fem_labels/` | `fem_labels_v1.json`；76 cap 电极；1 mA；world-RAS |
| 第 2.3 节：CNN/FPN、轴向注意力、查询/电极 token 和共享响应 | `model.py`、`_spec.py` | `head2field_v1.json`；固定 v1 权重 schema |
| 公式 7–9：能量归一化监督 | `training/cache.py`、`dataset.py`、`losses.py` | 现有训练配置；50/100 mm 径向边界；权重 0.60/0.30/0.10 |
| 公式 10–13：FEM 校准积分物理损失 | `training/physics.py`、`physics_data.py`、`steps.py` | 四项权重均为 5；Huber delta 0.1；kappa 1；10,000 步递增 |
| 第 2.5.2 节：完整 epoch、验证与早停 | `training/runner.py`、`validation.py`、`cli.py` | AdamW；25 epoch；余弦周期 25；patience 5；seed 42 |
| 缓存后的完整基底推理 | `inference.py`、`cli.py` | 有序 `[75, Q, 3]` float32 输出，1 mA 下单位为 V/m |

实现列中的路径相对于 `src/head2field/`。配置文件位于 `configs/`，相同副本随 wheel 提供。

<a id="default-training-configuration-and-derived-assets"></a>

## 默认训练配置与衍生资产

真实数据训练使用 `configs/training_v1.json`，合成安装检查使用 `configs/training_smoke_v1.json`。两者均采用以下近/中/远分区：

```text
rho = min(distance(query, source), distance(query, Oz))
near: rho < 50 mm
middle: 50 mm <= rho < 100 mm
far: rho >= 100 mm
```

统计来自每个被试的完整 FEM 电场，在查询分块之前计算。能量 floor 使用该被试内部有效源电场的中位数，不拟合跨被试统计。物理校准仅使用训练被试，随后对验证和测试保持固定。

每步数据损失缩放为 `M / (75 * N_train)`，`M` 包含完整训练 epoch 中全部源/查询组合及末尾不完整批次。验证和测试也使用由训练确定的相同缩放。

运行时直接核对参数、分组、相对路径、文件大小、坐标顺序、shape、dtype 和有限值。训练恢复比较完整配置与数据清单；统计必须按当前 50/100 mm 设置和训练划分生成。

<a id="experimental-reproduction-scope"></a>

## 实验复现范围

论文使用 200 个 HCP 被试，按家系互斥划分为 128/32/40（seed 42），嵌套训练规模为 10、20、40、80 和 128。应通过授权数据访问重建划分；源码发布不包含参与者或家系清单。训练命令接收分组互斥清单，但不重建论文中的私有家系分配。

公式 14–16 在匹配的 world-RAS 向量上计算相对 L2 和平均方向余弦。高场掩膜根据 FEM 幅值确定，百分位阈值采用包含等号的边界。先在被试内平均各源误差，再进行队列平均。幅值保留率先取被试内部源中位数，再计算队列中位数及被试 P10–P90。分块相对 L2 均值属于训练诊断，不能替代上述完整电场评估指标。

论文实验使用不同模型资产：tDCS 比较使用 80 被试模型；完整基底、TIS 和电极位移实验使用 128 被试模型。单份已验证 state dict 不提供所有训练规模、仅数据监督消融或比较模型。

发布包提供可复用基底和坐标查询推理 API。Montage 比较、TIS 优化器、位移电极 FEM 生成、队列统计检验和制图属于该源码包之外的研究流程，不能仅通过合成示例或训练命令复现其结果。

<a id="weight-and-performance-provenance"></a>

## 权重与性能来源

论文在一个代表被试的 1,392,017 个 GM/WM 查询上报告：含设备到主机传输为 8.928 s，含 float32 输出写入与刷新为 12.073 s，使用 RTX 2080 Ti、warmup 和 PyTorch 编译。约 100 倍加速指先导场生成；约 4.7 倍端到端加速估计还计入约 20 分钟分割。这些是论文测量，不是默认 eager CLI 的基准。测量其他部署时，请分别记录输入加载、预处理、模型加载、编译、warmup、传输及输出 I/O。
