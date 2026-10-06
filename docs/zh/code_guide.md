<a id="code-guide"></a>

# 代码阅读指南

[English](../en/code_guide.md) · [简体中文](code_guide.md)

先阅读定义 v1 模型创建函数的 [`_spec.py`](../../src/head2field/_spec.py)、负责张量构建的 [`inputs.py`](../../src/head2field/inputs.py) 和组织预测流程的 [`inference.py`](../../src/head2field/inference.py)。完整网络位于 [`model.py`](../../src/head2field/model.py)。四个文件使用相同的坐标表示、电极顺序和查询点索引。

<a id="package-map"></a>

## 软件包结构

| 模块 | 职责 |
| --- | --- |
| `_spec.py` | 模型创建函数、结构参数和损失名称 |
| `weights.py` | 内置权重校验、裸 FP32 state dictionary 验证及严格加载 |
| `inputs.py` | 头部通道、坐标 bundle、查询特征及稳定的电极对索引 |
| `model.py` | 坐标验证、采样、编码器/FPN、轴向 mixer、token 与共享响应解码 |
| `inference.py` | 单被试缓存、电极批次、查询分块及有序预测 |
| `cli.py` | 预测参数解析、输入验证、输出数组及清单 |
| `preprocessing/` | T1 对齐、CHARM adapter、电导率、cap 及 prepared subject 组装 |
| `fem_labels/` | SimNIBS 先导场执行、HDF5 审计和体素标签导出 |
| `training/` | 训练数据与缓存格式、损失统计、物理损失采样数据、训练及恢复 |
| `resource_paths.py` | 访问随包配置、schema、文档和权重文件 |

顶层包导出 `Head2FieldV1`、`create_head2field_v1`、`load_head2field_v1_weights` 和 `Head2FieldInferenceSession`。预处理、FEM 和训练使用各自的子模块，只有调用这些功能时才加载相关依赖。

<a id="prediction-flow"></a>

## 预测流程

```mermaid
flowchart LR
    P[准备好的被试输入] --> V[验证数组、坐标与顺序]
    V --> I[构造输入张量]
    W[内置或自定义权重] --> S[推理会话]
    I --> S
    S --> H[缓存头部与电极特征]
    H --> Q[编码查询分块]
    Q --> G[解码共享电极响应]
    G --> E[提取源电极减参考电极的差值]
    E --> O[写入数组与清单]
```

`predict_prepared_subject_v1()` 负责读取输入并写出预测结果。推理会话以评估模式运行模型，缓存头部 FPN、全局被试特征、电极 token 和部分查询特征。后续查询分块复用已有头部和电极编码，因此不必重复计算。

解码一个电极对批次时，程序只计算该批次涉及的电极。多个源电极共用同一参考时，会复用参考电极的特征和响应。电极首次出现的顺序和请求的电极对顺序保持不变，以减少显存占用。

切换被试或模型权重时，请新建推理会话。已有缓存不会因输入文件被修改而自动更新。

<a id="tensor-and-coordinate-notation"></a>

## 张量与坐标记号

`B` 表示被试、`U` 表示唯一电极、`P` 表示电极对、`Q` 表示查询、`C` 表示特征通道。文件级 v1 推理采用 `B=1`、`U=76`、`P=75`；底层 bundle 也支持选定或重复电极对。

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| Canonical 组织/T1/电导率数组 | `[256,256,256]` | 原始 1 mm i/j/k 网格 |
| 编码器头部输入 | `[B,12,128,128,128]` | 十种组织、归一化电导率与 T1 |
| 查询行索引 | `[B,Q]` | `gmwm_ijk` 中的精确行索引 |
| 查询坐标 | `[B,Q,3]` | 体素 i/j/k、归一化 i/j/k 和 world-RAS mm 字段 |
| 原始网格查询特征 | `[B,Q,12]` | 来自 1 mm 网格的组织、电导率与 T1 特征 |
| 唯一电极坐标 | `[B,U,3]` | 归一化 i/j/k 及 world-RAS mm |
| 电极对索引 | `[B,P,2]` | 唯一电极库中的源和参考位置 |
| 共享响应库 | `[B,U,Q,3]` | `G(head,query,electrode)` |
| 电极对电场 | `[B,P,Q,3]` | world-RAS 中的 `G(source)-G(reference)` |
| 保存的预测 | `[75,Q,3]` | 单被试电极对电场，采用 1 mA 约定，单位 V/m |

归一化坐标采用 `2 * ijk / (shape - 1) - 1`，对应 `align_corners=True`。128³ 编码器输入覆盖与 256³ 网格相同的物理范围。`sample_volume_at_points()` 将 i/j/k 反转为 k/j/i，以适应 PyTorch 采样网格的 W/H/D 顺序。即使网络使用 autocast，仿射计算仍保留原精度。

解剖原点为 GM/WM 查询位置的均值。相对坐标特征以此为原点；绝对坐标分支接受当前数据或物理分支指定的受控增强。

<a id="reading-the-network"></a>

## 阅读网络

在 `model.py` 中依次阅读：

1. `SubjectGeometry`、`QueryCoordinateBundle` 和 `ElectrodeCoordinateBundle` 保存电极和查询点索引，并检查坐标转换。
2. `AnatomyPyramidEncoder` 与 `CoarseGlobalAnatomyMixer` 生成解剖特征；`Head2FieldV1.encode_head()` 构建四层 FPN。
3. `build_global_context()`、`encode_queries()` 和 `encode_unique_electrodes()` 生成被试、查询及电极 token。
4. `decode_response_bank()` 对每个唯一电极执行共享解码；`gather_pair_field()` 根据电极对索引计算源响应与参考响应之差。
5. `forward_bundled()` 评估一个分支；`forward_data_and_physics()` 为两个分支共享头部编码。

各电极对都由共享响应相减得到，因此满足反对称和可加关系。对于同一查询点，所有电极响应加上同一个向量不会改变响应差，监督损失和物理损失因此都作用于电极对电场。结构参数见[模型结构](model_architecture.md)。

<a id="reading-the-training-path"></a>

## 阅读训练路径

[`training/cli.py`](../../src/head2field/training/cli.py) 验证执行配置，建立数据集、完整 epoch 采样器、训练配置记录、step 函数和训练器。职责如下：

| 文件 | 职责 |
| --- | --- |
| `contracts.py` | 验证训练配置并保存完整运行配置和数据划分 |
| `dataset.py` | 验证数据文件，构建监督/物理张量 bundle |
| `cache.py` | 准备 query-major 目标和完整电场能量统计 |
| `physics_data.py` | 准备积分采样数据及仅由训练数据拟合的 FEM 校准 |
| `losses.py` | 全局、径向及高场稳定能量监督 |
| `physics.py` | 积分算子和校准的单侧超额损失 |
| `steps.py` | 共享编码、组合数据/物理损失及应用 curriculum |
| `runner.py` | AMP、反向传播、裁剪、优化器/调度器及 epoch 日志 |
| `validation.py` | 按固定顺序验证并进行最终测试 |
| `state.py` | 完整本地恢复状态及独立 RNG 流 |

数据分支使用独立的随机数生成器控制增强和 dropout。物理分支不进行坐标增强，也不使用数据分支的随机数。有效采样位置和校准值来自 FEM 数据，模型预测不会改变它们。全局、径向和高场损失的分母按被试完整电场计算，每个查询分块只计算对应误差。Huber、MAE 和相对 L2 可用于观察训练过程。

运行命令及 checkpoint 内容见[训练说明](training.md)。

<a id="editing-and-verification"></a>

## 修改与验证

修改代码时，请检查参数名与形状、输入处理、电极和查询点顺序、坐标系、单位及默认参数是否仍然一致。模型结构变化需要配套的权重加载方法和配置。软件版本与数据格式版本分别管理。

可读性重构应使用已有小模型测试，对比改动前后同一输入。源/查询索引变化应检查重排、重复 ID、共享参考和末尾不完整分块；损失变化应比较数值和梯度。开发命令与针对性验证见[贡献指南](../CONTRIBUTING_zh.md)。

公开 API 名称和 CLI 帮助使用英文，新增源码注释与 docstring 使用中文。中英文指南中的命令和参数设置应保持一致。更新随包文档后，同步 `src/head2field/resources/` 副本；已有打包测试逐字节核对这些副本。
