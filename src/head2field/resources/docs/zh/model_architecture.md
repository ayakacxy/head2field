<a id="model-architecture"></a>

# 模型结构

[English](../en/model_architecture.md) · [简体中文](model_architecture.md)

<a id="overview"></a>

## 概览

Head2Field v1 结合解剖编码、查询特征和电极几何，预测 75 个 source→Oz 先导场。固定工厂及严格权重 schema 保证训练与推理使用一致结构。模型合同如下；用途与局限见[模型说明](../../MODEL_CARD_zh.md)。[代码阅读指南](code_guide.md)给出各阶段的实现位置和网络张量维度。

<a id="input-streams"></a>

## 输入流

模型结合三类信息：

1. 十二通道头部体积，由十种组织指示通道、归一化标量电导率和归一化 T1 强度组成。
2. GM/WM 体素的查询坐标与局部特征。
3. 有序的源电极和参考电极坐标，单位为 world-RAS 毫米。

构建张量前由 prepared subject 加载器验证来源数据。坐标转换显式执行，不将数组索引视为世界坐标。

<a id="network-outline"></a>

## 网络结构

128³ 编码器含四层，通道数为 16/32/64/128，采用 group normalization、SiLU 和残差瓶颈。两个 96 维 mixer block 在 16³ 瓶颈网格上使用四头全局轴向注意力。32 通道 FPN 通过横向投影和自顶向下融合组合四种尺度。

查询 token 结合三线性采样的 FPN 特征、原始 1 mm 输入特征，以及相对质心的六频带 Fourier 坐标。两个 SwiGLU 残差 block 生成 256 维 token。共享电极编码器将七点 patch（中心及六个轴向邻居、半径 6 mm）、相对质心坐标和 128 维被试上下文组合为 192 维电极 token。带门控的绝对坐标分支补充两类 token。

两个 FiLM 条件残差 block 使用 128 维瓶颈，解码共享电极响应 `G(head, query, electrode)`。相对几何包括归一化坐标差、物理位移及按 128 mm 缩放的距离。源到参考电场是响应差：

```text
E(a -> b) = G(a) - G(b)
E(a -> b) = -E(b -> a)
E(a -> c) = E(a -> b) + E(b -> c)
```

这些恒等关系来自共享响应库，数值计算存在浮点舍入。模型不设置独立的源/参考响应网络或角色 embedding。精确设置记录在 `configs/head2field_v1.json`，结构对应论文第 2.3 节。

推理 session 缓存被试特征及电极对准备结果，使查询按有限大小分块执行。分块改变内存占用，应保留源顺序、查询顺序、坐标系和输出单位。

<a id="fixed-v1-factory"></a>

## 固定 v1 工厂

`create_head2field_v1()` 是唯一官方 v1 工厂，精确构造参数同步在 [`../configs/head2field_v1.json`](../../configs/head2field_v1.json)。权重加载器还检查：

- 非空 `str -> Tensor` 映射，无 wrapper 元数据；
- 精确包含 208 个张量和 2,607,870 个标量；
- 每个张量均为 float32；
- 参数名和张量形状完全匹配；
- 严格加载，不含缺失或额外参数。

修改工厂字段、参数名、坐标约定或模型输入变换会产生不同模型合同，需要单独审查相应版本。

<a id="stable-top-level-api"></a>

## 稳定的顶层 API

`head2field` 命名空间仅导出：

- `Head2FieldV1`；
- `create_head2field_v1`；
- `Head2FieldInferenceSession`；
- `load_head2field_v1_weights`。

预处理、FEM 与训练通过显式子模块和 CLI 流程访问，不为顶层 API 增加重型可选依赖导入。

<a id="model-versions"></a>

## 模型版本

模型版本采用显式命名：当前模型由 `Head2FieldV1`、`create_head2field_v1()`、`load_head2field_v1_weights()`、`configs/head2field_v1.json` 和 `head2field_v1_weights.pth` 描述。未来版本沿用同样命名规则，拥有各自的结构配置和权重加载器。

软件包版本（`0.1.0rc1`）和数据 schema 版本独立于模型版本。例如，`v2` 训练统计 schema 仍可描述用于训练 Head2Field v1 的数据。
