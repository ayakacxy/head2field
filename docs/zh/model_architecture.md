<a id="model-architecture"></a>

# 模型结构

[English](../en/model_architecture.md) · [简体中文](model_architecture.md)

<a id="overview"></a>

## 概览

Head2Field v1 结合解剖编码、查询特征和电极几何，预测 75 个 source→Oz 先导场。训练与推理使用同一套 v1 结构，加载权重时检查参数名和形状。以下介绍网络的组成；用途与局限见[模型说明](../MODEL_CARD_zh.md)，源码和张量维度见[代码阅读指南](code_guide.md)。

<a id="input-streams"></a>

## 输入流

模型结合三类信息：

1. 十二通道头部体积，由十种组织指示通道、归一化标量电导率和归一化 T1 强度组成。
2. GM/WM 体素的查询坐标与局部特征。
3. 有序的源电极和参考电极坐标，单位为 world-RAS 毫米。

构建张量前，程序先检查输入文件。体素索引和世界坐标使用不同的坐标表示，转换时根据 affine 计算。

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

这些恒等关系来自共享响应库，数值计算存在浮点舍入。模型不设置独立的源/参考响应网络或角色 embedding。精确设置记录在 `configs/head2field_v1.json`。

推理 session 缓存被试特征及电极对准备结果，使查询按有限大小分块执行。分块改变内存占用，应保留源顺序、查询顺序、坐标系和输出单位。

<a id="fixed-v1-factory"></a>

## 创建 v1 模型

`create_head2field_v1()` 创建与内置权重匹配的 v1 模型，结构参数见 [`../configs/head2field_v1.json`](../../configs/head2field_v1.json)。权重加载器还检查：

- 非空 `str -> Tensor` 映射，无 wrapper 元数据；
- 精确包含 208 个张量和 2,607,870 个标量；
- 每个张量均为 float32；
- 参数名和张量形状完全匹配；
- 严格加载，不含缺失或额外参数。

修改模型结构、参数名、坐标约定或输入处理方式时，需要使用新的模型版本，并准备与其匹配的权重。

<a id="stable-top-level-api"></a>

## 稳定的顶层 API

`head2field` 命名空间仅导出：

- `Head2FieldV1`；
- `create_head2field_v1`；
- `Head2FieldInferenceSession`；
- `load_head2field_v1_weights`。

预处理、FEM 和训练通过各自的子模块或命令调用，只有使用这些功能时才加载相关可选依赖。

<a id="model-versions"></a>

## 模型版本

当前模型的名称包括 `Head2FieldV1`、`create_head2field_v1()`、`load_head2field_v1_weights()`、`configs/head2field_v1.json` 和 `head2field_v1_weights.pth`。未来版本沿用同样命名规则，拥有各自的结构配置和权重加载器。

软件包版本（`0.1.0`）和数据 schema 版本独立于模型版本。例如，`v2` 训练统计 schema 仍可描述用于训练 Head2Field v1 的数据。
