<a id="model-architecture"></a>

# 模型结构

[English](../en/model_architecture.md) · [简体中文](model_architecture.md)

<a id="overview"></a>

## 概览

Head2Field 先编码被试头部的解剖信息，再预测各查询位置上的电极响应。源电极与参考电极的响应相减，得到 75 个 source→Oz 先导场。训练和推理使用相同的模型结构与输入处理。

本指南介绍网络组成。源码位置和张量维度见[代码阅读指南](code_guide.md)，用途与局限见[模型说明](../MODEL_CARD_zh.md)。

<a id="input-streams"></a>

## 输入流

网络接收三类输入：

1. 十二通道头部体积，由十种组织指示通道、归一化标量电导率和归一化 T1 强度组成。
2. GM/WM 体素的查询坐标与局部特征。
3. 有序的源电极和参考电极坐标，单位为 world-RAS 毫米。

程序检查输入文件后再构建张量。体素索引表示数组中的位置，世界坐标表示物理空间中的位置，两者通过影像 affine 转换。

<a id="network-outline"></a>

## 网络结构

解剖编码器接收 128³ 输入，依次经过通道数为 16/32/64/128 的四层，使用 group normalization、SiLU 和残差瓶颈。在 16³ 瓶颈网格上，两个 96 维 mixer block 执行四头全局轴向注意力。随后，32 通道 FPN 通过横向投影和自顶向下融合合并四种尺度的特征。

对于每个查询点，程序通过三线性采样提取 FPN 特征和原始 1 mm 网格特征，再加入相对解剖质心的六频带 Fourier 坐标。两个 SwiGLU 残差 block 将其编码为 256 维查询 token。

共享电极编码器接收电极周围的七点 patch（中心及半径 6 mm 处的六个轴向邻居）、相对质心坐标和 128 维被试上下文，输出 192 维电极 token。带门控的绝对坐标分支为查询和电极 token 补充信息。

解码器使用两个带 128 维瓶颈的 FiLM 条件残差 block，预测 `G(head, query, electrode)`。输入几何信息包括归一化坐标差、物理位移和按 128 mm 缩放的距离。两个电极响应相减，即得到源到参考的电场：

```text
E(a -> b) = G(a) - G(b)
E(a -> b) = -E(b -> a)
E(a -> c) = E(a -> b) + E(b -> c)
```

共享响应库使电极对电场满足上述反对称和可加关系，数值计算允许浮点舍入。源电极和参考电极使用同一网络，不另设角色 embedding。完整设置见 `configs/head2field_v1.json`。

推理会话缓存被试特征和准备好的电极对，随后分块计算查询点，以限制内存占用。每个分块均保持相同的源顺序、查询顺序、坐标系和输出单位。

<a id="fixed-v1-factory"></a>

## 创建 v1 模型

使用 `create_head2field_v1()` 创建与内置权重匹配的网络，设置见 [`../configs/head2field_v1.json`](../../configs/head2field_v1.json)。加载权重时，程序要求：

- 非空 `str -> Tensor` 映射，无 wrapper 元数据；
- 精确包含 208 个张量和 2,607,870 个标量；
- 每个张量均为 float32；
- 参数名和张量形状完全匹配；
- 严格加载，不含缺失或额外参数。

修改模型结构、参数名、坐标约定或输入处理方式后，应为模型指定新版本，并提供匹配的权重。

<a id="stable-top-level-api"></a>

## 稳定的顶层 API

`head2field` 顶层提供以下公开接口：

- `Head2FieldV1`；
- `create_head2field_v1`；
- `Head2FieldInferenceSession`；
- `load_head2field_v1_weights`。

预处理、FEM 和训练使用各自的子模块与命令，调用时才加载对应的可选依赖。

<a id="model-versions"></a>

## 模型版本

内置模型使用 `Head2FieldV1`、`create_head2field_v1()`、`load_head2field_v1_weights()`、`configs/head2field_v1.json` 和 `head2field_v1_weights.pth`。新增模型版本时，应沿用同样命名方式，提供独立的配置和权重加载器。

模型版本、软件包版本和数据格式版本分别管理。软件包版本为 `0.1.0`；`v2` 训练统计格式仍可保存训练 Head2Field v1 所用的统计量。
