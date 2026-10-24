# 模型权重

[English](README.md) · [简体中文](README_zh.md)

预训练 Head2Field v1 权重随包提供，位于
`src/head2field/resources/weights/head2field_v1_weights.pth`，预测时默认加载此文件。

| 字段 | 值 |
| --- | --- |
| 格式 | 参数名到张量的字典（state dictionary） |
| 张量 | 208 个 float32 张量 |
| 标量数 | 2,607,870 |
| 文件大小 | 10,497,106 字节 |
| 模型创建函数 | `create_head2field_v1()` |

加载器在严格加载 state dictionary 前，检查参数名、张量形状、float32 类型、
数值是否有限，以及元素总数。文件只保存模型张量，
不含优化器、训练控制信息、参与者标识、路径或随机数状态。

```python
from head2field import load_head2field_v1_weights

model = load_head2field_v1_weights(device="cpu")
```

使用兼容的自定义权重时，向 `head2field-predict` 传入 `--weights custom_v1.pth`，
或将路径交给加载器。训练导出的 `best_model.pt` 是仅含模型参数的 state dictionary；
完整的 `train_state_step_*.pt` 文件用于本地恢复。每次新训练得到的模型都需要单独评估。

保存或分享权重时，请同时附上模型版本、结构、校验值、许可证和
[数据使用声明](../docs/DATA_USE_zh.md)，软件版本单独记录。
更改训练配置会影响后续训练，不会修改已有权重文件。
预测 API 见[推理说明](../docs/zh/inference.md)，比较与计时方法见
[评估说明](../docs/zh/paper_reproduction.md)。
