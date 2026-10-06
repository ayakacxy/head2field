# 模型权重

[English](README.md) · [简体中文](README_zh.md)

预训练 Head2Field v1 权重随包提供，位于
`src/head2field/resources/weights/head2field_v1_weights.pth`，默认自动加载。

| 字段 | 值 |
| --- | --- |
| 格式 | 裸 string-to-tensor state dictionary |
| 张量 | 208 个 float32 张量 |
| 标量数 | 2,607,870 |
| 文件大小 | 10,497,106 字节 |
| 模型创建函数 | `create_head2field_v1()` |

加载时会检查参数名、张量形状、float32 类型和元素数量，并检查是否包含 NaN 或 Inf。权重文件只保存模型参数，不含优化器、训练进度、被试标识、文件路径或随机数状态。

```python
from head2field import load_head2field_v1_weights

model = load_head2field_v1_weights(device="cpu")
```

自定义兼容权重可通过 `head2field-predict --weights custom_v1.pth` 或加载器参数指定。
训练输出的 `best_model.pt` 是裸模型权重；`train_state_step_*.pt` 仅用于本地恢复。
自行训练的模型需要单独评估。

保存或分享权重时，请同时记录模型版本、结构、校验值和许可证，并附上[数据使用说明](../docs/DATA_USE_zh.md)。更改训练配置会影响后续训练，不会修改已有权重文件。加载方法见[推理说明](../docs/zh/inference.md)，比较与计时方法见[评估说明](../docs/zh/paper_reproduction.md)。
