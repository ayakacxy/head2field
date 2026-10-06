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
| 模型工厂 | `create_head2field_v1()` |

加载器直接核对参数键、形状、FP32 dtype、有限值和总数，并严格加载。文件仅含模型张量，
不含优化器、训练控制、被试标识、路径或随机数状态。

```python
from head2field import load_head2field_v1_weights

model = load_head2field_v1_weights(device="cpu")
```

自定义兼容权重可通过 `head2field-predict --weights custom_v1.pth` 或加载器参数指定。
训练输出的 `best_model.pt` 是裸模型权重；`train_state_step_*.pt` 仅用于本地恢复。
新训练产物需要独立评价。

软件版本、模型结构和权重校验值分别记录。修改训练配置影响新的训练运行，已有预训练张量仍保留各自的身份。分发权重时保留模型版本、结构、许可证和[数据使用说明](../docs/DATA_USE_zh.md)。完整 API 见[推理说明](../docs/zh/inference.md)，比较与计时方法见[评估说明](../docs/zh/paper_reproduction.md)。
