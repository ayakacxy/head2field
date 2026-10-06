<a id="data-contracts"></a>

# 数据格式与检查

[English](../en/data_contracts.md) · [简体中文](data_contracts.md)

JSON schema 定义各清单需要哪些字段。加载文件时，程序还会检查数组形状、
坐标、电导率、电极顺序和单位，并检查数值是否包含 NaN 或 Inf。

| 文件格式 | 用途 |
| --- | --- |
| `prepared_subject_v1` | 预处理结果，供推理、FEM 和训练使用 |
| `fem_labels_v1` | 75 个源电极的 FEM 标签，供训练使用 |
| `training_loss_statistics_v2` | 50/100 mm 分区的能量统计 |
| `training_dataset_v1` | 训练数据路径及训练、验证、测试分组 |

字段定义见 [schema](../../schemas/)，输入目录结构见[输入格式说明](input_contract.md)。
文件路径相对于对应清单，程序会检查路径是否安全、文件是否存在、大小是否一致。

预测、FEM 标签和训练统计必须使用相同的查询点与源电极顺序：

- 查询点顺序以 `gmwm_ijk` 为准；源电极顺序是移除 Oz 后的电极列表。
- FEM 导出保存 `query_ijk.npy`，并与预处理查询点逐项比较。
- 能量统计 NPZ 保存查询点、affine 和电极坐标，加载时与输入数据比较。

电导率表的全部数值和单位也会检查。FEM 验证还包括电流、向量分量、HDF5
元数据，以及求解器实际使用的电导率。恢复训练时，配置、分组和数据文件路径
必须与已保存记录一致。旧清单中的可选摘要字段不参与检查。
