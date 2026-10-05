# 数据合同

[English](../en/data_contracts.md) · [简体中文](data_contracts.md)

JSON schema 描述字段结构，运行时验证器直接核对数组、空间坐标、电导率、电极顺序、有限值和单位。

| 合同 | 使用模块 |
| --- | --- |
| `prepared_subject_v1` | 预处理、推理、FEM 和训练 |
| `fem_labels_v1` | 完整 75 源 FEM 导出与训练 |
| `training_loss_statistics_v2` | 50/100 mm 分区能量目标 |
| `training_dataset_v1` | 按组划分训练与本地恢复 |

见 [schema](../../schemas/) 和[输入合同](input_contract.md)。输入和训练资产采用相对路径。
加载器检查安全路径、文件存在及大小、shape 和科学参数。查询身份是有序 `gmwm_ijk` 数组，
源身份是移除 Oz 后的有序电极列表。新 FEM 导出保存 `query_ijk.npy` 并逐项比较；能量统计
NPZ 保存查询、affine 和电极坐标，并与当前输入直接比较。

电导率检查完整数值和单位；FEM 还检查电流、向量分量、HDF5 元数据和求解器电导率回读。
训练恢复比较保存的完整配置、分组和资产路径。旧清单的可选摘要字段不参与验证。
