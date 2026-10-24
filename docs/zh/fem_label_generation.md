# FEM 标签生成

[English](../en/fem_label_generation.md) · [简体中文](fem_label_generation.md)

使用 SimNIBS 4.6.0，根据预处理后的被试数据和对应的 CHARM 头网格生成训练标签。
流程采用标准 76 电极布局、Oz 参考、直径 10 mm、厚度 4 mm 的圆形电极，
以及完整的各向同性组织电导率表，计算 75 个源电极到 Oz 的电场。输出目录必须尚不存在。

先用 `plan` 查看模拟设置，不写入文件；再用 `generate` 求解并导出标签，
最后用 `validate` 检查结果：

```bash
head2field-fem plan --prepared-subject-dir prepared_subject --head-mesh head_model/subject.msh

head2field-fem generate   --prepared-subject-dir prepared_subject   --head-mesh head_model/subject.msh   --leadfield-output-dir leadfield_run   --labels-output-dir fem_labels

head2field-fem validate   --prepared-subject-dir prepared_subject   --fem-labels-dir fem_labels
```

已有通过检查的求解结果时，可用 `export` 将电场线性插值到 GM/WM 查询点。
所需的 HDF5、求解器日志、运行清单和输出参数见 `head2field-fem export --help`。
SimNIBS 的 1 A 电场乘以 0.001 后，对应源电极 `+1 mA`、Oz `-1 mA`；
标签保存 float32 world-RAS 向量，单位为 `V/m`。

请保留求解器日志、HDF5 输出和 runtime MAT 文件。
程序会比较配置中的电导率与 MAT 文件回读值。标签清单记录源电极顺序、文件相对路径、
大小及 `[Q,3]` 数组形状。数组必须为 float32，所有数值都必须有限。
导出时还会保存 `query_ijk.npy`，并将其有序坐标与预处理被试逐项比较。

验证覆盖电流、Oz 参考、场类型、单位、组织、源电极顺序、坐标、电导率、数组和文件路径。
所有运行使用同一套检查，旧清单中的可选摘要不作为通过条件。
完整结果先写入目标旁的临时目录，再保存到新目录；已有结果不会被覆盖。

保存字段的详细定义见[数据格式与检查](data_contracts.md)和
[FEM schema](../../schemas/fem_labels_v1.schema.json)。
