# FEM 标签生成

[English](../en/fem_label_generation.md) · [简体中文](fem_label_generation.md)

使用外部 SimNIBS 4.6.0 和 prepared head。所有目标目录必须全新。求解器使用标准 76 电极、
Oz 参考、直径 10 mm/厚度 4 mm 的圆柱电极和完整各向同性组织电导率表，生成 75 个源场。

```bash
head2field-fem plan --prepared-subject-dir prepared_subject --head-mesh head_model/subject.msh

head2field-fem generate   --prepared-subject-dir prepared_subject   --head-mesh head_model/subject.msh   --leadfield-output-dir leadfield_run   --labels-output-dir fem_labels

head2field-fem validate   --prepared-subject-dir prepared_subject   --fem-labels-dir fem_labels
```

`export` 可以从已有且验证过的求解器运行插值；HDF5、日志、运行清单和输出参数见
`head2field-fem export --help`。线性插值使用 GM/WM 查询；SimNIBS 的 1 A 场乘以 0.001，
转换为源 +1 mA/Oz −1 mA 的 float32 world-RAS 向量，单位 V/m。

运行保留日志、HDF5 和 runtime MAT，并将设置的电导率与 MAT 回读值直接比较。标签记录源
顺序、相对路径、大小、`[Q,3]`、float32 和有限值。新导出保存 `query_ijk.npy`，与 prepared
subject 的有序坐标逐项比较。

验证核对电流、参考、场类型、单位、组织、源顺序、坐标、电导率、数值数组与安全路径，采用
统一验证模式。旧清单可选摘要字段不参与通过条件。输出由同级 staging 原子发布，不覆盖
已有结果。见[数据合同](data_contracts.md)和 [FEM schema](../../schemas/fem_labels_v1.schema.json)。
