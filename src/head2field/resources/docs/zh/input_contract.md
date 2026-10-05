# Prepared subject 输入合同

[English](../en/input_contract.md) · [简体中文](input_contract.md)

推理读取已经准备的目录，可通过 `head2field-prepare` 生成。

```text
prepared-subject/
├── prepared_subject_v1.json
├── images/
│   ├── T1_1mm.nii.gz
│   ├── final_tissues.nii.gz
│   └── sigma_1mm.nii.gz
├── voxels/gmwm_ijk.npy
└── electrodes/
    ├── electrode_names_all.json
    ├── electrode_names_no_oz.json
    └── electrode_coords_mm.npy
```

三份体积均为 256³、1 mm、RAS，affine 对角为 `[1,1,1,1]`，平移为 `[-128,-128,-128]`。
T1 和 sigma 为 float32；组织为精确整数 0–10，1=WM、2=GM。电导率（S/m）逐体素等于
`[0,0.126,0.275,1.654,0.010,0.465,0.500,0.008,0.025,0.600,0.160]` 查表结果，模型使用
float32 `sigma/1.654`。

`gmwm_ijk.npy` 为非空整数 `[Q,3]` 数组，必须精确等于组织 `{1,2}` 的 C-order `np.argwhere`。
76 个电极名称唯一且包含一个 Oz，坐标为有限的 world-RAS 毫米 `[76,3]` 数组；移除 Oz 后的
75 个源保留原顺序。

清单记录必需相对路径、文件大小、空间参数、完整电导率表、输入变换和 SimNIBS 4.6.0
头模型来源。加载器核对实际数组与空间信息；FEM 预检还核对唯一 CHARM 网格、运行时版本及
prepared/CHARM 组织内容。推理只读输入，运行前后检查输入和权重文件的大小及修改时间。

见 [schema](../../schemas/prepared_subject_v1.schema.json) 和[预处理](preprocessing.md)。
旧清单中的可选摘要字段只作为未使用的来源记录读取。
