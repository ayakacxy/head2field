<a id="prepared-subject-input-contract"></a>

# 输入文件与坐标要求

[English](../en/input_contract.md) · [简体中文](input_contract.md)

Head2Field 从 `head2field-prepare` 生成的被试目录中读取数据。
请将输入清单及其引用的文件保存在一起：

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

## 影像与电导率

T1、组织分割图和电导率图必须使用相同的网格，并满足以下数据类型要求：

- 体积大小为 256³，分辨率为 1 mm，坐标方向为 RAS。
- affine 对角为 `[1,1,1,1]`，平移为 `[-128,-128,-128]`。
- T1 和电导率数组使用 float32。
- 组织标签为整数 0–10，其中 1 表示 WM，2 表示 GM。

电导率单位为 S/m。每个组织标签对应表中的同序号数值：
`[0,0.126,0.275,1.654,0.010,0.465,0.500,0.008,0.025,0.600,0.160]`。
模型的电导率输入为 float32 `sigma/1.654`。

## 查询点与电极

`gmwm_ijk.npy` 是非空整数 `[Q,3]` 数组，包含全部 GM/WM 体素的位置。
数组内容及行顺序必须与 `np.argwhere(np.isin(tissues, (1, 2)))` 的 C-order 结果相同。
预测结果的查询轴沿用这些行的顺序。

电极文件包含 76 个不重复的名称，其中 Oz 恰好出现一次。
坐标为 world-RAS 毫米值，数组形状为 `[76,3]`，所有数值都必须有限。
移除 Oz 后，得到的 75 个源电极保持原有相对顺序。

## 文件检查

`prepared_subject_v1.json` 记录文件相对路径、大小、空间信息、电导率表、
模型输入变换和 SimNIBS 4.6.0 头模型信息。加载时，程序既检查清单，也检查实际数组。
FEM 预检还会核对 CHARM 网格和软件版本，并比较被试目录与 CHARM 目录中的组织图。

推理只读取输入文件，不修改输入；运行前后会检查输入和权重文件的大小与修改时间。
旧清单中的可选摘要字段不用于这些检查。

各 JSON 字段见 [schema](../../schemas/prepared_subject_v1.schema.json)，
数据准备步骤见[预处理说明](preprocessing.md)。
