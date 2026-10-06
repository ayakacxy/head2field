<a id="prepared-subject-input-contract"></a>

# 输入文件与坐标要求

[English](../en/input_contract.md) · [简体中文](input_contract.md)

推理使用预处理后的被试数据目录。运行 `head2field-prepare` 可生成以下文件：

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

T1、组织分割图和电导率图需要使用同一个网格：

- 体积大小为 256³，分辨率为 1 mm，坐标方向为 RAS。
- affine 对角为 `[1,1,1,1]`，平移为 `[-128,-128,-128]`。
- T1 和电导率数组使用 float32。
- 组织标签为整数 0–10，其中 1 表示 WM，2 表示 GM。

电导率单位为 S/m，每个体素按组织标签从
`[0,0.126,0.275,1.654,0.010,0.465,0.500,0.008,0.025,0.600,0.160]`
中取值。输入模型时使用 float32 `sigma/1.654`。

## 查询点与电极

`gmwm_ijk.npy` 是非空整数 `[Q,3]` 数组，记录全部 GM/WM 体素的位置。
其内容和顺序必须与组织 `{1,2}` 的 C-order `np.argwhere` 结果一致。
预测数组的查询轴沿用这个顺序。

电极文件包含 76 个不重复的名称，Oz 出现一次。坐标数组形状为 `[76,3]`，
使用 world-RAS 毫米坐标，不能包含 NaN 或 Inf。移除 Oz 后，其余 75 个源电极
保持原来的排列顺序。

## 文件检查

`prepared_subject_v1.json` 记录文件相对路径、大小、空间信息、电导率表、
模型输入变换和 SimNIBS 4.6.0 头模型信息。加载数据时，程序还会检查实际数组。
FEM 运行前会进一步核对 CHARM 网格、软件版本，以及预处理目录与 CHARM
目录中的组织图是否一致。

推理期间输入文件保持只读；程序会在运行前后检查输入和权重文件的大小与修改时间。
旧版清单中的可选摘要字段不参与这些检查。

JSON 字段定义见 [schema](../../schemas/prepared_subject_v1.schema.json)，
数据生成步骤见[预处理说明](preprocessing.md)。
