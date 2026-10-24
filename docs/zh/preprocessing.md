<a id="preprocessing-profile"></a>

# 预处理流程

[English](../en/preprocessing.md) · [简体中文](preprocessing.md)

T1 预处理分为三步：对齐与重采样、生成 CHARM 头模型、整理模型输入。
请在运行命令的环境中安装 `preprocess` 可选依赖和 SimNIBS 4.6.0。
T1 影像由用户准备；处理非 HCP 原始数据时，还需要方向模板。

<a id="end-to-end-command"></a>

## 完整预处理命令

```bash
head2field-prepare \
  --input-t1 raw_t1.nii.gz \
  --profile non_hcp_raw \
  --template orientation_template.nii.gz \
  --canonical-output-dir canonical_stage \
  --head-model-output-dir charm_head_model \
  --prepared-output-dir prepared_subject
```

三个输出目录必须尚不存在、互不相同，且不能相互嵌套。命令不提供覆盖选项。
某个阶段失败时，已完成阶段的结果会保留，用户原有的结果不会被删除或替换。

Python 入口为 `head2field.preprocessing.prepare_subject`。

<a id="stage-1-canonical-t1"></a>

## 阶段 1：T1 对齐与重采样

根据输入影像选择处理方式：

- `hcp_acpc`：用于已完成 ACPC 对齐和畸变校正的数据，只将头部居中并重采样，不估计旋转。
- `non_hcp_raw`：用于原始 T1，需要提供方向模板，进行一次六自由度刚体配准；不接受缩放或剪切。

两种方式都输出 float32、256 x 256 x 256、1 mm、RAS T1，仿射平移为
`[-128, -128, -128]`，并生成 QC PNG 和隐去路径信息的清单。
单独使用对齐与重采样功能时，可先查看：

```bash
python -m head2field.preprocessing canonicalize --help
```

<a id="stage-2-external-charm"></a>

## 阶段 2：外部 CHARM

Head2Field 调用已安装 SimNIBS 4.6.0 中的 `simnibs.segmentation.charm_main.run`。
未安装 SimNIBS 或版本不符时，程序会停止。CHARM 使用自身安装目录中的资源；
Head2Field 不附带 CHARM 运行环境或图谱，也不以其他分割方法替代。

CHARM 先写入目标目录旁的临时目录。结果保存到目标位置前，Head2Field 会检查：

- 从头模型目录外传入的预处理 T1；
- CHARM 目录中的 `T1.nii.gz` 和 `final_tissues.nii.gz`；
- 目录根层唯一的 `*.msh` 头网格；
- 已安装的 SimNIBS 版本为 `4.6.0`。

`charm_runtime_v1.json` 记录这些影像、头网格和软件版本。
后续整理输入和运行 FEM 时，程序用该记录确认文件来自同一个头模型。

请在头模型目录中保留完整 CHARM 输出。下一步会另建一个较小的被试目录，存放模型输入。

<a id="stage-3-prepared-subject-assembly"></a>

## 阶段 3：整理模型输入文件

这一步读取预处理 T1、`final_tissues.nii.gz` 和 CHARM 头模型目录，
先根据 `charm_runtime_v1.json` 检查文件是否对应。

电极位置默认读取自 `eeg_positions/EEG10-10_UI_Jurak_2007.csv`。
文件必须包含标准 76 个电极名称和有限的 world-RAS 坐标。
输出采用固定顺序，Oz 位于第一行。自行提供电极文件时，必须同时传入
`--electrode-names-json` 和 `--electrode-coords-mm-npy`，并满足相同要求。

随后，程序依次完成：

1. 检查影像网格、affine 和组织标签。
2. 复制预处理 T1 与组织图，并生成 float32 标量电导率。
3. 按 C-order 保存 GM/WM 体素查询点，以及排序后的电极文件。
4. 写入并检查输入清单，确认来源文件未变，将完整被试目录保存到目标位置。

被试目录包含预处理 T1 和组织图的副本，不包含头网格。
规划或运行 FEM 时，请传入同一个 CHARM 目录中的唯一网格。
程序会将其与 `charm_runtime_v1.json` 和输入清单中的 `head_model_provenance` 记录核对。

底层 Python 函数为 `head2field.preprocessing.assemble_prepared_subject`。

<a id="small-public-inputs"></a>

## 合成示例

示例和软件测试可以调用
`head2field.preprocessing.synthetic.write_synthetic_preparation_inputs`，
生成固定的 32³ 椭球 T1、方向图和标准 76 个电极名称对应的合成球面坐标。
这些非解剖数据用于检查软件运行，不用于测量头模型或 FEM 精度。

<a id="validation-boundary"></a>

## 测试说明

合成测试覆盖处理方式选择、刚体配准调用、查询点顺序、输入文件保护、清单与文件的一致性，
也检查阶段失败时能否保留已有结果，以及程序是否拒绝覆盖输出。
评估 CHARM 分割时，应另用真实被试运行，并记录 SimNIBS 版本。
