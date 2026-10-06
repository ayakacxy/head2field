<a id="preprocessing-profile"></a>

# 预处理流程

[English](../en/preprocessing.md) · [简体中文](preprocessing.md)

预处理依次完成 T1 对齐与重采样、CHARM 头模型生成和输入文件整理，输出可供推理和训练使用的被试数据目录。请在运行命令的 Python 环境中安装 `preprocess` 可选依赖和 SimNIBS 4.6.0；T1 影像和方向模板由用户提供。

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

三个输出路径必须是尚不存在、互不相同且不互相嵌套的目录，不提供覆盖选项。后续阶段失败时保留已完成阶段的结果，不删除或重写用户已有输出。

同一流程也可通过 `head2field.preprocessing.prepare_subject` 调用。

<a id="stage-1-canonical-t1"></a>

## 阶段 1：T1 对齐与重采样

可选择以下两种处理方式：

- `hcp_acpc` 假定 ACPC 对齐与畸变校正已完成，不估计旋转，仅将头部居中并重采样到固定网格。
- `non_hcp_raw` 需要用户提供方向模板，进行一次六自由度刚体配准，不进行缩放或剪切。

两者均输出 float32、256 x 256 x 256、1 mm、RAS T1，仿射平移为 `[-128, -128, -128]`，并生成 QC PNG 和隐去路径信息的清单。底层命令为：

```bash
python -m head2field.preprocessing canonicalize --help
```

<a id="stage-2-external-charm"></a>

## 阶段 2：外部 CHARM

默认 adapter 从用户环境导入 SimNIBS 4.6.0，调用 `simnibs.segmentation.charm_main.run`。版本不匹配或运行时缺失时直接报错，不采用替代分割。CHARM 使用外部安装中的资源；包内不复制其运行时或图谱。

CHARM 先在同级临时目录中生成头模型。完成后，程序检查以下文件和版本信息，再保存到目标目录：

- 位于头模型目录之外的预处理 T1；
- 头模型目录内 CHARM 自身的 `T1.nii.gz` 和 `final_tissues.nii.gz`；
- 根目录唯一的 `*.msh` 头网格；
- SimNIBS 版本为 `4.6.0`。

`charm_runtime_v1.json` 记录该次运行的 T1、组织图、唯一头网格和 SimNIBS 版本。组装与 FEM 命令通过此记录检查输入是否来自同一个头模型。

头模型目录保留 CHARM 的完整输出；推理使用的预处理数据保存在另一个目录。

<a id="stage-3-prepared-subject-assembly"></a>

## 阶段 3：整理模型输入文件

这一步使用预处理 T1、`final_tissues.nii.gz` 和 CHARM 头模型目录。程序通过 `charm_runtime_v1.json` 检查这些文件是否来自同一次头模型生成。

电极位置默认从 `eeg_positions/EEG10-10_UI_Jurak_2007.csv` 读取。文件必须包含标准的 76 个电极名称；坐标不能含 NaN 或 Inf，输出按固定顺序排列，Oz 位于第一行。通常无需另外准备电极文件。需要自定义输入时，必须同时提供 `--electrode-names-json` 和 `--electrode-coords-mm-npy`，并满足相同的名称与坐标要求。随后执行：

1. 检查影像网格、affine 和组织标签。
2. 复制准备后的 T1 与组织图，生成 float32 标量电导率。
3. 按 C-order 保存 GM/WM 查询点，并保存电极名称和坐标。
4. 写入并验证 prepared 清单，检查来源文件未改变，将临时目录保存为目标目录。

被试数据目录只复制预处理 T1 和组织图，网格仍保存在 CHARM 头模型目录中。运行 FEM 时，请传入该头模型目录下唯一的网格文件；程序会将它与 `charm_runtime_v1.json` 和输入清单中的 `head_model_provenance` 记录进行比较。

底层 API 为 `head2field.preprocessing.assemble_prepared_subject`。

<a id="small-public-inputs"></a>

## 合成示例

示例与测试可调用 `head2field.preprocessing.synthetic.write_synthetic_preparation_inputs`。该函数生成固定的 32³ 椭球 T1、第二张方向图，以及固定 76 名称对应的合成球面坐标。这些数据只用于软件测试，不能用来评估真实头模型或 FEM 的精度。

<a id="validation-boundary"></a>

## 测试说明

合成测试检查处理方式选择、刚体配准调用、查询点顺序、输入文件是否被改动，以及清单内容是否与文件一致。测试还会确认程序失败时保留已有结果，并且不覆盖输出。实际 CHARM 分割需要另外运行，并记录使用的 SimNIBS 版本。
