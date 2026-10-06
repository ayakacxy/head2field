<a id="preprocessing-profile"></a>

# 预处理流程

[English](../en/preprocessing.md) · [简体中文](preprocessing.md)

预处理将用户提供的 T1 图像依次进行 canonicalization、外部 CHARM 头模型生成和 prepared subject 组装，得到通过验证的 `prepared_subject_v1` 目录。活动环境需要安装 `preprocess` extra 和 SimNIBS 4.6.0；模板与参与者输入由用户提供。

<a id="end-to-end-command"></a>

## 端到端命令

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

## 阶段 1：canonical T1

支持两个显式配置：

- `hcp_acpc` 假定 ACPC 对齐与畸变校正已完成，不估计旋转，仅将头部居中并重采样到固定网格。
- `non_hcp_raw` 需要用户提供方向模板，精确执行一次六自由度刚体估计，拒绝缩放与剪切。

两者均输出 float32、256 x 256 x 256、1 mm、RAS T1，仿射平移为 `[-128, -128, -128]`，并生成 QC PNG 和隐去路径信息的清单。底层命令为：

```bash
python -m head2field.preprocessing canonicalize --help
```

<a id="stage-2-external-charm"></a>

## 阶段 2：外部 CHARM

默认 adapter 从用户环境导入 SimNIBS 4.6.0，调用 `simnibs.segmentation.charm_main.run`。版本不匹配或运行时缺失时直接报错，不采用替代分割。CHARM 使用外部安装中的资源；包内不复制其运行时或图谱。

头模型先在同级 staging 目录生成，仅在下列身份输入全部存在后发布：

- 位于头模型目录之外的用户 canonical T1；
- 头模型目录内 CHARM 自身的 `T1.nii.gz` 和 `final_tissues.nii.gz`；
- 根目录唯一的 `*.msh` 头网格；
- 精确的外部 SimNIBS 版本 `4.6.0`。

`charm_runtime_v1.json` 记录该次运行的 T1、组织图、唯一头网格和 SimNIBS 版本。组装与 FEM 命令通过此记录检查输入是否来自同一个头模型。

生成的头模型目录是持久保存的外部运行时输出，与紧凑的 prepared subject 目录分别管理。

<a id="stage-3-prepared-subject-assembly"></a>

## 阶段 3：prepared subject 组装

组装接收 canonical T1、`final_tissues.nii.gz` 和当前 CHARM 头模型目录。首先验证 `charm_runtime_v1.json`，要求输入 T1 与组织图匹配该次 CHARM 运行身份。默认读取 `eeg_positions/EEG10-10_UI_Jurak_2007.csv`，要求精确的固定 76 名称集合，将有限的 world-RAS 坐标重排为 Oz 优先的固定列表；用户不需提前生成名称与坐标文件。成对的 `--electrode-names-json`、`--electrode-coords-mm-npy` 选项保留为同样严格的显式 adapter override。随后执行：

1. 验证 canonical 体积的空间几何和组织数值。
2. 复制准备后的 T1 与组织图，生成 float32 标量电导率。
3. 按 C-order 保存 GM/WM 体素查询及有序 cap 文件。
4. 写入并验证 prepared 清单，检查来源文件未改变，由 staging 发布新目录。

prepared 目录仅复制准备后的 T1 和最终组织图，不复制或隐式选择网格。网格保留于外部 CHARM 头模型目录。下游 FEM 规划与执行必须提供同一个根目录唯一网格，并同时对照 `charm_runtime_v1.json` 和 prepared 清单中的 `head_model_provenance` 验证。

底层 API 为 `head2field.preprocessing.assemble_prepared_subject`。

<a id="small-public-inputs"></a>

## 小型公开输入

示例与测试可调用 `head2field.preprocessing.synthetic.write_synthetic_preparation_inputs`。该函数生成确定性的 32³ 椭球 T1、第二张方向图，以及固定 76 名称对应的合成球面坐标。这些输入不代表真实解剖，也不证明 CHARM 或 FEM 输出具有科学有效性。

<a id="validation-boundary"></a>

## 验证边界

合成测试覆盖流程选择、单次 6-DOF 行为、C-order 查询组装、来源不可变性、严格的运行时清单字段、跨清单身份绑定，以及失败/不覆盖事务。真实 SimNIBS/CHARM 执行属于外部运行时集成检查；发布验证时应记录精确安装版本。
