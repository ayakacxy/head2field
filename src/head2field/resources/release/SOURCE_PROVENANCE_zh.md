<a id="source-provenance"></a>

# 实现来源

[English](SOURCE_PROVENANCE.md) · [简体中文](SOURCE_PROVENANCE_zh.md)

Head2Field 从父级研究项目中提取并改编模型、预处理、FEM 标签和训练组件。下表记录相对于仓库的来源路径及主要适配边界。父项目路径保留原名称以便追溯；公开软件包、Python API、命令和模型标识采用 Head2Field 命名。

默认训练配置、统计生成与读取、合成批次使用 50/100 mm 径向分区和每步监督缩放 `M / (75 * N_train)`。统计和运行合同将这些设置绑定到每次训练。推理使用固定的 v1 网络及仅含张量的权重 schema。方法对应、实验范围、权重及性能来源见[复现与评估](zh/paper_reproduction.md)。

源码与 Python 分发包包含项目代码、文档和已验证的仅含张量的预训练 v1 权重，不内嵌外部运行时、原生二进制文件、模板、图谱或研究数据集。依赖声明见[第三方声明](THIRD_PARTY_NOTICES_zh.md)。

<a id="files-with-preserved-executable-logic"></a>

## 保留可执行逻辑的文件

| 项目文件 | 父项目来源 | 范围 |
| --- | --- | --- |
| `src/head2field/model.py` | `head2field/models/head2field.py` | 保留网络数值逻辑；采用 Head2Field v1 标识及英文注释/docstring |
| `src/head2field/fem_labels/transaction.py` | `head2field/fem_labels/transaction.py` | 目录原子发布辅助函数 |

这些文件保留所列父项目来源中的数值与事务逻辑。注释和 docstring 使用英文，网络类和合同标识采用 Head2Field 命名，张量名称和形状保留 v1 权重格式。

<a id="adapted-inference-core"></a>

## 适配的推理核心

| 项目文件 | 父项目来源 | 适配边界 |
| --- | --- | --- |
| `src/head2field/_spec.py` | `head2field/config/spec.py` | 改为包内模型导入 |
| `src/head2field/inputs.py` | `head2field/data/model_inputs.py` | 改为包内模型与预处理导入 |
| `src/head2field/inference.py` | `head2field/evaluation/inference.py` | 改为包内输入与模型导入 |
| `src/head2field/weights.py` | `head2field/weights.py` | 改为包内工厂与模型导入 |
| `src/head2field/cli.py` | `head2field/cli/predict_lead_field.py` | 移除特定环境的项目根目录发现；保留输入只读及新输出目录保护 |

运行时直接核对参数、分组、相对路径、文件大小、坐标顺序、shape、dtype 和有限值。训练恢复比较完整配置与数据清单；统计必须按当前 50/100 mm 设置和训练划分生成。

<a id="adapted-preprocessing-implementation"></a>

## 适配的预处理实现

| 项目文件 | 父项目来源 | 适配边界 |
| --- | --- | --- |
| `src/head2field/preprocessing/alignment.py` | `scripts/preprocessing/align_t1_rigid.py` | 参数化 HCP 绕过流程与非 HCP 6-DOF 配准器 |
| `src/head2field/preprocessing/pipeline.py` | `head2field/preprocessing/pipeline_v1.py` | 包内 adapter、显式执行配置和事务输出 |
| `src/head2field/preprocessing/conductivity.py` | `head2field/preprocessing/conductivity.py` | 延迟导入 NIfTI 依赖以支持最小推理安装 |
| `src/head2field/preprocessing/validation.py` | `head2field/preprocessing/prepared_subject_v1.py` | 包布局及共享电导率合同 |
| `src/head2field/preprocessing/transaction.py` | `head2field/fem_labels/transaction.py` | 将事务模式适配到 prepared subject 输出 |
| `src/head2field/preprocessing/charm_adapter.py` | `scripts/preprocessing/build_head_model.py` | 通过 adapter 隔离外部 SimNIBS 调用 |

<a id="adapted-fem-label-implementation"></a>

## 适配的 FEM 标签实现

| 项目文件 | 父项目来源 | 适配边界 |
| --- | --- | --- |
| `src/head2field/fem_labels/contracts.py` | `head2field/fem_labels/contracts.py` | 公开验证合同及包内导入 |
| `src/head2field/fem_labels/simnibs_adapter.py` | `head2field/fem_labels/simnibs_adapter.py` | 外部安装边界与显式报错 |
| `src/head2field/fem_labels/leadfield_run.py` | `scripts/simulation/run_oz_reference_leadfield.py` | 参数化 source→Oz 流程及来源记录 |
| `src/head2field/fem_labels/audit.py` | 项目的 `leadfield_run.py` 和 `generation.py` 验证代码 | 独立的 schema、路径安全及可选完整哈希审计 |

<a id="adapted-training-implementation"></a>

## 适配的训练实现

| 项目文件 | 父项目来源 | 适配边界 |
| --- | --- | --- |
| `src/head2field/training/contracts.py` | `head2field/training/contracts.py` | 独立公开数据集与恢复合同 |
| `src/head2field/training/losses.py` | `head2field/training/losses.py` | 包内导入及固定的公开目标函数 |
| `src/head2field/training/physics.py` | `head2field/physics/losses.py` | 公开 N128 物理损失实现 |
| `src/head2field/training/state.py` | `head2field/training/runner.py` | 提取 RNG 和 checkpoint 语义；包含 AMP、最优模型与早停状态 |
| `src/head2field/training/runner.py` | `head2field/training/runner.py` | CUDA FP16 训练器、epoch 调度、早停、最优模型测试及本地恢复 |
| `src/head2field/training/dataset.py` | `head2field/data/head2field.py` | prepared subject 与 FEM 标签读取合同；完整 epoch 遍历 |
| `src/head2field/training/steps.py` | 父项目训练流程 | 基于包内模块的公开流程组织 |
| `src/head2field/training/cache.py` | `scripts/training/prepare_training_cache.py` 及预测器损失统计脚本 | 参数化缓存和统计阶段 |
| `src/head2field/training/physics_data.py` | `head2field/data/physics_primitives.py` | 公开物理原语与统计合同 |
| `src/head2field/training/validation.py` | 父项目验证方法 | 完整的确定性验证，a(t)=1 |
| `src/head2field/training/cli.py` | 父项目训练与预检 CLI | 重新实现公开命令及严格的参数分离 |

<a id="project-authored-implementation"></a>

## 为本项目编写的实现

以下源码文件为本软件包编写：

- 包入口与辅助模块：`src/head2field/__init__.py`、`geometry.py` 和 `resource_paths.py`；
- 预处理入口与组合：`preprocessing/{__init__,__main__,assembly,cli,electrodes,synthetic}.py`；
- FEM 标签入口与组合：`fem_labels/{__init__,__main__,cli,generation}.py`；
- 训练入口与合成支持：`training/{__init__,__main__,synthetic}.py`。

`tests/` 与 `examples/` 中的文件、全部包元数据、顶层配置和 schema、包资源副本及发布文档均为本软件包编写。测试与示例仅使用虚构标识和生成数组。

<a id="explicitly-excluded"></a>

## 明确排除的内容

允许列表排除研究数据集、完整 checkpoint 与恢复状态、FEM 输出、训练缓存、分析结果、可选性能后端、刺激规划代码、第三方二进制文件、模板、图谱和归档实验材料。允许列表中的预训练 v1 权重为 `src/head2field/resources/weights/` 内的裸 `str -> Tensor` state dictionary，随 wheel 和源码分发包提供。

实现及发布资产清单发生变化时，应同步更新本文。模型权重保留版本和校验记录，参见[权重说明](../weights/README_zh.md)及[数据使用说明](DATA_USE_zh.md)。
