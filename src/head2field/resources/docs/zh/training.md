<a id="training"></a>

# 训练

[English](../en/training.md) · [简体中文](training.md)

训练从预处理后的被试数据和 FEM 标签开始。本指南依次介绍损失统计、物理采样数据、CUDA FP16 训练、验证和断点恢复。训练同时约束电场误差与物理性质，日志和恢复文件保存在输出目录。

<a id="execution-profiles"></a>

## 选择训练配置

训练完整模型时选择真实数据配置，检查安装时选择合成数据配置。以下命令使用 `configs/` 中的文件；仅安装 wheel 时，请先按[上手指南](getting_started.md#configuration-files-after-wheel-installation)将内置默认配置复制到工作目录。

| 配置 | 用途 | 执行方式 |
| --- | --- | --- |
| [training_v1.json](../../configs/training_v1.json) | 固定 v1 网络的真实数据训练 | CUDA FP16 AMP、完整 epoch、验证和早停 |
| [training_smoke_v1.json](../../configs/training_smoke_v1.json) | 合成输入的安装与恢复检查 | 仅供测试的小型模型、CPU FP32、两个 epoch |

<a id="training-settings"></a>

## 训练设置

默认真实数据训练配置采用以下设置：

| 设置 | 数值 |
| --- | --- |
| 优化器 | AdamW |
| 初始学习率 | `2e-4` |
| Weight decay | `1e-4` |
| 被试批次 | 每步一个被试 |
| 电极批次 | 八个 source→Oz 电极对 |
| 查询分块 | 49,152 个 GM/WM 位置 |
| 精度 | CUDA FP16 自动混合精度 |
| 梯度范数裁剪 | `1.0`，在梯度 unscale 后执行 |
| 最大训练预算 | 25 个 epoch |
| 学习率调度 | 以 epoch 为单位、覆盖 25 个 epoch 的余弦退火，最低 LR `0` |
| 验证 | 每轮结束，组合目标采用 `a(t)=1` |
| 早停 | 连续五轮验证损失未降低 |
| 测试 checkpoint | 按验证损失选择的最优 checkpoint |

每个 epoch 覆盖全部训练被试、75 个源电极对和所有 GM/WM 查询点。采样器按本轮随机种子打乱被试、源和查询顺序，也处理末尾不完整批次。每个被试需要 `ceil(75 / 8) * ceil(number_of_queries / 49152)` 步，一轮总步数为所有被试之和。学习率在一轮内保持不变，该轮结束后再更新。

AMP 的初始梯度缩放值为 1024，增长间隔为 2000。合成配置改用 8³ 小模型，在 CPU 上以 FP32 运行两个短 epoch，用于检查安装和恢复流程。

<a id="data-and-physics-objective"></a>

## 数据与物理目标

默认 GP 配置采用以下数据损失与物理损失设置：

- N128 GP 模型和 75 个有序 source→Oz 电极对；
- 磁盘上的原始 `S/m` 标量电导率，以及共享的 float32 `sigma / 1.654` 网络变换；
- 全局、径向与高场稳定能量数据项，权重为 `0.60 / 0.30 / 0.10`；
- 分母下限比例 `0.01`，径向分区阈值为 50 与 100 mm，源对权重 `(alpha=1, sigma=40 mm)`，peak 分位数 0.90；
- 环流、WM/GM 法向电流连续性、WM/GM 切向电场连续性及通量计算，每类使用 512 组积分采样数据；
- `one_sided_excess_v1`、仅由训练数据拟合的 FEM scale floor 和 MAD，各物理项权重为 5，10,000 步内从 0 递增到 1 的 curriculum；
- world-RAS 向量分量，单位为 `V/m`，对应源电极 `+1 mA` 和 Oz `-1 mA`。

数据损失公式见 [losses.py](../../src/head2field/training/losses.py)，物理算子见 [physics.py](../../src/head2field/training/physics.py)。预测电场与 FEM 电场使用同一组算子，有效采样位置和拟合阈值由准备好的数据确定，训练中保持不变。

每步使用相同的数据损失缩放 `M / (75 * N_train)`，末尾不完整的源批次和查询分块也不例外。验证和测试使用训练分区确定的缩放。能量分母采用每个被试内部有效源电场的中位数。数据划分、完整电场指标和计时方法见[评估与可重复实验](paper_reproduction.md)。

<a id="real-data-manifest"></a>

## 训练数据清单

在数据集清单中列出训练数据，字段定义见 [`training_dataset_v1.schema.json`](../../schemas/training_dataset_v1.schema.json)。文件路径相对于该清单。每个匿名 `subject_key` 对应以下四个文件：

1. `prepared_subject_v1.json`；
2. `fem_labels_v1.json`；
3. `training_loss_statistics_v2.json`；
4. `training_physics_pool_v1.json`。

加载时，程序检查预处理数据和全部 75 个 FEM 标签文件。源电极与查询点顺序、电导率表 ID 及数值、`sigma / 1.654` 变换 ID、数组形状、数据类型和记录的文件大小必须一致，数值中不能有 NaN 或 Inf。每组只能属于 `train`、`validation` 或可选的 `test` 分区之一，训练与验证分组均不能缺少。

恢复训练时，保留原有配置、分组、文件路径和源电极顺序。需要修改这些设置时，请开始新运行。

<a id="asset-preparation"></a>

## 训练数据准备

为以下每条命令指定新的目标目录。准备过程中先写入暂存目录，不覆盖已有结果。

query-major 目标缓存是可选项，用于调整训练时的读取顺序，不修改 FEM 标签：

```bash
head2field-prepare-cache \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/cache-v1
```

从已检查的 FEM 标签计算稳定能量统计，在数据集清单中记录统计清单路径和文件大小：

```bash
head2field-prepare-statistics \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/loss-statistics-v1
```

物理采样数据按三步准备：为每个被试生成原始数据，用训练被试拟合校准统计，再将同一统计用于各被试。将拟合单独列为一步，可以避免验证集 FEM 值参与阈值设置：

```bash
# 对每个训练和验证被试生成几何及原始 FEM 残差/尺度，不拟合阈值。
head2field-prepare-physics raw \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/physics-raw-v1

# 仅传入训练分区的原始池；支持重复指定该选项。
head2field-prepare-physics fit \
  --train-raw-pool derived/subject-a/physics-raw-v1/training_physics_raw_pool_v1.json \
  --train-raw-pool derived/subject-b/physics-raw-v1/training_physics_raw_pool_v1.json \
  --output-dir derived/train-only-physics-statistics-v1

# 将同一份训练集拟合统计应用于每个训练或验证原始池。
head2field-prepare-physics finalize \
  --raw-pool derived/subject-a/physics-raw-v1/training_physics_raw_pool_v1.json \
  --statistics derived/train-only-physics-statistics-v1/training_physics_statistics_v1.json \
  --output-dir derived/subject-a/physics-pool-v1
```

`fit` 只接收训练分区的原始数据，并记录其路径、源列表和数组大小。验证数据仅用于 `raw` 和 `finalize`，不要传给 `fit`。

<a id="synthetic-installation-and-resume-smoke"></a>

## 合成安装与恢复检查

检查安装时，生成一个虚构训练被试和一个虚构验证被试。数据为 8³ 网格上的解析数组，源电极顺序与真实输入的 75 源顺序相同：

```bash
head2field-make-synthetic-training \
  --output-dir synthetic-training-v1

head2field-train-preflight \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset synthetic-training-v1/synthetic_training_dataset_v1.json

head2field-train \
  --config configs/training_smoke_v1.json \
  --synthetic-dataset synthetic-training-v1/synthetic_training_dataset_v1.json \
  --output-dir runs/synthetic-smoke
```

示例在 CPU 上以 FP32 最多运行两个 epoch，经过前向计算、稳定能量损失、四类物理算子、训练状态更新和验证。该小模型不能加载完整 N128 模型的预训练权重，输出只用于检查软件流程，不代表预测精度。

<a id="real-data-training"></a>

## 真实数据训练

开始训练前，先检查配置和数据集：

```bash
head2field-train-preflight \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json

head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --output-dir runs/custom-cuda-run
```

从已有模型权重开始新训练时，使用 `--init-weights`，输入应是裸 `str -> FP32 Tensor` state dictionary。N128 模型要求恰好 208 个张量、2,607,870 个标量，参数名和形状均需匹配。优化器、调度器、步数、采样器和全部随机数生成器从初始状态开始。

继续中断的训练时，使用 `--resume-state` 加载本训练代码保存的完整状态。该文件使用 pickle，因此命令还要求 `--trust-local-resume-state`；仅对可信文件启用它，并为后续训练指定新输出目录：

```bash
head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --resume-state runs/part-a/train_state_step_000005000.pt \
  --trust-local-resume-state \
  --output-dir runs/part-b
```

恢复文件保存以下内容：

- 模型、AdamW、学习率调度器和 AMP scaler 状态；
- 当前步数、采样顺序和当前位置，以及 Python、NumPy、Torch CPU/CUDA 和增强/dropout 的随机数状态；
- 验证最优模型、验证损失和早停计数；
- 本次运行的配置、数据集记录和已有日志。

程序先将状态写入并刷新同目录的临时文件，再原子保存，不覆盖已有文件。这类本地训练文件不要放入 `src/head2field/resources/weights/`。

CPU 恢复测试比较连续运行 `N` 步与运行 `K` 步后保存、加载并继续到 `N` 步的结果。模型、优化器、调度器、随机数状态、采样器、早停状态和规范化 JSONL 日志逐项相同。CUDA 状态也会保存，但这项 CPU 检查不能证明 GPU 的逐位可重复性。

<a id="validation-and-outputs"></a>

## 验证与输出

每轮训练结束后，验证按固定顺序遍历全部验证被试及源/查询组合，使用 `model.eval()` 和确定性分支，不使用训练随机数生成器。数据与物理目标按被试平均，始终取 `a(t)=1`。

step 日志记录验证指标、学习率和早停计数。验证损失严格降低时，程序保存新的最优 checkpoint 并重置计数；连续五轮没有改善时停止训练。

训练在新目录输出：

```text
training_log_v1.jsonl
best_model.pt
train_state_step_000000001.pt
train_state_step_000005000.pt
...
```

`best_model.pt` 保存验证损失最低的裸模型 state dictionary。测试前，内存模型会恢复到这份权重；恢复文件则保留最新训练模型及完整优化器和调度器状态。

训练中断时，程序导出已完成验证轮次中的最优 checkpoint。真实数据清单包含 `test` 分区时，在达到 epoch 上限或早停后，用恢复的最优模型评估一次，写入 `test_metrics_v1.json`。中途停止训练不会触发测试评估。

自行训练的权重应与配置、数据划分和评估结果一同保存。参与者数据需另行获取。
