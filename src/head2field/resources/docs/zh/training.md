<a id="training"></a>

# 训练

[English](../en/training.md) · [简体中文](training.md)

训练流程提供数据与物理损失、明确的数据集合同、数据准备命令、CUDA FP16 epoch 训练、确定性验证和完整本地恢复状态。影像、prepared subject、FEM 标签、缓存、统计量和物理池由用户准备，日志及恢复状态保存在本地。

<a id="execution-profiles"></a>

## 执行配置

以下命令使用 `configs/` 中的现有配置。仅安装 wheel 时，可按[上手指南](getting_started.md#configuration-files-after-wheel-installation)复制内置默认配置。

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

每个 epoch 访问全部训练被试，以及所有电极批次与查询分块组合。被试、源和查询顺序每轮确定性打乱，包含末尾不完整批次和分块，因此覆盖全部 75 个源电极对与 GM/WM 查询。epoch 步数是各被试 `ceil(75 / 8) * ceil(number_of_queries / 49152)` 之和。学习率在 epoch 内保持不变，仅在该轮结束后推进。

AMP 初始梯度 scale 为 1024，增长间隔为 2000。合成安装配置采用仅供测试的 8³ 小模型、CPU FP32 和两个紧凑 epoch，与真实数据训练设置分别管理。

<a id="data-and-physics-objective"></a>

## 数据与物理目标

v1 GP 目标采用：

- 固定 N128 GP 工厂和 75 个有序 source→Oz 电极对；
- 磁盘上的原始 `S/m` 标量电导率，以及共享的 float32 `sigma / 1.654` 网络变换；
- 全局、径向与高场稳定能量数据项，权重为 `0.60 / 0.30 / 0.10`；
- 分母 floor 比例 `0.01`，径向边界 50 与 100 mm，源对权重 `(alpha=1, sigma=40 mm)`，peak 分位数 0.90；
- 环流、WM/GM 法向电流连续性、WM/GM 切向电场连续性及通量算子，每类 512 个原语；
- `one_sided_excess_v1`、仅由训练数据拟合的 FEM scale floor 和 MAD，各物理项权重为 5，10,000 步内从 0 递增到 1 的 curriculum；
- world-RAS 向量分量，单位为 `V/m`，对应源电极 `+1 mA` 和 Oz `-1 mA`。

公式位于 [losses.py](../../src/head2field/training/losses.py) 和 [physics.py](../../src/head2field/training/physics.py)。模型预测与 FEM 电场使用同一组离散物理算子；预测值不能改变有效掩膜或拟合阈值。

每步数据损失缩放为 `M / (75 * N_train)`，在源批次与查询分块之间保持固定，包含末尾不完整批次。验证和测试复用训练确定的缩放。能量分母中位数在每个被试内部的有效源电场之间计算。数据划分、完整电场指标和计时方法见[评估与可重复实验](paper_reproduction.md)。

<a id="real-data-manifest"></a>

## 真实数据清单

机器 schema 为 [`training_dataset_v1.schema.json`](../../schemas/training_dataset_v1.schema.json)。全部路径相对于数据集清单，每个引用清单均记录明确的相对路径。每个匿名 `subject_key` 引用：

1. `prepared_subject_v1.json`；
2. `fem_labels_v1.json`；
3. `training_loss_statistics_v2.json`；
4. `training_physics_pool_v1.json`。

加载器重新验证 prepared subject 与全部 75 个 FEM 标签文件，核对源顺序、查询顺序、电导率表及 `sigma / 1.654` 变换、数组形状、dtype、有限值和文件大小。分组决定 `train`、`validation` 或可选 `test` 分区，同一组不能跨分区，训练与验证分区均为必需。

修改训练配置、分组、引用数据路径或源顺序会改变运行合同，已有状态不能直接恢复到不同合同的运行。

<a id="asset-preparation"></a>

## 资产准备

以下目标目录均必须尚不存在。命令使用 staging 目录，不覆盖已有输出。

可选的 query-major 目标缓存无需修改 FEM 标签：

```bash
head2field-prepare-cache \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/cache-v1
```

```bash
head2field-prepare-statistics \
  --prepared-subject prepared/subject-a \
  --fem-labels fem/subject-a \
  --output-dir derived/subject-a/loss-statistics-v1
```

物理准备分为三个阶段，保证验证 FEM 值不能参与阈值拟合：

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

<a id="synthetic-installation-and-resume-smoke"></a>

## 合成安装与恢复检查

生成包含一个虚构训练被试和一个虚构验证被试的紧凑数据集，保留真实 75 源顺序，仅包含 8³ 网格上的解析数组：

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

该流程以 CPU FP32 最多运行两个 epoch，经过模型 forward、稳定能量损失、四类物理算子、训练状态机和确定性验证。小模型仅供测试，不接受 N128 预训练初始化权重。

<a id="real-data-training"></a>

## 真实数据训练

创建新训练输出目录前先执行预检：

```bash
head2field-train-preflight \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json

head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --output-dir runs/custom-cuda-run
```

`--init-weights` 仅用于裸 `str -> FP32 Tensor` state dictionary。固定 N128 模型初始化前，CLI 还要求官方的 208 张量、2,607,870 标量 schema。优化器、调度器、步数、采样器和全部 RNG 重新初始化。

`--resume-state` 仅用于本训练代码创建的完整状态。PyTorch 完整状态使用 pickle，因此 CLI 还要求 `--trust-local-resume-state`；不要为不可信下载启用该选项。恢复始终写入新输出目录：

```bash
head2field-train \
  --config configs/training_v1.json \
  --dataset-manifest dataset/training_dataset_v1.json \
  --resume-state runs/part-a/train_state_step_000005000.pt \
  --trust-local-resume-state \
  --output-dir runs/part-b
```

完整本地状态包含模型、AdamW、余弦调度器、AMP scaler、全局步数、Python/NumPy/Torch CPU/CUDA RNG、命名的增强/dropout generator、采样器排列与游标、验证最优损失及模型、早停计数、精确运行合同和之前全部日志行。状态先写入同目录临时文件，刷新后原子发布，不覆盖已有文件。这些是本地资产，不能放入 `src/head2field/resources/weights/`。

CPU 确定性回归比较连续运行 `N` 步与运行 `K` 步、保存/重载、再继续至 `N` 步：模型张量、优化器、调度器、全部 RNG、采样器、早停状态和规范 JSONL 日志完全相同。CUDA 状态也保存，但 CPU 测试不证明 CUDA 位级等价。

<a id="validation-and-outputs"></a>

## 验证与输出

验证按固定顺序遍历全部验证被试及源/查询组合，使用 `model.eval()`、确定性分支，不使用训练 generator。组合数据与物理目标按被试平均，始终采用 `a(t)=1`。每轮验证指标、学习率和早停计数写入 step 日志。验证损失严格降低时重置计数并选取新的最优 checkpoint；连续五轮无改善时停止。

训练在新目录输出：

```text
training_log_v1.jsonl
best_model.pt
train_state_step_000000001.pt
train_state_step_000005000.pt
...
```

`best_model.pt` 是按验证损失选择的裸模型 state dictionary，返回的内存模型恢复到该权重以进行测试。恢复状态保留最新训练模型及完整优化器/调度器状态。中断运行导出已完成验证 epoch 中的最优 checkpoint。真实数据清单含 `test` 分区时，在达到 epoch 预算或早停后，使用恢复的最优模型评估一次，写入 `test_metrics_v1.json`。人为中途停止训练不触发测试评估。

这些输出是自定义模型产物。用于推理前，应记录对应配置、数据划分和评估结果。参与者数据由用户另行提供。
