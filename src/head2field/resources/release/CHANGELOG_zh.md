<a id="changelog"></a>

# 更新日志

[English](CHANGELOG.md) · [简体中文](CHANGELOG_zh.md)

软件和模型的版本分别记录。

## 未发布

- 按安装、输入准备、训练、推理、评估和运行时间测量整理用户指南。
- 补充权重兼容性和操作步骤，并同步中英文文档及随包副本。

<a id="010"></a>

## 0.1.0

### 安装与分发

- 提供 `head2field` Python 包、九个工作流程 CLI 入口，以及预处理/FEM 可选依赖。
- 在 `v0.1.0` 标签下提供可安装 wheel、源码包和 `SHA256SUMS`。
- 提供中英文 README、技术指南、模型与数据使用说明、示例，以及配置/schema 资源。

<a id="bundled-pretrained-model"></a>

### 内置预训练模型

- 两种分发包均包含仅含张量的 Head2Field v1 权重和校验文件，默认自动加载。
- 支持通过 `--weights` 选择兼容的自定义权重。
- 提供固定的 v1 模型创建函数和严格的 FP32 state dictionary 加载，
  检查参数名、张量形状和标量总数。

<a id="workflows"></a>

### 工作流程

- 提供 T1 准备、外部 CHARM 头模型、标量电导率、GM/WM 查询组装和标准 cap 处理。
- 提供 Oz 参考 SimNIBS 先导场生成、求解器输出检查和有序 FEM 标签导出。
- 支持分块推理、被试特征缓存、电流模式合成和输出清单。
- 提供数据集、缓存、统计量及物理损失采样数据准备，检查坐标、源电极顺序和单位。

<a id="training"></a>

### 训练

- 支持 CUDA FP16 AMP 和 AdamW，初始学习率 `2e-4`、weight decay `1e-4`，
  每步八个 source→Oz 电极对和 49,152 个查询。
- 稳定能量监督采用 50/100 mm 径向分区、`training_loss_statistics_v2`，
  全局/径向/高场权重为 0.60/0.30/0.10。
- 积分物理损失仅用训练数据校准，四项权重均为 5。
- 使用完整 epoch、最多 25 轮、余弦调度、验证、五轮早停，以及验证最优模型选择。
- 提供完整本地恢复状态和可选的最终测试评估。
- 提供小型 CPU FP32 合成配置，用于检查安装和恢复流程。

### 兼容性与项目资源

- 提供 `Head2FieldV1`、`create_head2field_v1()`、
  `load_head2field_v1_weights()` 和 `Head2FieldInferenceSession`。
- 使用各向同性标量电导率，输出 `[75, Q, 3]` world-RAS 向量，
  单位为 V/m，对应源 +1 mA、Oz −1 mA。
- 软件分发包不含参与者数据、生成的研究数组、自定义 checkpoint 和完整训练恢复状态。
