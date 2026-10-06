<a id="changelog"></a>

# 更新日志

[English](CHANGELOG.md) · [简体中文](CHANGELOG_zh.md)

软件和模型使用各自的版本号。

## 未发布

- 用户指南按安装、输入准备、训练、推理、评估和运行时间测量组织。
- 明确权重兼容性，补齐工作流程说明，保持中英文文档及随包副本一致。

<a id="010"></a>

## 0.1.0

### 安装与分发

- 提供 `head2field` Python 包、九个工作流程 CLI 入口，以及预处理/FEM 可选依赖。
- `v0.1.0` 标签对应可安装 wheel、源码包和 `SHA256SUMS`。
- 提供中英文 README、技术指南、模型与数据使用说明、示例及随包配置/schema 资源。

<a id="bundled-pretrained-model"></a>

### 内置预训练模型

- 两种分发包均提供仅含张量的 Head2Field v1 权重及校验文件，默认自动加载。
- `--weights` 可选择兼容的自定义权重。
- 固定 v1 工厂及严格 FP32 state dictionary 加载器核对参数名、张量形状和标量总数。

<a id="workflows"></a>

### 工作流程

- T1 准备、外部 CHARM 头模型、标量电导率、GM/WM 查询组装及标准 cap 处理。
- Oz 参考 SimNIBS 先导场生成、求解器输出验证和有序 FEM 标签导出。
- 分块先导场推理、被试特征缓存、电流模式合成及明确的输出清单。
- 数据集、缓存、统计量和物理损失采样数据准备，检查坐标、源电极顺序和单位。

<a id="training"></a>

### 训练

- CUDA FP16 AMP 和 AdamW，初始学习率 `2e-4`，weight decay `1e-4`，每步八个 source→Oz 电极对和 49,152 个查询。
- 稳定能量监督采用 50/100 mm 径向分区及 `training_loss_statistics_v2`，全局/径向/高场权重为 0.60/0.30/0.10。
- 积分物理损失仅使用训练集校准，四项权重均为 5。
- 完整 epoch、最多 25 轮、余弦调度、验证、五轮早停及验证最优模型选择。
- 完整本地恢复状态及可选的最终测试评估。
- 独立的小型 CPU FP32 合成配置用于安装与恢复检查。

### 兼容性与项目资源

- v1 模型 API 包括 `Head2FieldV1`、`create_head2field_v1()`、`load_head2field_v1_weights()` 和 `Head2FieldInferenceSession`。
- 模型采用各向同性标量电导率，输出 `[75, Q, 3]` world-RAS 向量；单位为 V/m，对应源 +1 mA、Oz −1 mA。
- 软件分发包排除参与者数据、生成的研究数组、自定义 checkpoint 和完整训练恢复状态。
