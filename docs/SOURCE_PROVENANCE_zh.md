<a id="source-provenance"></a>

# 实现与依赖

[English](SOURCE_PROVENANCE.md) · [简体中文](SOURCE_PROVENANCE_zh.md)

模型、预处理、FEM 标签与训练组件源自 Head2Field 研究代码，并在此作为独立 Python 包维护。本指南说明当前模块职责及外部集成；张量维度与执行流程见[代码指南](zh/code_guide.md)。

## 包组件

下列路径相对于仓库根目录。

| 组件 | 主要源码路径 | 职责 |
| --- | --- | --- |
| 模型与结构 | `src/head2field/model.py`、`src/head2field/_spec.py` | 解剖编码、查询/电极 token、共享响应及 v1 模型创建函数 |
| 输入与几何 | `src/head2field/inputs.py`、`src/head2field/geometry.py` | 模型张量、体素/世界坐标转换及查询点与电极顺序 |
| 权重加载 | `src/head2field/weights.py` | 内置模型访问及严格 FP32 state dictionary 验证 |
| 推理 | `src/head2field/inference.py`、`src/head2field/cli.py` | 被试缓存、电极对/查询分块、电场数组与预测清单 |
| 预处理 | `src/head2field/preprocessing/` | T1 准备、外部 CHARM 调用、电导率及 模型输入文件整理 |
| FEM 标签 | `src/head2field/fem_labels/` | SimNIBS 执行、求解器输出验证、插值与标签清单 |
| 训练 | `src/head2field/training/` | 数据集读取、统计量、物理损失采样数据、损失、验证及本地恢复 |
| 资源 | `src/head2field/resource_paths.py`、`src/head2field/resources/` | 随包配置、schema、文档和预训练权重 |

## 验证与输出

加载器验证实际数组、几何、组织电导率、源/查询顺序、文件大小和单位。每个流程用 JSON 清单记录输入输出文件，并把结果写入新目录。恢复训练时，配置和数据文件应与上次运行一致。字段定义见[数据格式与检查](zh/data_contracts.md)，输出含义见[推理说明](zh/inference.md)。

模型创建函数与权重加载器使用相同的 v1 参数名及张量形状。软件、模型和数据 schema 版本分别管理。兼容性与用途见[模型说明](MODEL_CARD_zh.md)和[权重说明](../weights/README_zh.md)。

## 外部集成

PyTorch、NumPy 和 NiBabel 提供基础张量及影像运行时。SciPy 和 h5py 是预处理与 FEM 数据处理的可选依赖。SimNIBS/CHARM 4.6.0 单独安装，由预处理和 FEM adapter 调用。

包内不嵌入外部运行时、原生库、模板、图谱或参与者数据集。依赖及许可证声明见[第三方声明](THIRD_PARTY_NOTICES_zh.md)，数据集条款与致谢见[数据使用说明](DATA_USE_zh.md)。

## 安装包内容

wheel 包含项目代码、配置/schema 资源、文档与内置的仅含张量的 v1 权重；源码包另提供测试和示例生成器。参与者数据、生成的 FEM 或预测数组、训练缓存、日志和完整恢复状态单独保存。

测试和示例采用虚构标识与生成数组。开发与打包步骤见[贡献指南](CONTRIBUTING_zh.md)和[发布清单](RELEASE_CHECKLIST_zh.md)。
