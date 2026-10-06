<a id="data-use-and-privacy"></a>

# 数据使用与隐私

[English](DATA_USE.md) · [简体中文](DATA_USE_zh.md)

软件分发包包含源码、文档、配置、schema、合成示例和预训练 v1 权重。参与者图像、分割、网格、电极定位、FEM 输出、队列清单、标签、预测、训练统计与缓存、日志及训练恢复状态均不在安装包包含的内容内。

<a id="software-weights-and-datasets"></a>

## 软件、权重与数据集

软件采用 [GPL-3.0-only](LICENSE_GUIDE_zh.md) 许可证。代码许可证不授予上游数据集访问权限，也不替代其数据使用条款。

wheel 和源码包内置已验证的 v1 模型权重及校验文件，并附带模型说明、权重说明、许可证信息和数据使用声明。已验证的 v1 文件是裸模型 state dictionary；仅含张量的格式并不决定其训练数据的分发条款。参见[权重说明](../weights/README_zh.md)。

用户应按适用条款获取自己的影像数据并安装外部科学计算依赖。本仓库不提供参与者数据集、模板、图谱或 SimNIBS 安装环境。

<a id="keep-participant-data-out-of-source-changes"></a>

## 源码改动不包含参与者数据

不要将参与者级图像、网格、FEM 标签、预测、被试或家系标识、分组清单、再识别密钥、训练恢复包或凭据提交到源码改动或问题报告。复现软件问题时使用小型合成示例。

生成的研究输出应存放于独立的数据位置。本地开发检查可使用被忽略的 `.work/` 任务目录。共享日志或示例前，请移除参与者标识、账户信息和本地存储路径。

<a id="synthetic-examples"></a>

## 合成示例

随项目提供的生成器按固定规则生成小型数组和虚构被试名称，不变换、平均或重新分发真实参与者数据。这些示例用于检查文件读取和软件流程，不能用来评估真实数据上的模型精度。

<a id="hcp-acknowledgement-and-data-use-terms"></a>

## HCP 致谢与数据使用条款

本方法的开发受益于 Human Connectome Project 的 WU-Minn Consortium 数据。我们感谢该联盟在数据采集、处理方面的工作，以及 NIH 和 McDonnell Center 的支持。

HCP 用户，包括通过共享副本访问数据的用户，应接受适用的数据使用条款。这些条款涉及参与者隐私、机构研究要求、原始与衍生数据分发、致谢和方法引用：

- [HCP Young Adult 数据使用条款](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms)
- [WU-Minn HCP 开放访问数据使用条款及指定致谢文本](https://www.humanconnectome.org/storage/app/media/data_use_terms/DataUseTerms-HCP-Open-Access-26Apr2013.pdf)

使用 HCP 数据发表论文或进行报告时，应遵循官方条款规定的致谢文本，并引用相关采集与处理方法。共享原始或衍生数据时适用上游条款；本文不判定训练权重文件的法律分类。

<a id="research-use"></a>

## 科研使用

用户负责数据集访问、机构与伦理要求、安全存储，以及输入与依赖条款的遵守。模型支持[模型说明](MODEL_CARD_zh.md)所列假设下的非临床研究，不提供参与者同意、数据集访问、临床验证或校准的不确定性估计。
