<a id="data-use-and-privacy"></a>

# 数据使用与隐私

[English](DATA_USE.md) · [简体中文](DATA_USE_zh.md)

Head2Field 分发包包含源码、文档、配置、schema、合成示例和预训练模型权重。
参与者影像、分割、网格、电极定位、FEM 输出、队列清单、标签、预测、训练统计与缓存、
日志及训练恢复状态均保存在软件包之外。

<a id="software-weights-and-datasets"></a>

## 软件、权重与数据集

源码使用 [GPL-3.0-only](LICENSE_GUIDE_zh.md) 许可证。数据集的访问和使用仍受上游条款约束。

wheel 和源码包包含模型权重、校验文件、模型说明、权重说明、许可证信息及数据使用声明。
权重是仅含模型参数的 state dictionary；仅含张量的格式并不替代训练数据的使用条款。
参见[权重说明](../weights/README_zh.md)。

影像数据和外部科学计算依赖需要按各自条款获取。仓库不提供参与者数据集、模板、图谱或
SimNIBS 安装环境。

<a id="keep-participant-data-out-of-source-changes"></a>

## 源码改动不包含参与者数据

报告问题或提供贡献示例时，使用小型合成输入。不要在提交和问题附件中加入参与者影像、
网格、FEM 标签、预测、被试或家系标识、分组清单、再识别密钥、训练恢复文件或凭据。

生成的研究输出与源码分开保存。本地开发检查可使用 `.work/` 下被忽略的任务目录。
共享日志或示例前，请移除参与者标识、账户信息和本地存储路径。

<a id="synthetic-examples"></a>

## 合成示例

生成器按固定规则创建小型非解剖数组和虚构标识，不变换、平均或重新分发参与者数据。
这些示例用于检查文件读取和软件流程；参与者数据上的模型精度需要另外评估。

<a id="hcp-acknowledgement-and-data-use-terms"></a>

## HCP 致谢与数据使用条款

本方法的开发受益于 Human Connectome Project 的 WU-Minn Consortium 数据。我们感谢该联盟在数据采集、处理方面的工作，以及 NIH 和 McDonnell Center 的支持。

HCP 用户，包括通过共享副本访问数据的用户，应接受适用的数据使用条款。这些条款涉及参与者隐私、机构研究要求、原始与衍生数据分发、致谢和方法引用：

- [HCP Young Adult 数据使用条款](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms)
- [WU-Minn HCP 开放访问数据使用条款及指定致谢文本](https://www.humanconnectome.org/storage/app/media/data_use_terms/DataUseTerms-HCP-Open-Access-26Apr2013.pdf)

使用 HCP 数据发表论文或进行报告时，应遵循官方条款规定的致谢文本，并引用相关采集与处理方法。共享原始或衍生数据时适用上游条款；本文不判定训练权重文件的法律分类。

<a id="research-use"></a>

## 科研使用

用户负责数据集访问、机构与伦理要求、安全存储，以及输入和依赖条款的遵守。
[模型说明](MODEL_CARD_zh.md)列出了非临床科研使用的假设。
软件不提供参与者同意、数据集访问、临床验证或经过校准的不确定性估计。
