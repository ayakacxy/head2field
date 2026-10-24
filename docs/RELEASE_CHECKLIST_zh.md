<a id="release-checklist"></a>

# 发布清单

[English](RELEASE_CHECKLIST.md) · [简体中文](RELEASE_CHECKLIST_zh.md)

准备软件发布时，按以下清单检查。

<a id="upload-inventory"></a>

## 上传内容

以包含 `pyproject.toml` 的目录作为仓库根目录。
提交以下内容，以及 `src/` 中对应的包资源副本：

| 内容 | 仓库路径 |
| --- | --- |
| Python 包及包资源 | `src/` |
| 默认配置与 schema | `configs/`、`schemas/` |
| 项目说明、技术指南与项目主图（PNG 和 PDF） | `docs/` |
| 合成示例生成器与测试 | `examples/`、`tests/` |
| 中英文项目概览 | `README.md`、`README_zh.md` |
| 引用与许可证 | `CITATION.cff`、`LICENSE` |
| 安装与源码打包 | `pyproject.toml`、`MANIFEST.in`、`.gitignore` |
| 内置预训练模型与校验文件 | `src/head2field/resources/weights/` |

wheel 包含可安装的 Python 包、资源和预训练 Head2Field 权重，源码包包含同一份内置权重。
两种分发包的权重说明、校验文件、许可证信息和数据使用声明应保持一致。

`.gitignore` 和源码打包清单排除 `.work/`、缓存、构建产物、自定义 checkpoint
和生成的研究数据，随包提供的预训练权重作为例外保留。

<a id="version-and-documentation"></a>

## 版本与文档

- [ ] 保持 `pyproject.toml`、`CITATION.cff`、包元数据和发布说明中的软件版本一致。
- [ ] 在 `docs/CHANGELOG.md` 中记录影响用户的改动和兼容性说明。
- [ ] 同步中英文文档，检查 `docs/en/` 与 `docs/zh/` 中的同名文件、
  正文同语言链接和同篇文档的语言切换。
- [ ] 对照发布源码检查安装命令、CLI 示例和相对链接。
- [ ] 将随包配置、schema 和文档与源文件同步。
- [ ] 使用已核实的贡献者、仓库、论文和 DOI 信息，补齐已有资料，不编造标识。

<a id="software-verification"></a>

## 软件验证

- [ ] 在具备所需 SimNIBS 和 CUDA 依赖的环境中运行完整测试集，
  解决失败项，并排查跳过的测试。
- [ ] 检查推理、预处理、FEM 数据检查、训练、恢复、CLI 入口和包资源。
- [ ] 按仓库配置运行 Markdown lint，并检查本地链接；仓库根目录只保留两个 README。
- [ ] 记录软件版本、GPU、测试结果和发布文件校验值。
  模型精度使用评估数据测量，合成测试用于检查软件行为。

<a id="build-and-artifact-contents"></a>

## 构建与发布包内容

- [ ] 在 `.work/` 下的任务目录中构建已审查源码，将临时、缓存和输出路径都设为该目录。
- [ ] 先构建源码包，再从解包后的内容构建 wheel，检查源码包是否完整。
- [ ] 检查两种分发包中的代码、资源、文档、许可证、引用信息及唯一的内置预训练权重。
- [ ] 排除本地缓存、测试输出、原始参与者数据、生成的研究结果和自定义 checkpoint，
  并逐字节比较源文件与资源副本。
- [ ] 将 wheel 安装到独立的任务目录，针对实际安装运行资源与元数据测试，
  并检查全部九个 CLI 的帮助命令。
- [ ] 为 wheel 和源码包生成 `SHA256SUMS`。

<a id="model-weights-and-data-use-notices"></a>

## 模型权重与数据使用声明

- [ ] 随权重提供许可证、数据使用声明、版本、模型说明和校验值。
- [ ] 检查文件仅含预期模型张量，不含优化器、调度器、scaler、步数、指标、标识
  或未使用的张量存储载荷。
- [ ] 提供数据集致谢和方法引用；使用 HCP 数据时，
  遵循[数据使用说明](DATA_USE_zh.md)所链接的官方要求。
- [ ] 说明代码、权重和上游数据集分别适用哪些声明。

<a id="distribution-verification"></a>

## 发布检查

- [ ] 按实际贡献记录提交署名，再创建指向该源码提交的版本 tag。
- [ ] 发布 tag、发布说明、wheel、源码包和 `SHA256SUMS`；
  tag 与软件版本保持一致，正式版关闭 prerelease 标记。
- [ ] 保留所有者选定的仓库可见性，并用相应访问权限检查仓库、Release、引用和下载链接。
- [ ] 读取发布后的名称、标签、draft/prerelease 状态、提交和下载文件；
  重新下载文件，对比记录的 SHA-256 校验值。
- [ ] 更新两个 README 的版本徽章和已发布版本链接。
