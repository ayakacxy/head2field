<a id="release-checklist"></a>

# 发布清单

[English](RELEASE_CHECKLIST.md) · [简体中文](RELEASE_CHECKLIST_zh.md)

每次软件发布均可使用本清单。这是可重复的维护者流程，不是已经完成检查的历史记录。

<a id="upload-inventory"></a>

## 上传内容

以包含 `pyproject.toml` 的目录作为仓库根目录，提交下列文件和目录，包括 `src/` 中的包资源副本：

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

wheel 是可直接安装的发布资产，包含 Python 包、资源及预训练 Head2Field v1 权重。源码分发包也包含同一份允许列表中的权重，可随 wheel 一同发布。两类分发包的权重说明、校验文件、许可证信息和数据使用声明应保持一致。

`.gitignore` 和源码打包清单排除 `.work/`、缓存、构建产物、自定义 checkpoint 及生成的研究数据，显式保留已验证的预训练 v1 权重。

<a id="version-and-documentation"></a>

## 版本与文档

- [ ] 保持 `pyproject.toml`、`CITATION.cff`、包元数据和发布说明中的软件版本一致。
- [ ] 检查 `docs/CHANGELOG.md`，记录影响用户的兼容性变化。
- [ ] 对齐中英文文档，包括 `docs/en/` 与 `docs/zh/` 内的同名文件；核验正文链接保持当前语言、语言切换进入同一篇文档。
- [ ] 对照发布源码核验安装命令、CLI 示例和相对文档链接。
- [ ] 同步包资源与顶层配置、schema 和文档。
- [ ] 贡献者、仓库、论文与 DOI 元数据应使用已验证信息；信息可用时补齐，不编造标识。

<a id="software-verification"></a>

## 软件验证

运行时直接核对参数、分组、相对路径、文件大小、坐标顺序、shape、dtype 和有限值。训练恢复比较完整配置与数据清单；统计必须按当前 50/100 mm 设置和训练划分生成。

<a id="build-and-artifact-contents"></a>

## 构建与发布包内容

<a id="model-weights-and-data-use-notices"></a>

## 模型权重与数据使用声明

- [ ] 随内置权重提供适用许可证、数据使用声明、版本、模型合同及校验值。
- [ ] 确认权重仅包含预期模型张量，不含优化器/调度器/scaler、步数、指标、标识或未使用的张量存储载荷。
- [ ] 提供数据集致谢与相关方法引用。使用 HCP 时遵循[数据使用说明](DATA_USE_zh.md)所链接的官方致谢要求。
- [ ] 在发布文档中分别说明代码许可证、权重声明和上游数据集条款。

<a id="distribution-verification"></a>

## 分发验证

- [ ] 发布已审查的源码 tag、发布说明及预期资产。
- [ ] 检查仓库、Release、引用和下载链接可访问。
- [ ] 读回发布版本及资产元数据，核对下载文件与记录的校验值。
