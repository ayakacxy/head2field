<a id="contributing"></a>

# 贡献指南

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING_zh.md)

欢迎报告问题、修正文档或提交范围明确的代码改动。请说明要解决的问题、改动后的行为，
以及与本次改动相关的检查结果。

<a id="report-an-issue"></a>

## 报告问题

请提供软件版本、Python/PyTorch 版本、操作系统、相关命令或 API 调用，以及最小可复现示例。
合成输入通常更便于分享。日志和附件中应去除参与者标识、凭据及本地存储路径。

提出功能需求时，请说明它支持的研究流程，以及对模型输入输出格式的影响。

<a id="development-setup"></a>

## 开发环境

在源码仓库根目录安装可编辑版本：

```bash
python -m pip install -e '.[dev]'
```

开发预处理或 FEM adapter 时，需要单独安装 SimNIBS 4.6.0。真实数据训练需要 CUDA 版 PyTorch。
单元测试使用合成输入，部分集成测试需要可选依赖。

生成数据、checkpoint、临时文件、缓存和构建产物不应进入版本控制。
以下命令将本地检查文件放在 `.work/` 下被忽略的任务目录中：

```bash
mkdir -p .work/dev/tmp .work/dev/cache .work/dev/pycache
export TMPDIR="$PWD/.work/dev/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PWD/.work/dev/cache"
export PYTHONPYCACHEPREFIX="$PWD/.work/dev/pycache"
export PIP_CACHE_DIR="$PWD/.work/dev/cache/pip"
export RUFF_CACHE_DIR="$PWD/.work/dev/cache/ruff"
export TORCH_HOME="$PWD/.work/dev/cache/torch"

python -m pytest \
  --basetemp .work/dev/pytest \
  -o cache_dir=.work/dev/cache/pytest
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m compileall -q src tests examples
```

按改动选择相应的检查。推理和权重改动可运行 `tests/test_inference.py`、
`tests/test_weights.py` 和 `tests/test_cli_smoke.py`；包内文档改动可运行
`tests/test_packaging_resources.py`。范围较小的改动只需选择相关测试文件。
结束后，仅清理本次任务创建的文件。

从干净的工作副本执行 `python -m build` 构建发布包，并通过 `--outdir` 指定任务输出目录。
打包检查见[发布清单](RELEASE_CHECKLIST_zh.md)。
修改模型或训练代码前，先阅读[代码指南](zh/code_guide.md)，了解模块职责、张量约定和数据流。

<a id="prepare-a-contribution"></a>

## 准备贡献

- 说明当前行为、预期行为和已经完成的检查。
- 修改命令或输入输出格式时，同步更新指南和示例。
- 示例与测试使用相对路径和虚构标识。
- 影响数值结果、顺序、检查或恢复行为的改动应加入回归测试，并注明跳过的可选测试。
- 保留可选依赖的延迟导入，使最小推理流程不加载 SimNIBS、SciPy、h5py 或训练流水线。

<a id="scientific-compatibility"></a>

## 模型兼容性

已有权重和预处理数据依赖参数名与形状、组织标签、电导率缩放、world-RAS 坐标、
电极与查询顺序、电流约定和输出单位。修改这些内容时，请用相同输入比较结果，
并说明与已有文件的兼容性。不兼容的改动需要新的模型或数据格式版本。

优化时，使用相同输入比较运行时间、内存占用和数值结果。合成单元测试检查软件行为；
预测精度和泛化能力需要适当的评估数据。

<a id="documentation-language"></a>

## 文档语言

代码、注释、docstring、公开 API 名称、CLI 信息和主文档均使用英文。
公开指南另外提供简体中文翻译。

技术指南放在 `docs/en/` 和 `docs/zh/`，两边使用相同文件名。
`docs/` 中的项目说明、根目录 README、示例和权重说明保留 `_zh.md` 对应文件。
命令、坐标、单位和训练设置应保持一致。正文链接保持当前语言，
语言切换则进入同一篇文档的另一语言版本。

修改文档时，同步更新 `src/head2field/resources/` 中随包分发的副本。
全部指南见[文档目录](zh/README.md)。

<a id="attribution"></a>

## 贡献者署名

提交时使用贡献者本人的 Git 作者信息。改编已有代码时，请保留原有版权和许可证声明。

<a id="data-dependencies-and-licensing"></a>

## 数据、依赖与许可证

参与者数据、被试分组清单、完整训练恢复状态和生成的研究结果不应随源码提交。
内置模型权重和校验文件作为包资源提供，模型结构和输入要求另有说明。
参见[数据使用说明](DATA_USE_zh.md)和[权重说明](../weights/README_zh.md)。

复制或改编代码时，记录来源和许可证。模板、图谱、数据集、二进制文件和其他第三方文件
需要具备适当的分发条款。贡献应兼容项目的 [GPL-3.0-only 许可证](LICENSE_GUIDE_zh.md)。
