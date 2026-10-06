<a id="contributing"></a>

# 贡献指南

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING_zh.md)

欢迎为 Head2Field 作出贡献。请让改动范围清晰，说明解决的问题，并提供与改动规模相符的验证。

<a id="report-an-issue"></a>

## 报告问题

报告错误时，请提供软件版本、Python/PyTorch 版本、操作系统、相关命令或 API 调用，以及最小可复现示例。尽量使用合成输入，并从日志和附件中去除参与者标识、凭据及本地存储路径。

提出功能需求时，请说明预期研究流程，以及新功能如何影响 v1 输入输出格式。

<a id="development-setup"></a>

## 开发环境

在源码仓库根目录中执行：

```bash
python -m pip install -e '.[dev]'
```

开发预处理或 FEM adapter 时需单独安装 SimNIBS 4.6.0。真实数据训练需要 CUDA 版 PyTorch。单元测试使用合成输入；集成测试可能需要可选依赖。

生成数据、checkpoint、临时文件、缓存和构建产物应排除在版本控制之外。使用 `.work/` 下的任务目录进行本地检查：

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

从干净的工作副本中执行 `python -m build` 构建发布包，通过 `--outdir` 指定任务专用输出目录。打包验证步骤见[发布清单](RELEASE_CHECKLIST_zh.md)。

对于范围明确的改动，只选择相关测试文件即可。例如，推理与权重改动使用 `tests/test_inference.py`、`tests/test_weights.py` 和 `tests/test_cli_smoke.py`；包内文档改动使用 `tests/test_packaging_resources.py`。检查结束后，仅清理本次任务创建的文件和目录。

修改模型或训练代码前，请先阅读[代码指南](zh/code_guide.md)，了解模块职责、张量约定与数据流。

<a id="prepare-a-contribution"></a>

## 准备贡献

- 说明当前行为、预期行为及相关验证。
- 命令或输入输出格式发生变化时，同步更新文档和示例。
- 示例与测试使用相对路径和虚构标识。
- 对影响数值结果、顺序、验证或恢复行为的改动加入回归测试，并注明跳过的可选测试。
- 保留可选依赖的延迟导入，使最小推理流程不加载 SimNIBS、SciPy、h5py 或训练流水线。

<a id="scientific-compatibility"></a>

## 模型兼容性

v1 模型对参数名与形状、组织标签、电导率缩放、world-RAS 坐标、电极和查询点顺序、电流及单位有固定要求。修改这些内容时，请用相同输入比较数值结果，说明是否兼容已有权重和数据。如果不再兼容，应升级模型或数据格式版本，并在文档中说明。

进行优化时，请使用相同输入比较运行时间、内存占用与数值差异。合成单元测试用于验证软件行为；模型精度和泛化能力需要适当的评估数据。

<a id="documentation-language"></a>

## 文档语言

公开 API 名称和 CLI 帮助使用英文，新增源码注释与 docstring 使用中文。技术指南分别放在 `docs/en/` 和 `docs/zh/`，两边使用相同文件名；`docs/` 中的项目说明、根目录 README、示例和权重文档保留对应的 `_zh.md` 文件。命令、坐标、单位和训练参数应保持一致。正文链接保持当前语言，语言切换进入同一篇文档的另一语言版本。随包分发的文档需要同步到 `src/head2field/resources/`。全部指南见[文档目录](zh/README.md)。

<a id="attribution"></a>

## 贡献者署名

项目作者与维护者是 [ayakacxy](https://github.com/ayakacxy)。Codex（OpenAI）协助开发、文档整理与验证。
每次提交保留实际作者信息。对于 ayakacxy 作为作者且 Codex 参与的提交，在提交说明末尾空一行后加入：

```text
Co-authored-by: Codex <codex@openai.com>
```

<a id="data-dependencies-and-licensing"></a>

## 数据、依赖与许可证

请不要把参与者数据、被试分组清单、训练恢复文件或生成的研究结果提交到仓库。安装包只包含指定的预训练 v1 权重及校验文件，新增权重时需同时提供对应模型说明。参见[数据使用说明](DATA_USE_zh.md)和[权重说明](../weights/README_zh.md)。

记录复制或改编代码的来源与许可证。模板、图谱、数据集、二进制文件和其他第三方文件需要具备适当的分发条款。贡献应兼容项目的 [GPL-3.0-only 许可证](LICENSE_GUIDE_zh.md)。
