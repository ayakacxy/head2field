<a id="third-party-notices-and-dependency-boundary"></a>

# 第三方声明与依赖边界

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES_zh.md)

本文记录上游依赖及软件分发包所包含的材料。项目 wheel 和源码分发包仅包含本项目 Python 源码、包资源（含预训练 v1 权重）、文档和许可证文件，不内嵌第三方 Python wheel、原生库、数据集、模板、图谱或 SimNIBS 源码及二进制文件。

<a id="runtime-and-optional-dependencies"></a>

## 运行时与可选依赖

| 组件 | 项目用途 | 上游许可证边界 |
| --- | --- | --- |
| PyTorch | 张量运行时、模型执行和训练 | 上游顶层许可证为 BSD-3-Clause；仓库及二进制分发包还包含受其他声明和许可证约束的组件。重新分发时以实际安装产物的 `LICENSE`/`NOTICE` 文件为准。 |
| NumPy | 数组处理与文件格式 | 顶层许可证为 BSD-3-Clause；源码及二进制产物可能包含适用其他条款的内置或生成组件。应保留并检查实际产物的许可证清单。 |
| NiBabel | NIfTI 输入输出与空间元数据 | 顶层许可证为 MIT；实际产物的上游 `COPYING` 文件还记录第三方代码与数据条款，重新分发时以其为准。 |
| SciPy | 刚体配准与数值预处理 | 上游项目采用 BSD-3-Clause。二进制 wheel 可能内置适用其他许可证的 OpenBLAS 和编译器运行库。 |
| h5py | FEM HDF5 验证与标签导出 | 上游项目采用 BSD-3-Clause；重新分发环境前应检查实际二进制 wheel 及链接的 HDF5/运行库声明。 |
| SimNIBS | 外部 CHARM 头模型和 TDCS 先导场后端 | 上游项目采用 GPL-3.0-only，并有独立的第三方声明与科学资产。用户单独安装，本项目不重新分发其软件。 |

<a id="development-only-tools"></a>

## 仅用于开发的工具

`pytest`、`jsonschema`、`Ruff`、`build`、`setuptools` 和 `wheel` 用于测试与打包，不由已安装运行时导入。准备发布时应记录其版本及对应产物的许可证清单。

<a id="what-a-release-artifact-contains"></a>

## 发布产物的内容

- 项目 wheel 包含 Python 代码和允许列表内的预训练 v1 `.pth` 资源及校验文件，排除其他 `.pth`、`.pt` 训练状态、NIfTI、网格、HDF5、NPY/NPZ、原生库和第三方数据集资产。
- 项目源码分发包应遵循相同的科学资产边界。
- 内置预训练模型为裸 state dictionary，不含优化器、调度器、参与者标识或训练恢复状态。
- 容器、Conda 环境、wheelhouse 或独立二进制程序属于不同分发产品，需要重新检查所有传递依赖与内置原生库的实际许可证清单。

<a id="authoritative-upstream-references"></a>

## 权威上游参考

- [SimNIBS 4.6.0 包元数据](https://github.com/simnibs/simnibs/blob/v4.6.0/pyproject.toml)
- [PyTorch 包元数据](https://github.com/pytorch/pytorch/blob/main/pyproject.toml)
- [NumPy 包元数据](https://github.com/numpy/numpy/blob/main/pyproject.toml)
- [NiBabel 仓库与许可证](https://github.com/nipy/nibabel)
- [SciPy 仓库](https://github.com/scipy/scipy)
- [h5py 仓库](https://github.com/h5py/h5py)

最终发布人员应将此清单与实际 wheel、源码包、内置预训练权重及构建/安装环境记录对照。复制或内嵌新的第三方材料会改变这一结论，需要重新审查。
