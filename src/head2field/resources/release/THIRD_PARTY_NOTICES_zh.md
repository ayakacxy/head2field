<a id="third-party-notices-and-dependency-boundary"></a>

# 第三方软件与许可证

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES_zh.md)

wheel 和源码包包含 Head2Field 的 Python 代码、包资源（含预训练权重）、文档和许可证文件。
第三方 Python wheel、原生库、数据集、模板、图谱，以及 SimNIBS 源码和二进制文件，
均需另行安装或获取。

<a id="runtime-and-optional-dependencies"></a>

## 运行时与可选依赖

| 组件 | 项目用途 | 上游许可证及注意事项 |
| --- | --- | --- |
| PyTorch | 张量运行时、模型执行和训练 | 上游顶层许可证为 BSD-3-Clause；仓库及二进制分发包还包含受其他声明和许可证约束的组件。重新分发时以实际安装产物的 `LICENSE`/`NOTICE` 文件为准。 |
| NumPy | 数组处理与文件格式 | 顶层许可证为 BSD-3-Clause；源码及二进制产物可能包含适用其他条款的内置或生成组件。应保留并检查实际产物的许可证清单。 |
| NiBabel | NIfTI 输入输出与空间元数据 | 顶层许可证为 MIT；实际产物的上游 `COPYING` 文件还记录第三方代码与数据条款，重新分发时以其为准。 |
| SciPy | 刚体配准与数值预处理 | 上游项目采用 BSD-3-Clause。二进制 wheel 可能内置适用其他许可证的 OpenBLAS 和编译器运行库。 |
| h5py | FEM HDF5 验证与标签导出 | 上游项目采用 BSD-3-Clause；重新分发环境前应检查实际二进制 wheel 及链接的 HDF5/运行库声明。 |
| SimNIBS | 外部 CHARM 头模型和 TDCS 先导场后端 | 上游项目采用 GPL-3.0-only，并有独立的第三方声明与数据和模型文件。用户单独安装，本项目不重新分发其软件。 |

<a id="development-only-tools"></a>

## 仅用于开发的工具

`pytest`、`jsonschema`、`Ruff`、`build`、`setuptools` 和 `wheel` 用于测试与打包，
不由已安装运行时导入。准备发布时，记录这些工具的版本及对应产物的许可证清单。

<a id="what-a-release-artifact-contains"></a>

## 发布包包含哪些文件

- wheel 包含 Python 代码、资源、内置预训练 v1 `.pth` 文件及校验文件，
  不含其他 `.pth`、`.pt` 训练状态、NIfTI、网格、HDF5、NPY/NPZ、原生库或第三方数据集。
- 源码包包含同样的项目材料和预训练权重。
- 内置权重是仅含模型参数的 state dictionary，不含优化器、调度器、参与者标识或训练恢复状态。
- 容器、Conda 环境、wheelhouse 和独立二进制程序各有自己的依赖清单，
  应检查实际包含的组件及其许可证声明。

<a id="authoritative-upstream-references"></a>

## 权威上游参考

- [SimNIBS 4.6.0 包元数据](https://github.com/simnibs/simnibs/blob/v4.6.0/pyproject.toml)
- [PyTorch 包元数据](https://github.com/pytorch/pytorch/blob/main/pyproject.toml)
- [NumPy 包元数据](https://github.com/numpy/numpy/blob/main/pyproject.toml)
- [NiBabel 仓库与许可证](https://github.com/nipy/nibabel)
- [SciPy 仓库](https://github.com/scipy/scipy)
- [h5py 仓库](https://github.com/h5py/h5py)

发布前，将此清单与实际 wheel、源码包、内置权重及构建/安装环境记录逐项对照。
新增复制或内嵌的第三方材料时，重新检查清单。
