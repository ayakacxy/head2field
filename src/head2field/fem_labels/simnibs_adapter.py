"""Isolate the external SimNIBS dependency within the FEM label module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .contracts import (
    PRODUCTION_PROFILE,
    FEMV1ContractError,
    FEMV1ProductionProfile,
    validate_leadfield_dataset_contract,
)

LEADFIELD_DATASET = "mesh_leadfield/leadfields/tdcs_leadfield"
MESH_TAG_DATASET = "mesh_leadfield/elm/tag1"
MESH_TAG2_DATASET = "mesh_leadfield/elm/tag2"
MESH_NODE_DATASET = "mesh_leadfield/nodes/node_coord"
MESH_CONNECTIVITY_DATASET = "mesh_leadfield/elm/node_number_list"


class SimNIBSAdapterError(RuntimeError):
    """Indicate that the external SimNIBS runtime is unavailable or unreadable."""


@dataclass(frozen=True)
class SimNIBSRuntime:
    """Store lazily imported SimNIBS modules and their version."""

    mesh_io: Any
    cython_msh: Any
    version: str


def installed_simnibs_version() -> str:
    """Return the installed external SimNIBS version, failing explicitly if unknown."""

    try:
        return metadata.version("simnibs")
    except metadata.PackageNotFoundError:
        try:
            import simnibs
        except ImportError as exc:
            raise SimNIBSAdapterError(
                "FEM 标签生成需要用户在 fem 环境中安装外部 SimNIBS。"
            ) from exc
        version = str(getattr(simnibs, "__version__", "")).strip()
        if not version:
            raise SimNIBSAdapterError("无法确定外部 SimNIBS 的安装版本。") from None
        return version


@lru_cache(maxsize=1)
def load_simnibs_runtime() -> SimNIBSRuntime:
    """Lazily load the external SimNIBS modules required for mesh interpolation."""

    try:
        from simnibs import mesh_io
        from simnibs.mesh_tools import cython_msh
    except ImportError as exc:
        raise SimNIBSAdapterError(
            "FEM 标签生成需要用户在 fem 环境中安装外部 SimNIBS。"
        ) from exc
    return SimNIBSRuntime(
        mesh_io=mesh_io,
        cython_msh=cython_msh,
        version=installed_simnibs_version(),
    )


def validate_leadfield_hdf5(
    hdf5_path: Path,
    *,
    profile: FEMV1ProductionProfile = PRODUCTION_PROFILE,
) -> dict[str, Any]:
    """Read HDF5 and validate the official v1 lead-field dataset contract."""

    if not hdf5_path.is_file():
        raise FileNotFoundError(f"leadfield HDF5 不存在：{hdf5_path}")
    try:
        import h5py
    except ImportError as exc:
        raise SimNIBSAdapterError("读取 leadfield HDF5 需要 h5py。") from exc

    with h5py.File(hdf5_path, "r") as handle:
        if LEADFIELD_DATASET not in handle:
            raise FEMV1ContractError(f"HDF5 缺少 dataset：{LEADFIELD_DATASET}")
        dataset = handle[LEADFIELD_DATASET]
        attrs = {name: dataset.attrs[name] for name in dataset.attrs}
        shape = tuple(int(value) for value in dataset.shape)
        if MESH_TAG_DATASET not in handle:
            raise FEMV1ContractError(f"HDF5 缺少 mesh dataset：{MESH_TAG_DATASET}")
        mesh_element_count = int(handle[MESH_TAG_DATASET].size)
    if shape[1] != mesh_element_count:
        raise FEMV1ContractError(
            "leadfield element 轴与 mesh 单元数不一致："
            f"{shape[1]} != {mesh_element_count}"
        )
    result = validate_leadfield_dataset_contract(
        shape=shape,
        attrs=attrs,
        profile=profile,
    )
    result["dataset_path"] = LEADFIELD_DATASET
    result["mesh_element_count"] = mesh_element_count
    return result


def build_base_mesh(hdf5_path: Path) -> Any:
    """Reconstruct a SimNIBS tetrahedral mesh from contract-validated HDF5."""

    validate_leadfield_hdf5(hdf5_path)
    runtime = load_simnibs_runtime()
    try:
        import h5py
    except ImportError as exc:
        raise SimNIBSAdapterError("读取 leadfield HDF5 需要 h5py。") from exc

    with h5py.File(hdf5_path, "r") as handle:
        required = (MESH_NODE_DATASET, MESH_CONNECTIVITY_DATASET, MESH_TAG_DATASET)
        missing = [name for name in required if name not in handle]
        if missing:
            raise FEMV1ContractError(f"HDF5 缺少 mesh dataset：{missing}")
        nodes = np.asarray(handle[MESH_NODE_DATASET], dtype=np.float64)
        connectivity = np.asarray(
            handle[MESH_CONNECTIVITY_DATASET],
            dtype=np.int32,
        )
        tag1 = np.asarray(handle[MESH_TAG_DATASET], dtype=np.int32).reshape(-1)
        tag2 = (
            np.asarray(handle[MESH_TAG2_DATASET], dtype=np.int32).reshape(-1)
            if MESH_TAG2_DATASET in handle
            else tag1.copy()
        )
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
        raise FEMV1ContractError("mesh node_coord 必须是有限的 [N, 3] 数组")
    if connectivity.ndim != 2 or connectivity.shape[0] != len(tag1):
        raise FEMV1ContractError("mesh connectivity 与 element tag 数量不一致")

    mesh = runtime.mesh_io.Msh(
        nodes=runtime.mesh_io.Nodes(nodes),
        elements=runtime.mesh_io.Elements(tetrahedra=connectivity),
    )
    mesh.elm.tag1 = tag1
    mesh.elm.tag2 = tag2
    return mesh


def read_source_element_field(
    hdf5_path: Path,
    *,
    source_index: int,
    scale: float = PRODUCTION_PROFILE.scale,
) -> np.ndarray:
    """Read an element-wise source-to-Oz E field and scale it to 1 mA."""

    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise TypeError("source_index 必须是整数")
    if source_index < 0 or source_index >= PRODUCTION_PROFILE.source_count:
        raise IndexError("source_index 超出正式 75-source 合同")
    if not np.isclose(float(scale), PRODUCTION_PROFILE.scale, rtol=0.0, atol=0.0):
        raise FEMV1ContractError(f"正式场缩放必须为 {PRODUCTION_PROFILE.scale!r}")
    try:
        import h5py
    except ImportError as exc:
        raise SimNIBSAdapterError("读取 leadfield HDF5 需要 h5py。") from exc
    with h5py.File(hdf5_path, "r") as handle:
        dataset = handle[LEADFIELD_DATASET]
        field = np.asarray(dataset[source_index, :, :], dtype=np.float32)
    if field.ndim != 2 or field.shape[1] != 3 or not np.isfinite(field).all():
        raise FEMV1ContractError("source element field 必须是有限的 [N, 3] 数组")
    return np.asarray(field * np.float32(scale), dtype=np.float32)


def _vector_image_data(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Normalize a SimNIBS vector NIfTI to [X, Y, Z, 3]."""

    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim == 5 and data.shape[3:] == (1, 3):
        data = data[..., 0, :]
    elif data.ndim == 4 and data.shape[-1] == 3:
        pass
    else:
        squeezed = np.squeeze(data)
        if squeezed.ndim != 4 or squeezed.shape[-1] != 3:
            raise FEMV1ContractError(
                f"SimNIBS 插值输出必须包含三个向量分量，实际 shape={data.shape}"
            )
        data = squeezed
    if not np.isfinite(data).all():
        raise FEMV1ContractError("SimNIBS 插值输出含 NaN 或 Inf")
    return np.asarray(data, dtype=np.float32)


def interpolate_element_field_at_queries(
    *,
    base_mesh: Any,
    element_field: np.ndarray,
    reference_nifti: Path,
    query_ijk: np.ndarray,
    keep_tissues: Sequence[int] = PRODUCTION_PROFILE.keep_tissues,
) -> np.ndarray:
    """Interpolate with external SimNIBS on a linear voxel grid in query order."""

    queries = np.asarray(query_ijk)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise FEMV1ContractError("query_ijk 必须为 [Q, 3]")
    if not np.issubdtype(queries.dtype, np.integer) and (
        not np.isfinite(queries).all()
        or not np.array_equal(
            queries,
            np.trunc(queries),
        )
    ):
        raise FEMV1ContractError("query_ijk 必须包含有限整数")
    queries = np.asarray(queries, dtype=np.int64)
    reference = nib.load(str(reference_nifti))
    shape = tuple(int(value) for value in reference.shape[:3])
    if (
        len(shape) != 3
        or np.any(queries < 0)
        or np.any(queries >= np.asarray(shape, dtype=np.int64))
    ):
        raise FEMV1ContractError("query_ijk 超出 reference NIfTI 网格")

    field = np.asarray(element_field, dtype=np.float32)
    if field.ndim != 2 or field.shape[1] != 3 or not np.isfinite(field).all():
        raise FEMV1ContractError("element_field 必须是有限的 [N, 3]")
    base_mesh.elmdata = []
    base_mesh.nodedata = []
    base_mesh.add_element_field(field, "E")
    try:
        available_tags = set(np.unique(base_mesh.elm.tag1).tolist())
        requested_tags = {int(value) for value in keep_tissues}
        mesh = (
            base_mesh
            if requested_tags == available_tags
            else base_mesh.crop_mesh(tags=sorted(requested_tags))
        )
        element_data = mesh.elmdata[0]
        image = element_data.to_nifti(
            shape,
            reference.affine,
            fn=None,
            qform=reference.header.get_qform(),
            method=PRODUCTION_PROFILE.interpolation_method,
        )
        volume = _vector_image_data(image)
        labels = volume[queries[:, 0], queries[:, 1], queries[:, 2], :]
        return np.asarray(labels, dtype=np.float32)
    finally:
        base_mesh.elmdata = []
        base_mesh.nodedata = []


__all__ = [
    "LEADFIELD_DATASET",
    "SimNIBSAdapterError",
    "SimNIBSRuntime",
    "build_base_mesh",
    "installed_simnibs_version",
    "interpolate_element_field_at_queries",
    "load_simnibs_runtime",
    "read_source_element_field",
    "validate_leadfield_hdf5",
]
