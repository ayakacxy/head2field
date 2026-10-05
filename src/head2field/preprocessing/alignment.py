"""Numerical core for six-degree-of-freedom rigid T1 canonicalization.

Only rotation and translation are applied; physical head size is preserved.
Outputs use a fixed 1 mm RAS grid. Callers select an explicit profile, publish
transactionally, and handle downstream segmentation.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation

TARGET_SHAPE = (256, 256, 256)
TARGET_AFFINE = np.array(
    [[1, 0, 0, -128], [0, 1, 0, -128], [0, 0, 1, -128], [0, 0, 0, 1]],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_3d(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"只支持三维 T1，实际 shape={data.shape}")
    data[~np.isfinite(data)] = 0
    return data


def robust_foreground(data: np.ndarray) -> np.ndarray:
    """Extract the largest head component from T1 intensities after boundary-noise removal."""

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"T1 必须为三维，实际 shape={data.shape}")
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("T1 中没有有限强度体素。")

    width = max(2, min(data.shape) // 40)
    border = np.zeros(data.shape, dtype=bool)
    border[:width, :, :] = True
    border[-width:, :, :] = True
    border[:, :width, :] = True
    border[:, -width:, :] = True
    border[:, :, :width] = True
    border[:, :, -width:] = True
    border_values = data[border & finite]
    background_median = float(np.median(border_values))
    background_mad = float(np.median(np.abs(border_values - background_median)))
    background_sigma = 1.4826 * background_mad
    upper_intensity = float(np.percentile(data[finite], 99.5))
    contrast = max(upper_intensity - background_median, np.finfo(np.float32).eps)
    threshold = max(
        background_median + 6.0 * background_sigma,
        background_median + 0.03 * contrast,
    )
    if threshold >= upper_intensity:
        raise ValueError("无法从 T1 分离头部前景。")

    candidate = finite & (data > threshold)
    candidate = ndimage.binary_opening(candidate, iterations=1)
    labels, count = ndimage.label(
        candidate,
        structure=ndimage.generate_binary_structure(3, 2),
    )
    if count == 0:
        raise ValueError("T1 阈值化后没有头部连通域。")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    if not mask.any():
        raise ValueError("T1 最大头部连通域为空。")
    return mask


def world_center(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return the center of the world-axis bounding box of all foreground coordinates."""

    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=np.float64)
    if mask.ndim != 3:
        raise ValueError(f"前景掩膜必须为三维，实际 shape={mask.shape}")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine 必须是有限的 4x4 矩阵。")
    # Transform all foreground voxels in z chunks, handling negative axes, permutations, and oblique affines.
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    foreground_count = 0
    for z_start in range(0, mask.shape[2], 16):
        z_stop = min(z_start + 16, mask.shape[2])
        indices = np.argwhere(mask[:, :, z_start:z_stop])
        if indices.size == 0:
            continue
        indices[:, 2] += z_start
        world = nib.affines.apply_affine(affine, indices)
        lower = np.minimum(lower, world.min(axis=0))
        upper = np.maximum(upper, world.max(axis=0))
        foreground_count += len(indices)
    if foreground_count == 0:
        raise ValueError("前景掩膜为空，无法计算世界坐标中心。")
    return 0.5 * (lower + upper)


def validate_optimizer_result(
    result: optimize.OptimizeResult, level_step: int
) -> np.ndarray:
    """Validate a single-level optimizer result and return six finite parameters."""

    if not bool(result.success):
        raise RuntimeError(
            f"刚体优化在 step={level_step} 失败：{getattr(result, 'message', '无错误信息')}"
        )
    parameters = np.asarray(result.x, dtype=np.float64)
    if parameters.shape != (6,) or not np.isfinite(parameters).all():
        raise RuntimeError(
            f"刚体优化在 step={level_step} 返回无效参数：shape={parameters.shape}"
        )
    objective = float(result.fun)
    if not np.isfinite(objective):
        raise RuntimeError(f"刚体优化在 step={level_step} 返回非有限目标函数值。")
    return parameters


def validate_rigid_transform(matrix: np.ndarray, atol: float = 1e-7) -> np.ndarray:
    """Require an orthogonal rotation and translation without scaling or shear."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("刚体变换必须是有限的 4x4 矩阵。")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=atol):
        raise RuntimeError("刚体变换的齐次矩阵末行无效。")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=atol):
        raise RuntimeError("刚体变换含有缩放或剪切，旋转块不正交。")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=0.0, atol=atol):
        raise RuntimeError(f"刚体旋转行列式必须为 +1，实际为 {determinant}。")
    return matrix


def head_mask(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Extract full-head foreground with the canonical pipeline's boundary-noise threshold."""

    return robust_foreground(load_3d(image))


def rigid_matrix(
    parameters: np.ndarray, source_center: np.ndarray, target_center: np.ndarray
) -> np.ndarray:
    rotation = Rotation.from_euler("xyz", parameters[:3], degrees=True).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = target_center + parameters[3:] - rotation @ source_center
    return matrix


def normalized(data: np.ndarray) -> np.ndarray:
    positive = data[data > 0]
    if positive.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    low, high = np.percentile(positive, [1, 99])
    return np.clip((data - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def mutual_information(fixed: np.ndarray, moving: np.ndarray, bins: int = 48) -> float:
    valid = (fixed > 0.01) | (moving > 0.01)
    if valid.sum() < 1000:
        return 0.0
    histogram, _, _ = np.histogram2d(
        fixed[valid], moving[valid], bins=bins, range=((0, 1), (0, 1))
    )
    probability = histogram / max(histogram.sum(), 1.0)
    px = probability.sum(axis=1, keepdims=True)
    py = probability.sum(axis=0, keepdims=True)
    expected = px @ py
    nz = probability > 0
    return float(np.sum(probability[nz] * np.log(probability[nz] / expected[nz])))


def registration_samples(
    template: np.ndarray,
    affine: np.ndarray,
    step: int,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic sparse samples covering head foreground and nearby background."""

    grid = np.mgrid[
        0 : template.shape[0] : step,
        0 : template.shape[1] : step,
        0 : template.shape[2] : step,
    ]
    ijk = grid.reshape(3, -1).T
    values = template[tuple(ijk.T)]
    if max_samples <= 0:
        raise ValueError("max_samples 必须为正数。")
    if len(ijk) > max_samples:
        foreground = np.flatnonzero(values > 0.01)
        background = np.flatnonzero(values <= 0.01)
        rng = np.random.default_rng(42 + step)
        foreground_budget = min(len(foreground), int(max_samples * 0.8))
        background_budget = min(len(background), max_samples - foreground_budget)
        selected = np.concatenate(
            [
                rng.choice(foreground, foreground_budget, replace=False),
                rng.choice(background, background_budget, replace=False),
            ]
        )
        selected.sort()
        ijk = ijk[selected]
        values = values[selected]
    world = nib.affines.apply_affine(affine, ijk)
    return world, values


def estimate_rigid(
    source_image,
    source,
    template_image,
    template,
    max_samples: int,
) -> tuple[np.ndarray, optimize.OptimizeResult, np.ndarray]:
    source_mask = head_mask(source_image)
    template_mask = head_mask(template_image)
    source_center = world_center(source_mask, source_image.affine)
    target_center = world_center(template_mask, template_image.affine)
    source_norm = normalized(source)
    template_norm = normalized(template)
    inv_source_affine = np.linalg.inv(source_image.affine)

    parameters = np.zeros(6, dtype=np.float64)
    result: optimize.OptimizeResult | None = None
    for step, maxiter in ((4, 80), (2, 60)):
        target_world, fixed = registration_samples(
            template_norm,
            template_image.affine,
            step,
            max_samples,
        )

        def objective(
            candidate: np.ndarray,
            target_world: np.ndarray = target_world,
            fixed: np.ndarray = fixed,
        ) -> float:
            target_from_source = rigid_matrix(candidate, source_center, target_center)
            source_world = nib.affines.apply_affine(
                np.linalg.inv(target_from_source), target_world
            )
            source_ijk = nib.affines.apply_affine(inv_source_affine, source_world)
            moving = ndimage.map_coordinates(
                source_norm, source_ijk.T, order=1, mode="constant", cval=0
            )
            return -mutual_information(fixed, moving)

        result = optimize.minimize(
            objective,
            parameters,
            method="Powell",
            bounds=[(-30, 30), (-30, 30), (-30, 30), (-35, 35), (-35, 35), (-35, 35)],
            options={"maxiter": maxiter, "xtol": 0.03, "ftol": 1e-5},
        )
        parameters = validate_optimizer_result(result, step)
    assert result is not None
    template_aligned = rigid_matrix(parameters, source_center, target_center)
    rotation = template_aligned[:3, :3]
    canonical_center = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
    centered = np.eye(4, dtype=np.float64)
    centered[:3, :3] = rotation
    centered[:3, 3] = canonical_center - rotation @ source_center
    return validate_rigid_transform(centered), result, source_center


def resample_to_canonical(
    source_image, source: np.ndarray, target_from_source: np.ndarray
) -> np.ndarray:
    target_from_source = validate_rigid_transform(target_from_source)
    source_from_target_voxel = (
        np.linalg.inv(source_image.affine)
        @ np.linalg.inv(target_from_source)
        @ TARGET_AFFINE
    )
    sampled = ndimage.affine_transform(
        source,
        matrix=source_from_target_voxel[:3, :3],
        offset=source_from_target_voxel[:3, 3],
        output_shape=TARGET_SHAPE,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.float32)
    if not np.isfinite(sampled).all():
        raise RuntimeError("刚体重采样输出含有非有限值。")
    return sampled


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Encode a PNG chunk with CRC."""

    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _write_grayscale_png(path: Path, pixels: np.ndarray, title: str) -> None:
    """Write lossless 8-bit grayscale PNG using only the standard library, avoiding plot caches."""

    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim != 2 or min(pixels.shape) <= 0:
        raise ValueError(f"QC 像素必须为非空二维数组，实际 shape={pixels.shape}")
    height, width = pixels.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    title_payload = b"Title\x00\x00\x00\x00\x00" + title.encode(
        "utf-8",
        errors="replace",
    )
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"iTXt", title_payload)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(encoded)


def save_qc(path: Path, data: np.ndarray, title: str) -> None:
    """Generate sagittal, coronal, and axial grayscale QC views."""

    mask = robust_foreground(data)
    center = np.round(np.argwhere(mask).mean(axis=0)).astype(int)
    views = [data[center[0], :, :].T, data[:, center[1], :].T, data[:, :, center[2]].T]
    foreground_values = data[mask]
    low, high = np.percentile(foreground_values, [1, 99])
    scale = max(float(high - low), np.finfo(np.float32).eps)
    normalized_views = [
        np.flipud(np.clip((view - low) / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
        for view in views
    ]
    target_height = max(view.shape[0] for view in normalized_views)
    padded_views: list[np.ndarray] = []
    for view in normalized_views:
        before = (target_height - view.shape[0]) // 2
        after = target_height - view.shape[0] - before
        padded_views.append(np.pad(view, ((before, after), (0, 0)), constant_values=0))
    separator = np.full((target_height, 4), 255, dtype=np.uint8)
    montage = np.concatenate(
        [padded_views[0], separator, padded_views[1], separator, padded_views[2]],
        axis=1,
    )
    _write_grayscale_png(path, montage, title)
