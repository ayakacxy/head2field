"""读取科学配置时忽略旧记录中的摘要字段。"""

from collections.abc import Mapping
from pathlib import Path


def scientific_metadata(value):
    """保留可读参数、数据顺序和文件信息，去掉旧的摘要记录。"""
    if isinstance(value, Mapping):
        return {
            key: scientific_metadata(item)
            for key, item in value.items()
            if not is_digest_key(key)
        }
    if isinstance(value, (tuple, list)):
        return [scientific_metadata(item) for item in value]
    return value


def is_digest_key(key):
    """识别旧清单中的摘要字段名称。"""
    return any(token in str(key).lower() for token in ("sha256", "checksum", "hash"))


def file_records(paths, *, relative_to):
    """直接保存文件路径、大小与修改时间，避免读取大型数组计算摘要。"""
    root = Path(relative_to)
    records = {}
    for path in sorted(set(Path(item) for item in paths)):
        stat = path.stat()
        records[str(path.relative_to(root))] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return records
