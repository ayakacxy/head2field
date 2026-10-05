"""Non-overwriting transaction for preprocessing output directories."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import TracebackType


def _path_entry_exists(path: Path) -> bool:
    """Recognize ordinary paths and broken symlinks."""

    return os.path.lexists(os.fspath(path))


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish directories atomically without replacement where supported."""

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("当前 Linux C 运行时不提供 renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"输出目录已存在，拒绝覆盖：{destination}")
        raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise RuntimeError("当前平台不支持安全的 no-replace 目录发布")


class StagedOutputDirectory:
    """Publish only the complete sibling staging directory created by this invocation."""

    def __init__(self, final_dir: Path) -> None:
        self.final_dir = Path(final_dir)
        self.staging_dir: Path | None = None

    def __enter__(self) -> Path:
        """Reject an existing destination and create unique staging."""

        if not self.final_dir.name:
            raise ValueError("最终输出目录不能是文件系统根目录")
        if _path_entry_exists(self.final_dir):
            raise FileExistsError(f"最终输出目录已存在，拒绝覆盖：{self.final_dir}")
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.final_dir.name}.staging-",
                dir=self.final_dir.parent,
            )
        )
        return self.staging_dir

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Clean only this staging directory on failure; promote without replacement on success."""

        staging = self.staging_dir
        if staging is None:
            return False
        if exc_type is not None:
            shutil.rmtree(staging, ignore_errors=True)
            self.staging_dir = None
            return False
        try:
            _rename_noreplace(staging, self.final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            self.staging_dir = None
            raise
        self.staging_dir = None
        return False


def staged_output_directory(final_dir: Path) -> StagedOutputDirectory:
    """Return a preprocessing output transaction that refuses existing destinations."""

    return StagedOutputDirectory(final_dir)


__all__ = ["StagedOutputDirectory", "staged_output_directory"]
