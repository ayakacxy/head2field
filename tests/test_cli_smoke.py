"""Argument and help smoke tests for the registered prediction CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from head2field.cli import parse_args


def test_predict_cli_accepts_custom_weights() -> None:
    args = parse_args(
        [
            "--prepared-subject",
            "synthetic-input",
            "--weights",
            "predictor-weights.pth",
            "--output-dir",
            "new-output",
        ]
    )
    assert args.prepared_subject == Path("synthetic-input")
    assert args.weights == Path("predictor-weights.pth")
    assert args.output_dir == Path("new-output")
    assert args.device == "cpu"


def test_predict_cli_defaults_to_bundled_weights() -> None:
    args = parse_args(
        ["--prepared-subject", "synthetic-input", "--output-dir", "new-output"]
    )
    assert args.prepared_subject == Path("synthetic-input")
    assert args.output_dir == Path("new-output")
    assert args.weights is None


def test_module_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "head2field.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--prepared-subject" in completed.stdout
    assert "--weights" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_prepare_help_does_not_import_scipy() -> None:
    """Check that a minimal installation can show preprocessing CLI help without importing SciPy."""

    source_root = Path(__file__).resolve().parents[1] / "src"
    code = r"""
import importlib.abc
import json
import sys

class _BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("测试禁止导入 scipy")
        return None

sys.meta_path.insert(0, _BlockScipy())
from head2field.preprocessing.cli import prepare_main
try:
    prepare_main(["--help"])
except SystemExit as error:
    print(json.dumps({
        "returncode": error.code,
        "scipy_loaded": any(
            name == "scipy" or name.startswith("scipy.") for name in sys.modules
        ),
    }))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [sys.executable, "-P", "-B", "-c", code],
        cwd=source_root.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload == {"returncode": 0, "scipy_loaded": False}
