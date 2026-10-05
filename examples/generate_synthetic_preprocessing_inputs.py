#!/usr/bin/env python3
"""Generate deterministic non-anatomical inputs for the preprocessing tutorial."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from head2field.preprocessing.synthetic import (
    write_synthetic_preparation_inputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the example output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for synthetic inputs; must not already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write synthetic T1, orientation-template, and fixed-cap override inputs."""

    args = parse_args(argv)
    result = write_synthetic_preparation_inputs(args.output_dir)
    print(
        json.dumps(
            {
                "purpose": "non_anatomical_preprocessing_example",
                "output_dir": str(result.root),
                "files": [
                    result.raw_t1.name,
                    result.orientation_template.name,
                    result.electrode_names_json.name,
                    result.electrode_coords_mm_npy.name,
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
