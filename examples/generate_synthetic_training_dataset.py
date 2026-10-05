#!/usr/bin/env python3
"""Generate a deterministic two-fictional-subject CPU training smoke dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from head2field.training.synthetic import write_synthetic_training_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse synthetic dataset output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for the synthetic training dataset; must not already exist",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate synthetic data transactionally and print its manifest path."""

    args = parse_args(argv)
    manifest = write_synthetic_training_dataset(args.output_dir, seed=args.seed)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
