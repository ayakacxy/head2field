"""Package CLI for dry-run planning and actual FEM v1 label generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .generation import (
    dry_run_fem_generation,
    export_fem_labels_from_hdf5,
    generate_fem_labels,
    validate_fem_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build a parser that distinguishes side-effect-free planning from generation."""

    parser = argparse.ArgumentParser(prog="head2field-fem")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan", help="Print the complete TDCS-to-label plan without writing files"
    )
    plan.add_argument("--prepared-subject-dir", type=Path, required=True)
    plan.add_argument("--head-mesh", type=Path, required=True)
    plan.add_argument("--cpus", type=int, default=1)

    generate = commands.add_parser(
        "generate", help="Generate HDF5 output and final labels from prepared inputs"
    )
    generate.add_argument("--prepared-subject-dir", type=Path, required=True)
    generate.add_argument("--head-mesh", type=Path, required=True)
    generate.add_argument("--leadfield-output-dir", type=Path, required=True)
    generate.add_argument("--labels-output-dir", type=Path, required=True)
    generate.add_argument("--cpus", type=int, default=1)

    export = commands.add_parser(
        "export", help="Export labels transactionally from an existing final HDF5 file"
    )
    export.add_argument("--prepared-subject-dir", type=Path, required=True)
    export.add_argument("--leadfield-hdf5", type=Path, required=True)
    export.add_argument("--solver-log", type=Path, required=True)
    export.add_argument("--leadfield-run-manifest", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)

    validate = commands.add_parser(
        "validate",
        help="Validate final labels and optional TDCS provenance without writing files",
    )
    validate.add_argument("--prepared-subject-dir", type=Path, required=True)
    validate.add_argument("--fem-labels-dir", type=Path)
    validate.add_argument("--leadfield-hdf5", type=Path)
    validate.add_argument("--solver-log", type=Path)
    validate.add_argument("--leadfield-run-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a dry run or transactional label generation."""

    args = build_parser().parse_args(argv)
    if args.command == "plan":
        plan = dry_run_fem_generation(
            prepared_subject_dir=args.prepared_subject_dir,
            head_mesh=args.head_mesh,
            cpus=args.cpus,
        )
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "generate":
        output = generate_fem_labels(
            prepared_subject_dir=args.prepared_subject_dir,
            head_mesh=args.head_mesh,
            leadfield_output_dir=args.leadfield_output_dir,
            labels_output_dir=args.labels_output_dir,
            cpus=args.cpus,
        )
    elif args.command == "export":
        output = export_fem_labels_from_hdf5(
            prepared_subject_dir=args.prepared_subject_dir,
            leadfield_hdf5=args.leadfield_hdf5,
            solver_log=args.solver_log,
            leadfield_run_manifest=args.leadfield_run_manifest,
            output_dir=args.output_dir,
        )
    else:
        result = validate_fem_artifacts(
            prepared_subject_dir=args.prepared_subject_dir,
            fem_labels_dir=args.fem_labels_dir,
            leadfield_hdf5=args.leadfield_hdf5,
            solver_log=args.solver_log,
            leadfield_run_manifest=args.leadfield_run_manifest,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    print(output)
    return 0


__all__ = ["build_parser", "main"]
