"""Head2Field CLI for T1 preprocessing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .profiles import T1_PREPROCESSING_PROFILES


def _add_canonical_arguments(parser: argparse.ArgumentParser) -> None:
    """Add public canonicalization command-line arguments."""

    parser.add_argument("--input-t1", type=Path, required=True)
    parser.add_argument("--profile", choices=T1_PREPROCESSING_PROFILES, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--max-samples", type=int, default=60_000)
    parser.add_argument("--minimum-margin-voxels", type=int, default=12)


def build_prepare_parser(
    *,
    prog: str = "head2field-prepare",
) -> argparse.ArgumentParser:
    """Build the complete raw-T1-to-prepared-subject parser."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Orchestrate T1 canonicalization, external SimNIBS/CHARM, conductivity, "
            "the fixed electrode cap, and prepared-subject v1 assembly."
        ),
    )
    _add_canonical_arguments(parser)
    parser.add_argument("--canonical-output-dir", type=Path, required=True)
    parser.add_argument("--head-model-output-dir", type=Path, required=True)
    parser.add_argument("--prepared-output-dir", type=Path, required=True)
    parser.add_argument(
        "--electrode-names-json",
        type=Path,
        help="Optional controlled override; defaults to the standard cap from this CHARM run",
    )
    parser.add_argument(
        "--electrode-coords-mm-npy",
        type=Path,
        help="Optional controlled override; must be supplied together with the names JSON",
    )
    return parser


def _run_prepare(args: argparse.Namespace) -> dict[str, object]:
    """Run complete preparation and return a summary without absolute paths."""

    # Do not import SciPy for --help; load heavy implementations only for preprocessing.
    from .assembly import prepare_subject

    prepare_subject(
        input_t1=args.input_t1,
        profile=args.profile,
        canonical_output_dir=args.canonical_output_dir,
        head_model_output_dir=args.head_model_output_dir,
        prepared_output_dir=args.prepared_output_dir,
        electrode_names_json=args.electrode_names_json,
        electrode_coords_mm_npy=args.electrode_coords_mm_npy,
        template=args.template,
        max_samples=args.max_samples,
        minimum_margin_voxels=args.minimum_margin_voxels,
    )
    return {
        "status": "complete",
        "outputs": {
            "canonical": "caller_selected_new_directory",
            "head_model": "caller_selected_new_directory",
            "prepared_subject": "caller_selected_new_directory",
        },
    }


def prepare_main(argv: Sequence[str] | None = None) -> int:
    """Run the complete public head2field-prepare entry point."""

    args = build_prepare_parser().parse_args(argv)
    print(json.dumps(_run_prepare(args), ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the module parser exposing stage 1 and complete preparation."""

    parser = argparse.ArgumentParser(prog="python -m head2field.preprocessing")
    commands = parser.add_subparsers(dest="command", required=True)

    canonicalize = commands.add_parser(
        "canonicalize",
        help="Run only the explicit T1 canonicalization stage",
    )
    _add_canonical_arguments(canonicalize)
    canonicalize.add_argument("--output-dir", type=Path, required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Generate a complete prepared-subject v1 from a raw T1 image",
    )
    _add_canonical_arguments(prepare)
    prepare.add_argument("--canonical-output-dir", type=Path, required=True)
    prepare.add_argument("--head-model-output-dir", type=Path, required=True)
    prepare.add_argument("--prepared-output-dir", type=Path, required=True)
    prepare.add_argument("--electrode-names-json", type=Path)
    prepare.add_argument("--electrode-coords-mm-npy", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a module subcommand."""

    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(_run_prepare(args), ensure_ascii=False, sort_keys=True))
        return 0
    # Do not import SciPy for --help; load it only when canonicalization runs.
    from .pipeline import preprocess_t1

    manifest = preprocess_t1(
        input_t1=args.input_t1,
        output_dir=args.output_dir,
        profile=args.profile,
        template=args.template,
        max_samples=args.max_samples,
        minimum_margin_voxels=args.minimum_margin_voxels,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["build_parser", "build_prepare_parser", "main", "prepare_main"]
