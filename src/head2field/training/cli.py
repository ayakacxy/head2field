"""Head2Field CLIs for training, preflight, caches, physics, and synthetic fixtures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from head2field._spec import create_head2field_v1
from head2field.weights import read_pure_v1_state_dict

from .cache import prepare_loss_statistics, prepare_training_cache
from .contracts import (
    TrainingConfig,
    build_run_contract,
    load_training_dataset_manifest,
)
from .dataset import DeterministicHead2FieldSampler, Head2FieldTrainingDataset
from .physics_data import (
    finalize_physics_pool,
    fit_physics_statistics,
    prepare_physics_raw_pool,
)
from .runner import LoopConfig, run_training_loop
from .steps import build_n128_gp_step_function
from .synthetic import (
    SyntheticHead2FieldDataset,
    create_synthetic_smoke_model,
    load_synthetic_training_dataset,
    write_synthetic_training_dataset,
)
from .validation import build_deterministic_validation_callback


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _initialization_arguments(parser: argparse.ArgumentParser) -> None:
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-weights", type=Path)
    initialization.add_argument("--resume-state", type=Path)
    parser.add_argument(
        "--trust-local-resume-state",
        action="store_true",
        help="Use only with a trusted local pickle resume state that you created yourself.",
    )


def _synthetic_run_contract(
    config: TrainingConfig, manifest_path: Path
) -> dict[str, object]:
    """保存合成训练的完整配置、分组和资产路径。"""
    from head2field.metadata import scientific_metadata

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "schema_id": "synthetic_training_run_contract_v3",
        "configuration": config.to_dict(),
        "dataset": scientific_metadata(payload),
        "data_loss_epoch_scale_rule": "steps_per_epoch_divide_75_training_subjects",
        "execution_profile_id": config.execution_profile_id,
    }


def train_main(argv: Sequence[str] | None = None) -> None:
    """Run CPU smoke tests or real CUDA FP16 training with epoch-based scheduling."""

    parser = argparse.ArgumentParser(description=__doc__)
    _config_argument(parser)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--synthetic-dataset", type=Path)
    data.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-after-steps", type=int)
    _initialization_arguments(parser)
    args = parser.parse_args(argv)
    config = TrainingConfig.load(args.config)
    step = build_n128_gp_step_function(config.data_loss, config.physics_loss)
    test_callback = None

    if args.synthetic_dataset is not None:
        if config.execution_profile_id != "cpu_fp32_synthetic_smoke":
            raise ValueError("合成小模型须使用 training_smoke_v1.json")
        if args.init_weights is not None:
            raise ValueError("test-only 小模型不接受官方 N128 --init-weights")
        subjects = load_synthetic_training_dataset(args.synthetic_dataset)
        dataset = SyntheticHead2FieldDataset(subjects, partition="train")
        validation_dataset = SyntheticHead2FieldDataset(
            subjects, partition="validation"
        )
        sampler = DeterministicHead2FieldSampler(
            dataset.subject_sizes,
            query_count=config.query_count,
            sources_per_step=config.sources_per_step,
            seed=config.seed,
        )
        model = create_synthetic_smoke_model()
        dataset.schedule_loss_scale = sampler.data_loss_scale
        validation_dataset.schedule_loss_scale = sampler.data_loss_scale
        loop = LoopConfig.from_training_config(
            config, steps_per_epoch=sampler.steps_per_epoch
        )
        contract = _synthetic_run_contract(config, args.synthetic_dataset.resolve())
        validation_callback = build_deterministic_validation_callback(
            dataset=validation_dataset,
            subject_sizes=validation_dataset.subject_sizes,
            data_config=config.data_loss,
            physics_config=config.physics_loss,
            query_count=config.query_count,
            sources_per_subject=config.sources_per_step,
        )
    else:
        if config.execution_profile_id != "cuda_amp_fp16_epoch_training":
            raise ValueError("真实训练须使用 CUDA FP16 epoch 配置")
        manifest = load_training_dataset_manifest(args.dataset_manifest)
        dataset = Head2FieldTrainingDataset(
            manifest,
            partition="train",
            encoder_input_shape=config.encoder_input_shape,
        )
        validation_dataset = Head2FieldTrainingDataset(
            manifest,
            partition="validation",
            encoder_input_shape=config.encoder_input_shape,
        )
        sampler = DeterministicHead2FieldSampler(
            dataset.subject_sizes,
            query_count=config.query_count,
            sources_per_step=config.sources_per_step,
            seed=config.seed,
        )
        torch.manual_seed(config.seed)
        dataset.schedule_loss_scale = sampler.data_loss_scale
        validation_dataset.schedule_loss_scale = sampler.data_loss_scale
        model = create_head2field_v1()
        if args.init_weights is not None:
            # Enforce the official bare-weight 208-tensor/FP32/value-count contract before the runner.
            read_pure_v1_state_dict(args.init_weights)
        loop = LoopConfig.from_training_config(
            config, steps_per_epoch=sampler.steps_per_epoch
        )
        contract = build_run_contract(config, manifest)
        contract = {
            **contract,
            "execution_profile_id": config.execution_profile_id,
            "official_weight_reproduction": False,
        }

        validation_sizes = validation_dataset.subject_sizes
        validation_callback = build_deterministic_validation_callback(
            dataset=validation_dataset,
            subject_sizes=validation_sizes,
            data_config=config.data_loss,
            physics_config=config.physics_loss,
            query_count=config.query_count,
            sources_per_subject=config.sources_per_step,
        )
        if manifest.subjects_for_partition("test"):
            test_dataset = Head2FieldTrainingDataset(
                manifest,
                partition="test",
                encoder_input_shape=config.encoder_input_shape,
            )
            test_dataset.schedule_loss_scale = sampler.data_loss_scale
            test_callback = build_deterministic_validation_callback(
                dataset=test_dataset,
                subject_sizes=test_dataset.subject_sizes,
                data_config=config.data_loss,
                physics_config=config.physics_loss,
                query_count=config.query_count,
                sources_per_subject=config.sources_per_step,
            )

    result = run_training_loop(
        model=model,
        sampler=sampler,
        batch_provider=dataset.batch,
        step_function=step,
        loop_config=loop,
        run_contract=contract,
        output_dir=args.output_dir,
        init_weights=args.init_weights,
        resume_state=args.resume_state,
        trusted_local_resume_state=args.trust_local_resume_state,
        stop_after_steps=args.stop_after_steps,
        validation_callback=validation_callback,
        test_callback=test_callback,
    )
    print(
        json.dumps(
            {
                "global_step": result.global_step,
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "best_epoch": result.best_epoch,
                "stopped_early": result.stopped_early,
                "test_metrics_path": str(result.test_metrics_path)
                if result.test_metrics_path
                else None,
                "state_path": str(result.state_path),
                "log_path": str(result.log_path),
                "official_weight_reproduction": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def preflight_main(argv: Sequence[str] | None = None) -> None:
    """Validate config, split, all hashes, and execution profile read-only without training."""

    parser = argparse.ArgumentParser(description=__doc__)
    _config_argument(parser)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--synthetic-dataset", type=Path)
    data.add_argument("--dataset-manifest", type=Path)
    args = parser.parse_args(argv)
    config = TrainingConfig.load(args.config)
    if args.synthetic_dataset is not None:
        if config.execution_profile_id != "cpu_fp32_synthetic_smoke":
            raise ValueError("合成小模型须使用 training_smoke_v1.json")
        subjects = load_synthetic_training_dataset(args.synthetic_dataset)
        counts = {
            partition: sum(subject.partition == partition for subject in subjects)
            for partition in ("train", "validation")
        }
        profile = config.execution_profile_id
    else:
        if config.execution_profile_id != "cuda_amp_fp16_epoch_training":
            raise ValueError("真实训练须使用 CUDA FP16 epoch 配置")
        manifest = load_training_dataset_manifest(args.dataset_manifest)
        train = Head2FieldTrainingDataset(
            manifest, partition="train", encoder_input_shape=(128, 128, 128)
        )
        validation = Head2FieldTrainingDataset(
            manifest, partition="validation", encoder_input_shape=(128, 128, 128)
        )
        # Accessing sizes triggers complete loading, hashing, and cross-checks for all four assets.
        train_subject_sizes = train.subject_sizes
        validation_subject_sizes = validation.subject_sizes
        counts = {
            "train": len(train_subject_sizes),
            "validation": len(validation_subject_sizes),
        }
        if manifest.subjects_for_partition("test"):
            test = Head2FieldTrainingDataset(
                manifest,
                partition="test",
                encoder_input_shape=config.encoder_input_shape,
            )
            counts["test"] = len(test.subject_sizes)
        profile = config.execution_profile_id
    print(
        json.dumps(
            {
                "status": "PASS",
                "execution_profile_id": profile,
                "subjects": counts,
                "official_weight_reproduction": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def prepare_cache_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=prepare_training_cache.__doc__)
    parser.add_argument("--prepared-subject", type=Path, required=True)
    parser.add_argument("--fem-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        prepare_training_cache(
            prepared_subject_dir=args.prepared_subject,
            fem_labels_dir=args.fem_labels,
            output_dir=args.output_dir,
        )
    )


def prepare_statistics_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=prepare_loss_statistics.__doc__)
    parser.add_argument("--prepared-subject", type=Path, required=True)
    parser.add_argument("--fem-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    args = parser.parse_args(argv)
    print(
        prepare_loss_statistics(
            prepared_subject_dir=args.prepared_subject,
            fem_labels_dir=args.fem_labels,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
        )
    )


def prepare_physics_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("raw")
    raw.add_argument("--prepared-subject", type=Path, required=True)
    raw.add_argument("--fem-labels", type=Path, required=True)
    raw.add_argument("--output-dir", type=Path, required=True)
    raw.add_argument("--seed", type=int, default=1042)
    fit = commands.add_parser("fit")
    fit.add_argument("--train-raw-pool", type=Path, action="append", required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--raw-pool", type=Path, required=True)
    finalize.add_argument("--statistics", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "raw":
        result = prepare_physics_raw_pool(
            prepared_subject_dir=args.prepared_subject,
            fem_labels_dir=args.fem_labels,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.command == "fit":
        result = fit_physics_statistics(
            train_raw_pool_manifests=args.train_raw_pool,
            output_dir=args.output_dir,
        )
    else:
        result = finalize_physics_pool(
            raw_pool_manifest=args.raw_pool,
            statistics_manifest=args.statistics,
            output_dir=args.output_dir,
        )
    print(result)


def prepare_synthetic_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=write_synthetic_training_dataset.__doc__
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(write_synthetic_training_dataset(args.output_dir, seed=args.seed))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("train", "preflight", "cache", "statistics", "physics", "synthetic"),
    )
    args, remainder = parser.parse_known_args(argv)
    functions = {
        "train": train_main,
        "preflight": preflight_main,
        "cache": prepare_cache_main,
        "statistics": prepare_statistics_main,
        "physics": prepare_physics_main,
        "synthetic": prepare_synthetic_main,
    }
    functions[args.command](remainder)


__all__ = [
    "main",
    "preflight_main",
    "prepare_cache_main",
    "prepare_physics_main",
    "prepare_statistics_main",
    "prepare_synthetic_main",
    "train_main",
]
