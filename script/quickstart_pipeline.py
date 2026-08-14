#!/usr/bin/env python3

import argparse
import copy
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DistMult pretraining followed by MAPLE reinforcement training."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/quickstart_train.yaml",
        help="Training configuration relative to the repository root.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Physical GPU index. The selected GPU is exposed as logical GPU 0 to both stages.",
    )
    parser.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Discard an existing DistMult checkpoint and pretrain it again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the commands without running them.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping: %s" % path)
    return config


def validate_config(config, config_path):
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("path"):
        raise ValueError("Configuration is missing dataset.path: %s" % config_path)
    if not config.get("pretrain_gen_model"):
        raise ValueError("Configuration is missing pretrain_gen_model: %s" % config_path)
    if not isinstance(config.get("engine"), dict):
        raise ValueError("Configuration is missing engine settings: %s" % config_path)

    dataset_path = resolve_project_path(dataset["path"])
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset directory does not exist: %s" % dataset_path)
    return dataset_path, resolve_project_path(config["pretrain_gen_model"])


def create_runtime_config(config, gpu):
    environment = os.environ.copy()
    if gpu is None:
        return None, environment

    runtime_config = copy.deepcopy(config)
    runtime_config["engine"]["gpus"] = [0]
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="maple_pipeline_",
        delete=False,
    ) as stream:
        yaml.safe_dump(runtime_config, stream, sort_keys=False)
        return Path(stream.name), environment


def run_command(command, environment, dry_run):
    print("[pipeline] %s" % shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main():
    args = parse_args()
    config_path = resolve_project_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError("Configuration file does not exist: %s" % config_path)

    config = load_config(config_path)
    dataset_path, checkpoint_path = validate_config(config, config_path)
    runtime_config_path, environment = create_runtime_config(config, args.gpu)
    active_config_path = runtime_config_path or config_path
    checkpoint_exists = checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0

    try:
        print("[pipeline] dataset: %s" % dataset_path, flush=True)
        print("[pipeline] DistMult checkpoint: %s" % checkpoint_path, flush=True)

        if checkpoint_exists and not args.force_pretrain:
            print("[pipeline] existing DistMult checkpoint found; skip pretraining", flush=True)
        else:
            if args.force_pretrain and checkpoint_exists:
                print("[pipeline] removing existing DistMult checkpoint before forced pretraining", flush=True)
                if not args.dry_run:
                    checkpoint_path.unlink()
            if not args.dry_run:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                [sys.executable, "pretrain.py", "-c", str(active_config_path)],
                environment,
                args.dry_run,
            )
            if not args.dry_run and not checkpoint_path.is_file():
                raise RuntimeError("DistMult pretraining did not create: %s" % checkpoint_path)

        run_command(
            [sys.executable, "script/train.py", "-c", str(active_config_path)],
            environment,
            args.dry_run,
        )
    finally:
        if runtime_config_path is not None and runtime_config_path.exists():
            runtime_config_path.unlink()


if __name__ == "__main__":
    main()
