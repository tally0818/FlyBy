"""Run data stages in separate processes so GPU memory is released between them."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ..common.config import load_config
from ..common.run import load_dotenv


def commands(args):
    cfg = load_config(args.config, args.override)
    generation = load_config(args.source_config, args.source_override)
    source = Path(generation["paths"]["source"])
    processed = Path(cfg["paths"]["processed"])
    dataset = [sys.executable, "-m", "src.data.build_dataset", "--config", args.config,
               "--split", args.split]
    for override in args.override:
        dataset.extend(["--override", override])
    stages = [dataset]
    if args.split == "eval":
        return stages
    sources = [sys.executable, "-m", "src.data.build_sources", "--config", args.source_config,
               "--data-config", args.config]
    for override in args.override:
        sources.extend(["--data-override", override])
    for override in args.source_override:
        sources.extend(["--override", override])
    stages += [sources,
               [sys.executable, "-m", "src.data.build_sft", "--source-input", str(source / "sft.parquet"),
                "--output-dir", str(processed), "--tokenizer", generation["model"]["name"],
                "--tokenizer-revision", generation["model"]["revision"]],
               [sys.executable, "-m", "src.data.build_rl", "--rl", str(source / "rl.jsonl"),
                "--sft", str(source / "sft_used.jsonl"),
                "--output", str(processed / "train_rl.jsonl"), "--seed", str(generation["seed"])]]
    stages[-1].extend(["--config", args.source_config])
    for override in args.source_override:
        stages[-1].extend(["--override", override])
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--source-config", default="configs/data_generation.yaml")
    parser.add_argument("--split", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--source-override", action="append", default=[])
    args = parser.parse_args()
    load_dotenv()  # HF_TOKEN and OPENROUTER_API_KEY also reach child processes.
    for command in commands(args):
        print(f"[build] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
