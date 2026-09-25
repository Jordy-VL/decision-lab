#!/usr/bin/env python3
"""Re-evaluate every retained Hugging Face checkpoint on its run's development split."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "modernbert-decisions"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from decisions.cli import predict
from decisions.config import parse
from decisions.data import load
from decisions.metrics import reports
from decisions.model import encode, initialize


def checkpoint_specs(root):
    specs = []
    for checkpoint in sorted(root.glob("runs/*/trainer/checkpoint-*")):
        run = checkpoint.parent.parent
        config_path = run / "config.yaml"
        model_path = checkpoint / "pytorch_model.bin"
        if config_path.exists() and model_path.exists():
            specs.append((run, checkpoint, config_path))
    return specs


def evaluate_one(spec_and_output, device):
    spec, output_root = spec_and_output
    run, checkpoint, config_path = spec
    command, config = parse(["evaluate", "--config", str(config_path)])
    del command
    config.checkpoint = ""
    config.data = config.development_data
    config.split = "development"
    config.device = "cpu" if device == "cpu" else "auto"

    rows = load(config.data, config, labeled=True)
    if any(row["split"] != "development" for row in rows):
        raise ValueError(f"{config.data} contains non-development rows")
    model, tokenizer = initialize(config)
    state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"{checkpoint}: state-dict mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    examples = [encode(row, tokenizer, config.max_length, config.max_options, config.decision_head)
                for row in rows]
    predictions = predict(model, tokenizer, examples, config, device)
    result = reports(predictions)

    output = output_root / run.name / checkpoint.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {
        "run": str(run.relative_to(ROOT)),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "split": "development",
        "rows": len(rows),
        "device": str(device),
        "step": int(checkpoint.name.split("-", 1)[1]),
        "overall": result["overall"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "development-checkpoint-evaluations")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--exclude-prefix", action="append", default=[],
                        help="exclude run names starting with this prefix")
    parser.add_argument("--include-prefix", action="append", default=[],
                        help="when set, include only run names starting with this prefix")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    specs = checkpoint_specs(args.root)
    if args.include_prefix:
        specs = [spec for spec in specs if any(spec[0].name.startswith(prefix)
                                               for prefix in args.include_prefix)]
    specs = [spec for spec in specs if not any(spec[0].name.startswith(prefix)
                                               for prefix in args.exclude_prefix)]
    if not specs:
        raise SystemExit("no retained trainer checkpoints found")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [spec for index, spec in enumerate(specs)
             if index % args.shard_count == args.shard_index]
    if not specs:
        raise SystemExit("selected shard contains no checkpoints")
    work = [(spec, args.output) for spec in specs]

    if args.workers == 1:
        results = [evaluate_one(item, args.device) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate_one, work, ["cuda"] * len(work)))
    results.sort(key=lambda item: (item["run"], item["step"]))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"checkpoints": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
