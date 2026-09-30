#!/usr/bin/env python3
"""Bounded train-only gradient/overfit probe; does not produce a research checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/modernbert-decisions"))

import torch
from decisions.config import Config
from decisions.data import load
from decisions.model import CANDIDATE_MARKER, collate, encode, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/kev-decision-v7/train.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--examples", type=int, default=8, choices=(8, 16))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--checkpoint", default="", help="Optional project-format candidate checkpoint; default is pinned pretrained encoder")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or args.learning_rate <= 0:
        parser.error("steps must be 1..100 and learning rate positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required for the pretrained diagnostic")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = Config(data=args.data, decision_head="candidate_masks", checkpoint=args.checkpoint,
                    revision="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13", max_length=args.max_length)
    rows = load(args.data, config)
    if any(r.get("split") != "train" for r in rows):
        raise ValueError("diagnostic requires an explicitly train-only input file")
    random.Random(args.seed).shuffle(rows)
    model, tokenizer = initialize(config)
    if model.decision_head != "candidate_masks":
        raise ValueError("checkpoint must use candidate_masks")
    examples, exclusions = [], []
    # Round-robin sources so eight examples are not one task's contiguous block.
    sources = {}
    for row in rows:
        sources.setdefault(row["source"], []).append(row)
    while sources and len(examples) < args.examples:
        for source in list(sources):
            row = sources[source].pop()
            if not sources[source]:
                del sources[source]
            try:
                examples.append(encode(row, tokenizer, config.max_length, config.max_options, config.decision_head))
            except ValueError as error:
                exclusions.append({"id": row["id"], "reason": str(error)})
            if len(examples) == args.examples:
                break
    if len(examples) != args.examples:
        raise ValueError("insufficient encodable training examples")
    model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    marker_id = tokenizer.convert_tokens_to_ids(CANDIDATE_MARKER)
    embedding = model.encoder.get_input_embeddings().weight
    initial_marker = embedding[marker_id].detach().clone()
    initial_head = torch.cat([p.detach().flatten() for p in model.head.parameters()]).clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def batch(example):
        return {k: v.to("cuda") for k, v in collate([example], tokenizer.pad_token_id, config.max_options).items()}

    def evaluate():
        model.eval()
        losses, correct = [], 0
        with torch.no_grad():
            for e in examples:
                logits = model(**batch(e)).logits.float()
                target = torch.tensor([e["row"]["target_index"]], device="cuda")
                losses.append(torch.nn.functional.cross_entropy(logits, target).item())
                correct += int(logits.argmax(-1).item() == target.item())
        return {"mean_ce": sum(losses) / len(losses), "accuracy": correct / len(examples)}

    report = {"args": vars(args), "purpose": "train-only bounded diagnostic; no test or development tuning",
              "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), *sorted(Path("packages/modernbert-decisions/decisions").glob("*.py"))]},
              "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "model_revision": config.revision, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "marker_id": marker_id,
              "examples": [{"id": e["row"]["id"], "source": e["row"]["source"], "target": e["row"]["target_index"],
                            "options": len(e["row"]["options"]), "tokens": len(e["input_ids"]),
                            "marker_positions": e["candidate_positions"]} for e in examples],
              "exclusions": exclusions, "initial": evaluate(), "updates": []}
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Microbatch one, accumulate across the same fixed tiny set per update.
        loss_total = 0.0
        for e in examples:
            logits = model(**batch(e)).logits.float()
            target = torch.tensor([e["row"]["target_index"]], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits, target) / len(examples)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite diagnostic loss")
            loss.backward()
            loss_total += loss.item()
        marker_grad = float(embedding.grad[marker_id].norm())
        head_grad = float(torch.stack([p.grad.norm().square() for p in model.head.parameters()]).sum().sqrt())
        total_grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        entry = {"step": step, "train_ce": loss_total, "marker_grad_norm": marker_grad,
                 "head_grad_norm": head_grad, "total_grad_norm_before_clip": total_grad,
                 "marker_change_norm": float((embedding[marker_id].detach() - initial_marker).norm()),
                 "head_change_norm": float((torch.cat([p.detach().flatten() for p in model.head.parameters()]) - initial_head).norm())}
        if step == 1 or step % 10 == 0 or step == args.steps:
            entry["tiny_set_evaluation"] = evaluate()
            print(json.dumps(entry), flush=True)
        report["updates"].append(entry)
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    report["final"] = evaluate()
    report["status"] = "completed"
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
