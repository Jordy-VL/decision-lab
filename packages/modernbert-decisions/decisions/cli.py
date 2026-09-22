"""Small train/evaluate/predict entry points; checkpoint continuation is weights-only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

from .config import parse, save_config
from .data import load


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def batch_inputs(examples, tokenizer, config, device):
    from .model import collate
    return {key: value.to(device) for key, value in collate(examples, tokenizer.pad_token_id, config.max_options).items()}


def train(model, tokenizer, examples, config, device):
    import torch
    from .loss import decision_loss
    if not examples:
        raise ValueError("no train rows; supply explicit splits or more document groups")
    forbidden = [e["row"]["id"] for e in examples if e["row"].get("evaluation_only") or
                 any(s in e["row"]["source"].lower().replace("_", "-") for s in ("decision-index", "rvl-cdip"))]
    if forbidden:
        raise ValueError(f"evaluation-only/rejected source in training: {forbidden[:5]}")
    if config.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng, updates = random.Random(config.seed), 0
    for epoch in range(config.epochs):
        model.train()
        order = list(range(len(examples)))
        rng.shuffle(order)
        batches = [order[i:i + config.batch_size] for i in range(0, len(order), config.batch_size)]
        total_loss = 0.0
        # Accumulate sums divided by actual window examples, including the partial final window.
        for start in range(0, len(batches), config.accumulation):
            window = batches[start:start + config.accumulation]
            count = sum(map(len, window))
            optimizer.zero_grad(set_to_none=True)
            for indices in window:
                items = [examples[i] for i in indices]
                logits = model(**batch_inputs(items, tokenizer, config, device))
                targets = torch.tensor([e["row"]["target_index"] for e in items], device=device)
                types = [e["row"]["type"] for e in items] if config.rank_by_type else None
                loss = decision_loss(logits, targets, config.aurc_lambda, types)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                (loss * len(items) / count).backward()
                total_loss += loss.item() * len(items)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        print(json.dumps(dict(epoch=epoch + 1, loss=total_loss / len(examples), optimizer_steps=updates)), flush=True)
    model.save(Path(config.output) / "checkpoint", tokenizer)
    return {"optimizer_steps": updates, "epochs": config.epochs, "continuation": "weights-only; fresh AdamW, shuffle and RNG"}


def predict(model, tokenizer, examples, config, device):
    import torch
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(examples), config.batch_size):
            items = examples[start:start + config.batch_size]
            batch_logits = model(**batch_inputs(items, tokenizer, config, device)).float().cpu()
            for e, padded_logits in zip(items, batch_logits):
                row = e["row"]
                logits = padded_logits[:len(row["options"])]
                if not torch.isfinite(logits).all():
                    raise ValueError(f'{row["id"]}: nonfinite logits')
                log_probs = logits.log_softmax(-1)
                probs = log_probs.exp().tolist()
                index = int(logits.argmax())
                result = {k: row[k] for k in ("id", "group_id", "source", "split", "type", "options")}
                result.update(index=index, logits=logits.tolist(), probabilities=probs, confidence=max(probs), token_counts=e["token_counts"])
                for key in ("evidence", "provenance"):
                    if key in row:
                        result[key] = row[key]
                if row.get("target_index") is not None:
                    result.update(target_index=row["target_index"], label=row["options"][row["target_index"]], nll=-log_probs[row["target_index"]].item())
                if row["type"] == "ordinal":
                    result.update(values=row["values"], expected_value=sum(p * v for p, v in zip(probs, row["values"])))
                if row["type"] == "boolean":
                    result["probability_true"] = probs[1]
                rows.append(result)
    return rows


def main(argv=None):
    command, config = parse(argv)
    rows = load(config.data, config, labeled=command != "predict")
    if command == "check":
        excessive = [r["id"] for r in rows if len(r["options"]) > config.max_options]
        if excessive:
            raise ValueError(f"max_options exceeded: {excessive[:5]}")
        print(json.dumps({"rows": len(rows), "splits": dict(Counter(r["split"] for r in rows)), "types": dict(Counter(r["type"] for r in rows)), "token_lengths": "not checked (no tokenizer loaded)"}))
        return
    if command != "train" and not config.checkpoint:
        raise ValueError("evaluate/predict require --checkpoint")
    selected = [r for r in rows if r["split"] == ("train" if command == "train" else config.split)]
    if not selected:
        raise ValueError("selected split is empty")
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output}")
    import torch
    from .model import initialize, encode
    from .metrics import reports
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, tokenizer = initialize(config)
    if config.max_length > model.encoder.config.max_position_embeddings:
        raise ValueError("max_length exceeds encoder context limit")
    # Validate every selected row before any optimizer update; never drop overflow rows.
    examples = [encode(r, tokenizer, config.max_length, config.max_options) for r in selected]
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output)
    metadata = dict(command=command, timestamp=datetime.now(timezone.utc).isoformat(), seed=config.seed,
                    device=device, model=config.model, requested_revision=config.revision,
                    resolved_encoder_revision=getattr(model.encoder.config, "_commit_hash", None),
                    data_revision=config.data_revision, data_sha256=hashlib.sha256(Path(config.data).read_bytes()).hexdigest(),
                    selected_rows=len(selected), versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "PyYAML")},
                    initialized_from=config.checkpoint or config.model)
    if config.checkpoint:
        parent_metadata = Path(config.checkpoint).parent / "metadata.json"
        metadata["parent_metadata"] = json.loads(parent_metadata.read_text(encoding="utf-8")) if parent_metadata.exists() else None
    write_json(output / "metadata.json", metadata)
    write_jsonl(output / "split_manifest.jsonl", [{k: r[k] for k in ("id", "group_id", "source", "split")} for r in rows])
    model.to(device)
    if command == "train":
        write_json(output / "training.json", train(model, tokenizer, examples, config, device))
    else:
        predictions = predict(model, tokenizer, examples, config, device)
        write_jsonl(output / "predictions.jsonl", predictions)
        if command == "evaluate":
            result = reports(predictions)
            write_json(output / "report.json", result)
            print(json.dumps(result["overall"]))


if __name__ == "__main__":
    main()
