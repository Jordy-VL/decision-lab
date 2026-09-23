"""Small train/evaluate/predict entry points; checkpoint continuation is weights-only."""
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from .config import parse, save_config
from .data import load


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def batch_inputs(examples, tokenizer, config, device):
    from .model import collate
    return {key: value.to(device) for key, value in collate(examples, tokenizer.pad_token_id, config.max_options).items()}


def predictions_from_logits(examples, padded_logits):
    import math

    rows = []
    for example, values in zip(examples, padded_logits):
        row = example["row"]
        logits = [float(value) for value in values[:len(row["options"])]]
        if not all(math.isfinite(value) for value in logits):
            raise ValueError(f'{row["id"]}: nonfinite active-option logits')
        maximum = max(logits)
        exp_values = [math.exp(value - maximum) for value in logits]
        denominator = sum(exp_values)
        probabilities = [value / denominator for value in exp_values]
        index = max(range(len(logits)), key=logits.__getitem__)
        result = {k: row[k] for k in ("id", "group_id", "source", "split", "type", "options")}
        result.update(index=index, logits=logits, probabilities=probabilities,
                      confidence=max(probabilities), token_counts=example["token_counts"])
        target = row.get("target_index")
        if target is not None:
            result.update(target_index=target, label=row["options"][target], nll=-math.log(probabilities[target]))
        if row["type"] == "ordinal":
            result.update(values=row["values"], expected_value=sum(p * v for p, v in zip(probabilities, row["values"])))
        if row["type"] == "boolean":
            result["probability_true"] = probabilities[1]
        rows.append(result)
    return rows


def save_prediction_rows(examples, predictions, output_dir, split, checkpoint, trainer, variant):
    import numpy as np

    if len(predictions) != len(examples):
        raise ValueError(f"{split}: prediction count does not match the split manifest")
    max_options = max(len(e["row"]["options"]) for e in examples)
    logits = np.full((len(examples), max_options), -np.inf, dtype=np.float32)
    labels = np.empty(len(examples), dtype=np.int64)
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        logits[i, :len(prediction["logits"])] = prediction["logits"]
        labels[i] = example["row"]["target_index"]
    directory = Path(output_dir) / "evaluations" / variant / split
    directory.mkdir(parents=True, exist_ok=False)
    write_jsonl(directory / "predictions.jsonl", predictions)
    np.savez_compressed(directory / "logits.npz", logits=logits, labels=labels,
                        option_counts=np.asarray([len(e["row"]["options"]) for e in examples], dtype=np.int32),
                        ids=np.asarray([e["row"]["id"] for e in examples]),
                        group_ids=np.asarray([e["row"]["group_id"] for e in examples]),
                        sources=np.asarray([e["row"]["source"] for e in examples]),
                        types=np.asarray([e["row"]["type"] for e in examples]))
    from .metrics import reports
    write_json(directory / "report.json", reports(predictions))
    write_json(directory / "metadata.json", {
        "split": split,
        "checkpoint": str(checkpoint),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "rows": len(examples),
        "logits_shape": list(logits.shape),
        "labels_shape": list(labels.shape),
        "trainer_metrics": getattr(trainer, "metrics", {}) if variant == "best" else {},
    })


def save_evaluation_artifacts(examples, prediction_output, output_dir, split, checkpoint, trainer):
    raw_logits = prediction_output.predictions
    if isinstance(raw_logits, tuple):
        raw_logits = raw_logits[0]
    if len(raw_logits) != len(examples) or len(prediction_output.label_ids) != len(examples):
        raise ValueError(f"{split}: Trainer prediction count does not match the split manifest")
    expected_labels = [example["row"]["target_index"] for example in examples]
    if list(prediction_output.label_ids) != expected_labels:
        raise ValueError(f"{split}: Trainer label order differs from the saved split manifest")
    predictions = predictions_from_logits(examples, raw_logits)
    trainer_metrics = prediction_output.metrics
    trainer.metrics = trainer_metrics
    save_prediction_rows(examples, predictions, output_dir, split, checkpoint, trainer, "best")


def train_with_hf(model, tokenizer, train_examples, development_examples, calibration_examples, config, output):
    from .hf_trainer import create_trainer

    if not train_examples:
        raise ValueError("no train rows; supply explicit splits or more document groups")
    forbidden = [e["row"]["id"] for e in train_examples if e["row"].get("evaluation_only") or
                 any(s in e["row"]["source"].lower().replace("_", "-") for s in ("decision-index", "rvl-cdip"))]
    if forbidden:
        raise ValueError(f"evaluation-only/rejected source in training: {forbidden[:5]}")
    if not development_examples or not calibration_examples:
        raise ValueError("training requires nonempty development and calibration partitions for saved logits")

    trainer = create_trainer(model, tokenizer, train_examples, development_examples, config, output)
    train_result = trainer.train(resume_from_checkpoint=config.resume or None)
    trainer.save_model(str(Path(output) / "hf_model"))
    trainer.save_state()
    model.save(Path(output) / "checkpoint", tokenizer)
    write_json(Path(output) / "trainer_state.json", asdict(trainer.state))
    write_json(Path(output) / "training.json", train_result.metrics)
    write_json(Path(output) / "trainer_log_history.json", trainer.state.log_history)

    development_output = trainer.predict(trainer.eval_dataset, metric_key_prefix="development")
    save_evaluation_artifacts(development_examples, development_output, output, "development",
                              Path(output) / "checkpoint", trainer)
    from .hf_trainer import EncodedRows
    calibration_output = trainer.predict(EncodedRows(calibration_examples), metric_key_prefix="calibration")
    save_evaluation_artifacts(calibration_examples, calibration_output, output, "calibration",
                              Path(output) / "checkpoint", trainer)

    from .model import DecisionModel
    final_model, _ = DecisionModel.load(Path(output) / "final_budget_checkpoint")
    device = next(trainer.model.parameters()).device
    final_model.to(device)
    final_development = predict(final_model, tokenizer, development_examples, config, device)
    final_calibration = predict(final_model, tokenizer, calibration_examples, config, device)
    save_prediction_rows(development_examples, final_development, output, "development",
                         Path(output) / "final_budget_checkpoint", trainer, "final_budget")
    save_prediction_rows(calibration_examples, final_calibration, output, "calibration",
                         Path(output) / "final_budget_checkpoint", trainer, "final_budget")
    return train_result.metrics


def predict(model, tokenizer, examples, config, device):
    import torch
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(examples), config.batch_size):
            items = examples[start:start + config.batch_size]
            batch_logits = model(**batch_inputs(items, tokenizer, config, device)).logits.float().cpu()
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
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise ValueError(f"output directory must be new or empty: {output}")
    if config.resume and not Path(config.resume).is_dir():
        raise ValueError(f"resume checkpoint does not exist: {config.resume}")
    import torch
    from .model import initialize, encode
    from .metrics import reports
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
    development_examples, calibration_examples = [], []
    extra_data_hashes = {}
    if command == "train":
        if not config.development_data or not config.calibration_data:
            raise ValueError("train requires data.development_path and data.calibration_path to save validation/calibration logits")
        development_rows, calibration_rows = [], []
        for name, path, expected, destination in (
            ("development", config.development_data, "development", development_rows),
            ("calibration", config.calibration_data, "calibration", calibration_rows),
        ):
            split_config = replace(config, data=path, data_split="")
            split_rows = load(path, split_config, labeled=True)
            if any(row["split"] != expected for row in split_rows):
                raise ValueError(f"{name} file must contain only the {expected!r} partition")
            destination.extend(split_rows)
            extra_data_hashes[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        train_groups = {r["group_id"] for r in selected}
        development_groups = {r["group_id"] for r in development_rows}
        calibration_groups = {r["group_id"] for r in calibration_rows}
        if train_groups & development_groups or train_groups & calibration_groups or development_groups & calibration_groups:
            raise ValueError("train, development and calibration groups must be disjoint")
        development_examples = [encode(r, tokenizer, config.max_length, config.max_options) for r in development_rows]
        calibration_examples = [encode(r, tokenizer, config.max_length, config.max_options) for r in calibration_rows]
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output)
    package_root = Path(__file__).resolve().parents[3]
    try:
        code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = None
    metadata = dict(command=command, timestamp=datetime.now(timezone.utc).isoformat(), seed=config.seed,
                    code_revision=code_revision,
                    device=device, model=config.model, requested_revision=config.revision,
                    resolved_encoder_revision=getattr(model.encoder.config, "_commit_hash", None),
                    data_revision=config.data_revision, data_sha256=hashlib.sha256(Path(config.data).read_bytes()).hexdigest(),
                    selected_rows=len(selected), versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "PyYAML")},
                    initialized_from=config.checkpoint or config.model)
    if extra_data_hashes:
        metadata["additional_data_sha256"] = extra_data_hashes
    if config.checkpoint:
        parent_metadata = Path(config.checkpoint).parent / "metadata.json"
        metadata["parent_metadata"] = json.loads(parent_metadata.read_text(encoding="utf-8")) if parent_metadata.exists() else None
    write_json(output / "metadata.json", metadata)
    write_jsonl(output / "split_manifest.jsonl", [{k: r[k] for k in ("id", "group_id", "source", "split")} for r in rows])
    if command == "train":
        metadata["status"] = "running"
        write_json(output / "metadata.json", metadata)
        try:
            metrics = train_with_hf(model, tokenizer, examples, development_examples, calibration_examples, config, output)
            metadata["status"] = "completed"
            metadata["training_metrics"] = metrics
            write_json(output / "metadata.json", metadata)
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            write_json(output / "metadata.json", metadata)
            raise
    else:
        model.to(device)
        predictions = predict(model, tokenizer, examples, config, device)
        write_jsonl(output / "predictions.jsonl", predictions)
        if command == "evaluate":
            result = reports(predictions)
            write_json(output / "report.json", result)
            print(json.dumps(result["overall"]))


if __name__ == "__main__":
    main()
