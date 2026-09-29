"""Decision Index adapter for one project-format ModernBERT checkpoint."""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "packages" / "modernbert-decisions"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch

from decisions.model import DecisionModel, collate, encode
from decision_index.engines import Engine, Unsupported


def _text(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class ModernBERTDecisionEngine(Engine):
    """Score all supplied questions with one local project-format checkpoint."""

    def __init__(self, checkpoint, device="auto", max_length=2048, **options):
        super().__init__(checkpoint=checkpoint, device=device, max_length=max_length, **options)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not (checkpoint_path / "model.json").is_file():
            raise FileNotFoundError(
                f"project-format checkpoint metadata not found: {checkpoint_path / 'model.json'}"
            )
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be a positive integer")

        self.model, self.tokenizer = DecisionModel.load(checkpoint_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be auto, cuda or cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.provenance = {
            "kind": "local-checkpoint",
            "checkpoint": str(checkpoint_path),
            "decision_head": self.model.decision_head,
            "max_options": self.model.max_options,
            "max_length": max_length,
        }

    def __call__(self, state, questions):
        rows, mapping = [], []
        for question_key, question in questions.items():
            question_type = question.get("type")
            if question_type == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or len(criteria) < 2:
                    raise ValueError(f"{question_key}: choice criteria must be a mapping with at least two options")
                keys = list(criteria)
                options = [
                    key if criteria[key] is None else f"{key}: {_text(criteria[key])}"
                    for key in keys
                ]
                prompt = _text(question.get("instructions", ""))
            elif question_type == "noul":
                keys = ["false", "true"]
                options = keys
                prompt = _text(question.get("instructions", ""))
                if question.get("criteria") is not None:
                    prompt += "\nBoolean definitions: " + _text(question["criteria"])
            else:
                raise Unsupported(f"unsupported question type: {question_type}")

            if len(options) > self.model.max_options:
                raise Unsupported(
                    f"{question_key}: {len(options)} options exceeds model capacity "
                    f"{self.model.max_options}"
                )
            rows.append({
                "id": str(question_key),
                "group_id": "decision-index-request",
                "source": "decision-index",
                "state": _text(state),
                "question": prompt,
                "options": options,
                "target_index": None,
                "type": "boolean" if question_type == "noul" else "categorical",
            })
            mapping.append((question_key, question_type, keys))

        if not rows:
            raise ValueError("request contains no questions")

        examples = []
        for row in rows:
            try:
                examples.append(encode(
                    row, self.tokenizer, self.max_length,
                    self.model.max_options, self.model.decision_head,
                ))
            except ValueError as exc:
                if "exceeds max_length" in str(exc):
                    raise Unsupported(f'{row["id"]}: {exc}') from exc
                raise

        batch = {
            name: value.to(self.device)
            for name, value in collate(
                examples, self.tokenizer.pad_token_id, self.model.max_options
            ).items()
        }
        with torch.inference_mode():
            logits = self.model(**batch).logits.float()

        answers = {}
        for i, (question_key, question_type, keys) in enumerate(mapping):
            probabilities = logits[i, :len(keys)].softmax(-1).cpu().tolist()
            if question_type == "noul":
                answers[question_key] = {"type": "noul", "noul": float(probabilities[1])}
            else:
                distribution = dict(zip(keys, map(float, probabilities)))
                answers[question_key] = {
                    "type": "choice",
                    "choice": keys[max(range(len(keys)), key=probabilities.__getitem__)],
                    "probabilities": distribution,
                }
        return {"model": "modernbert-decision-checkpoint", "answers": answers}, None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def runtime(self):
        return {
            "device": str(self.device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }
