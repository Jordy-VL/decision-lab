"""Canonical JSONL and a deliberately small Kev labeled-request adapter."""
import hashlib
import json
import math
from pathlib import Path


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def adapt(record):
    if "questions" not in record:
        return [dict(record)]
    meta = record.get("_meta", {})
    state = text(record["state"])
    base = str(record.get("id", meta.get("id", hashlib.sha256(state.encode()).hexdigest())))
    group = str(record.get("group_id", meta.get("group_id", hashlib.sha256(state.encode()).hexdigest())))
    questions = record["questions"]
    if not isinstance(questions, dict):
        raise ValueError("Kev materialized records lose task types; use labeled requests or convert explicitly")
    rows = []
    for key, q in questions.items():
        kind = {"choice": "categorical", "noul": "boolean", "score": "ordinal"}[q["type"]]
        criteria, label = q.get("criteria"), q.get("label")
        if kind == "boolean":
            options = ["false", "true"]
            if label is not None and type(label) is not bool:
                raise ValueError("Kev noul label must be boolean")
            target = int(label) if label is not None else None
        elif kind == "categorical":
            options = [k if v is None else f"{k}: {text(v)}" for k, v in criteria.items()]
            target = list(criteria).index(label) if label is not None else None
        else:
            options, target = [text(v) for v in criteria], label
            if "values" not in q:
                raise ValueError("Kev score needs explicit numeric 'values'; add e.g. [1,2,3,4,5] for star ratings")
        row = dict(id=f"{base}/{key}", group_id=group, source=q.get("src", meta.get("source", "kev")),
                   split=record.get("split", meta.get("split")), state=state,
                   question=text(q.get("instructions", "")), options=options, target_index=target, type=kind,
                   provenance=meta)
        if kind == "boolean" and criteria:
            row["question"] += "\nBoolean definitions: " + text(criteria)
        if kind == "ordinal":
            row["values"] = q["values"]
        rows.append(row)
    return rows


def validate(row, labeled=True):
    for key in ("id", "group_id", "source", "state", "question"):
        if not isinstance(row.get(key), str) or (key in ("id", "group_id", "source") and not row[key]):
            raise ValueError(f"{key} must be a string (identifiers must be nonempty)")
    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2 or any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError("options must contain at least two nonempty strings")
    if len(set(options)) != len(options):
        raise ValueError("duplicate option text")
    kind = row.get("type")
    if kind not in ("categorical", "boolean", "ordinal"):
        raise ValueError("unknown type")
    if kind == "boolean" and options != ["false", "true"]:
        raise ValueError("boolean options must be ['false','true']")
    target = row.get("target_index")
    if (labeled or target is not None) and (type(target) is not int or not 0 <= target < len(options)):
        raise ValueError("invalid zero-based target_index")
    if kind == "ordinal":
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(options) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("ordinal values must be finite numbers aligned with options")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("ordinal values must be strictly increasing")
    elif "values" in row:
        raise ValueError("numeric values are only for ordinal tasks")
    if row.get("split") is not None and (not isinstance(row["split"], str) or not row["split"]):
        raise ValueError("split must be a nonempty string")
    return row


def load(path, config, labeled=True):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                rows.extend(validate(r, labeled) for r in adapt(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError("empty dataset")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row ids")
    if config.data_split:
        for row in rows:
            if row.get("split") and row["split"] != config.data_split:
                raise ValueError("data_split conflicts with existing official split")
            row["split"] = config.data_split
    assigned = [bool(r.get("split")) for r in rows]
    if any(assigned) and not all(assigned):
        raise ValueError("mixed assigned and missing splits; preserve official splits explicitly")
    groups = {}
    for row in rows:
        if not any(assigned):
            value = int(hashlib.sha256(f'{config.seed}:{row["group_id"]}'.encode()).hexdigest()[:16], 16) / 2**64
            row["split"] = "test" if value < config.test_fraction else "validation" if value < config.test_fraction + config.validation_fraction else "train"
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f'group crosses splits: {row["group_id"]}')
    return rows
