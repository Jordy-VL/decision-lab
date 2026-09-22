"""Download and adapt the pinned Kev suite, preserving its frozen partitions."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/modernbert-decisions"))
from decisions.data import adapt, validate


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def obtain(url, path, expected):
    raw = path.read_bytes() if path.exists() else urllib.request.urlopen(url, timeout=120).read()
    if digest(raw) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}; refusing changed data")
    if not path.exists():
        path.write_bytes(raw)
    return raw


def convert(record, split):
    record = copy.deepcopy(record)
    meta = record.get("_meta", {})
    if not all(isinstance(meta.get(k), str) and meta[k] for k in ("id", "group_id", "source")):
        raise ValueError("missing upstream record id/group/source")
    if not isinstance(record.get("questions"), dict) or not record["questions"]:
        raise ValueError("expected nonempty labeled request questions")
    if "split" in record or "group_id" in record or "id" in record:
        raise ValueError("ambiguous root metadata; expected pinned _meta format")
    record["split"] = split  # _meta.split is the original dataset split, not suite partition.
    for key, q in record["questions"].items():
        if not isinstance(key, str) or "label" not in q or "instructions" not in q:
            raise ValueError("missing question identity/instructions/label")
        if q["type"] == "score":
            if not isinstance(q.get("criteria"), list) or "values" in q:
                raise ValueError("unexpected ordinal representation")
            q["values"] = list(range(len(q["criteria"])))
    rows = [validate(row) for row in adapt(record)]
    for row, (key, q) in zip(rows, record["questions"].items()):
        row["provenance"] = dict(meta, kev_question_id=key, kev_type=q["type"],
                                 kev_label=q["label"], kev_criteria=q.get("criteria"),
                                 suite_split=split)
        if row["type"] == "ordinal":
            row["provenance"]["ordinal_semantics"] = "zero_based_level_index"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kev-baseline.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/kev-decision-v7")
    parser.add_argument("--allow-test", action="store_true", help="Explicitly prepare locked test; never use for tuning")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["ordinal_values"] != "zero_based_level_index" or cfg["mixture"] != "unchanged_frozen_suite":
        raise ValueError("this adapter only supports the unchanged frozen baseline")
    splits = cfg["splits"] + (["test"] if args.allow_test else [])
    if cfg["splits"] != ["train", "calibration", "development"]:
        raise ValueError("baseline partitions must remain train/calibration/development")
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_url = (f'https://raw.githubusercontent.com/jaredpalmer/kev/{cfg["upstream_commit"]}'
                    f'/evals/{cfg["suite"]}/manifest.json')
    manifest = json.loads(obtain(manifest_url, raw_dir / "manifest.json", cfg["manifest_sha256"]))
    groups, ids, state_splits = {}, set(), collections.defaultdict(set)
    report = {"config": cfg, "test_enabled": args.allow_test, "splits": {}}
    pending = {}
    for split in splits:
        name = split + ".jsonl"
        expected = manifest["files"][name]
        url = f'https://huggingface.co/datasets/{cfg["dataset"]}/resolve/{cfg["dataset_revision"]}/{cfg["suite"]}/{name}'
        raw = obtain(url, raw_dir / name, expected["sha256"])
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(records) != expected["records"]:
            raise ValueError(f"record count mismatch: {split}")
        rows = [row for record in records for row in convert(record, split)]
        if len(rows) != expected["questions"]:
            raise ValueError(f"question count mismatch: {split}")
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f'duplicate question id: {row["id"]}')
            ids.add(row["id"])
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError(f'group crosses suite partitions: {row["group_id"]}')
            state_splits[digest(row["state"].encode())].add(split)
        output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        pending[name] = output
        report["splits"][split] = {"records": len(records), "questions": len(rows),
            "groups": len({r["group_id"] for r in rows}), "input_sha256": digest(raw),
            "output_sha256": digest(output), "types": dict(collections.Counter(r["type"] for r in rows)),
            "sources": dict(collections.Counter(r["source"] for r in rows)),
            "max_options": max(len(r["options"]) for r in rows)}
    report["cross_split_identical_states"] = sum(len(s) > 1 for s in state_splits.values())
    report["notes"] = ["Repeated groups within one partition are preserved (multiple questions/variants).",
        "Identical state text is audited but not removed: doing so would change the frozen baseline.",
        "No fuzzy or pretraining contamination certification; no model training performed."]
    # Write only after every selected partition passes integrity and schema checks.
    for name, output in pending.items():
        path = args.out / name
        if path.exists() and path.read_bytes() != output:
            raise ValueError(f"refusing to replace different prepared data: {path}")
    for name, output in pending.items():
        (args.out / name).write_bytes(output)
    (args.out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
