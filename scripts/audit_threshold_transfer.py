"""Describe historical threshold transfer by source/variant; never fit on test."""
import json
from pathlib import Path
from compare_runs import load, scored

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"CE": "x2-ce-5epoch", "AURC-only": "x2-aurc-only-5epoch",
        "CE+AURC": "x2-ce-aurc-mix-5epoch"}


def summarize(rows, threshold):
    accepted = [r for r in rows if threshold is not None and r["confidence"] >= threshold]
    errors = sum(r["error"] for r in accepted)
    return {"n": len(rows), "accuracy": 1 - sum(r["error"] for r in rows) / len(rows),
            "accepted": len(accepted), "errors": errors,
            "coverage": len(accepted) / len(rows),
            "risk": errors / len(accepted) if accepted else None}


def main():
    baseline = json.loads((ROOT / "reports/five-epoch-calibration-20260930.json").read_text())
    result = {"scope": "Exploratory descriptive slices after historical test inspection; no refitting or threshold selection",
              "baseline": "reports/five-epoch-calibration-20260930.json", "arms": {}}
    for arm, run in ARMS.items():
        frozen = baseline["results"][arm + "/temperature"]
        threshold = next(p["threshold"] for p in frozen["fixed_threshold_test"] if p["risk_budget"] == .05)
        splits = {}
        for split, suffix in (("calibration", run + "/evaluations/best/calibration"),
                              ("test", run + "-test-raw")):
            rows = scored(load(ROOT / "runs/remote-review-20260930/runs" / suffix / "predictions.jsonl"), frozen["temperature"])
            groups = {}
            for field in ("source", "variant"):
                key = lambda r: (r["source"] if field == "source" else r.get("provenance", {}).get("variant", "unspecified"))
                groups[field] = {v: summarize([r for r in rows if key(r) == v], threshold)
                                 for v in sorted({key(r) for r in rows})}
                assert sum(s["n"] for s in groups[field].values()) == len(rows)
                assert sum(s["errors"] for s in groups[field].values()) == summarize(rows, threshold)["errors"]
            splits[split] = {"overall": summarize(rows, threshold), "slices": groups}
        result["arms"][arm] = {"temperature": frozen["temperature"], "threshold": threshold, **splits}
    path = ROOT / "reports/threshold-transfer-slices-20260930.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for arm, data in result["arms"].items():
        print(arm)
        for split in ("calibration", "test"):
            print(split, data[split]["slices"]["variant"])


if __name__ == "__main__":
    main()
