"""Exploratory scalar correctness calibration; see frozen flexible protocol."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

from compare_runs import load, paired, scored, curve, reliability
from analyze_five_epoch_calibration import ARMS, ROOT, require, read


def partition(group):
    return int(hashlib.sha256(("confidence-v1:" + group).encode()).hexdigest()[:8], 16) % 10 < 6


def pava(values, weights):
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i + 1, weight, value * weight])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    result = [0.0] * len(values)
    for start, end, weight, total in blocks:
        result[start:end] = [total / weight] * (end - start)
    return result


def slopes(x, y):
    if len(x) == 1:
        return [0.0]
    h = [b - a for a, b in zip(x, x[1:])]
    d = [(b - a) / step for a, b, step in zip(y, y[1:], h)]
    if len(x) == 2:
        return [d[0], d[0]]
    result = [0.0] * len(x)
    for i in range(1, len(x) - 1):
        if d[i - 1] > 0 and d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            result[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    def endpoint(h0, h1, d0, d1):
        value = ((2*h0 + h1)*d0 - h0*d1) / (h0 + h1)
        return max(0.0, min(value, 3*d0))
    result[0] = endpoint(h[0], h[1], d[0], d[1])
    result[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
    return result


def fit(rows, method):
    if method == "raw":
        return {"method": method}
    groups = {}
    for row in rows:
        score = row["confidence"]
        key = score if method == "isotonic" else min(9, int(score * 10))
        groups.setdefault(key, []).append((score, 1 - row["error"]))
    x, y, weights = [], [], []
    for key in sorted(groups):
        items = groups[key]
        n = len(items)
        x.append(key if method == "isotonic" else sum(s for s, _ in items) / n)
        correction = 0.0 if method == "isotonic" else 1.0
        weights.append(n + correction)
        y.append((sum(v for _, v in items) + correction / 2) / (n + correction))
    y = pava(y, weights)
    result = {"method": method, "x": x, "y": y}
    if method == "spline":
        result["slopes"] = slopes(x, y)
    return result


def predict(model, score):
    if model["method"] == "raw":
        return score
    x, y = model["x"], model["y"]
    if score <= x[0]:
        return y[0]
    if score >= x[-1]:
        return y[-1]
    i = bisect.bisect_right(x, score) - 1
    if model["method"] == "isotonic":
        return y[i]
    width = x[i+1] - x[i]
    t = (score - x[i]) / width
    m = model["slopes"]
    value = (2*t**3 - 3*t*t + 1)*y[i] + (t**3 - 2*t*t + t)*width*m[i] + (-2*t**3 + 3*t*t)*y[i+1] + (t**3 - t*t)*width*m[i+1]
    return min(y[i+1], max(y[i], value))


def apply(model, rows):
    return [{**r, "confidence": predict(model, r["confidence"])} for r in rows]


def metrics(rows):
    n = len(rows)
    bins = reliability(rows)
    # Integral of cumulative errors / n; exact ties use mean error placement.
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    total, errors, start = 0.0, 0, 0
    while start < n:
        end = start + 1
        while end < n and ordered[end]["confidence"] == ordered[start]["confidence"]:
            end += 1
        failures = sum(r["error"] for r in ordered[start:end])
        total += (end-start)*errors + failures*(end-start)/2
        errors += failures
        start = end
    return {"n": n, "accuracy": 1-sum(r["error"] for r in rows)/n,
            "correctness_brier": sum((r["confidence"]-(1-r["error"]))**2 for r in rows)/n,
            "correctness_log_loss": -sum(math.log(max(1e-12, min(1-1e-12, r["confidence"]))) if not r["error"] else math.log(max(1e-12, min(1-1e-12, 1-r["confidence"]))) for r in rows)/n,
            "correctness_ece": sum(b["n"]*abs(b["confidence"]-b["accuracy"]) for b in bins)/n,
            "aurc": curve(rows)[1], "augrc": total/(n*n)}


def analyze(base, out):
    inputs, hashes, provenance = {}, {}, {}
    for name, stem in ARMS.items():
        train, td = base/stem, base/(stem+"-test-raw")
        cd = train/"evaluations/best/calibration"
        cm, tm, state = read(cd/"metadata.json"), read(td/"metadata.json"), read(train/"trainer_state.json")
        require(cm["checkpoint"] == tm["initialized_from"] and cm["best_model_checkpoint"] == state["best_model_checkpoint"], "checkpoint provenance mismatch")
        require(tm["parent_metadata"] == read(train/"metadata.json"), "parent provenance mismatch")
        best_step = int(state["best_model_checkpoint"].rsplit("-", 1)[1])
        evaluations = [x for x in state["log_history"] if "eval_nll" in x]
        selected = [x for x in evaluations if x["step"] == best_step]
        require(len(selected) == 1 and math.isclose(selected[0]["eval_nll"], state["best_metric"], abs_tol=1e-8)
                and math.isclose(min(x["eval_nll"] for x in evaluations), state["best_metric"], abs_tol=1e-8), "historical NLL selection mismatch")
        provenance[name] = {"export": cm["checkpoint"], "best_checkpoint": cm["best_model_checkpoint"]}
        splits = {}
        for split, directory in (("calibration", cd), ("test", td)):
            path = directory/"predictions.jsonl"
            rows = load(path)
            require(all(r["split"] == split and len(r["options"]) == len(r["logits"]) for r in rows), "invalid split/options")
            splits[split] = scored(rows)
            hashes[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for field in ("id", "group_id"):
            require(not ({r[field] for r in splits["calibration"]} & {r[field] for r in splits["test"]}), "calibration/test overlap")
        splits["fit"] = [r for r in splits["calibration"] if partition(r["group_id"])]
        splits["assessment"] = [r for r in splits["calibration"] if not partition(r["group_id"])]
        require(splits["fit"] and splits["assessment"], "empty calibration partition")
        inputs[name] = splits
    for splits in inputs.values():
        for split in ("calibration", "test", "fit", "assessment"):
            paired(inputs["CE"][split], splits[split])
    models = {name: {method: fit(splits["fit"], method) for method in ("raw", "isotonic", "spline")} for name, splits in inputs.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out/"frozen-mappings.json").write_text(json.dumps(models, indent=2)+"\n")
    results, thresholds = {}, {}
    # Freeze operating points using assessment only before test application.
    for name, splits in inputs.items():
        for method, model in models[name].items():
            cal = apply(model, splits["assessment"])
            points = []
            for budget in (.01, .05, .10):
                eligible = [p for p in curve(cal)[0] if p["risk"] <= budget and p["accepted"] >= 30]
                point = dict(max(eligible, key=lambda p: p["coverage"])) if eligible else {"threshold": None, "accepted": 0, "coverage": 0, "risk": None}
                point["errors"] = round(point["accepted"] * point["risk"]) if point["risk"] is not None else 0
                points.append(point)
            thresholds[name+"/"+method] = points
    (out/"frozen-thresholds.json").write_text(json.dumps(thresholds, indent=2)+"\n")
    for name, splits in inputs.items():
        for method, model in models[name].items():
            mapped = {s: apply(model, splits[s]) for s in ("fit", "assessment", "test")}
            operating = []
            for budget, point in zip((.01,.05,.10), thresholds[name+"/"+method]):
                threshold = point["threshold"]
                selected = [r for r in mapped["test"] if threshold is not None and r["confidence"] >= threshold]
                errors = sum(r["error"] for r in selected)
                operating.append({"risk_budget": budget, "threshold": threshold, "assessment": point,
                                  "test_accepted": len(selected), "test_errors": errors, "test_coverage": len(selected)/len(mapped["test"]),
                                  "test_risk": errors/len(selected) if selected else None})
            results[name+"/"+method] = {s: metrics(rows) for s, rows in mapped.items()}
            results[name+"/"+method]["operating_points"] = operating
    report = {"status": "exploratory scalar correctness calibration; historical NLL-selected seed 17; test previously inspected",
              "protocol_sha256": hashlib.sha256((ROOT/"docs/flexible-calibration-protocol-20260930.md").read_bytes()).hexdigest(),
              "prediction_sha256": hashes, "provenance": provenance, "results": results}
    (out/"aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT/"runs/remote-review-20260930/runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.runs, args.out)
    for name, result in report["results"].items():
        print(name, result["assessment"], result["test"])
