"""Compare paired prediction files. No model inference; calibration data is separate."""
import argparse
import csv
import json
import math
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("prediction files must have nonempty, unique IDs")
    for r in rows:
        z, y = r["logits"], r["target_index"]
        if len(z) < 2 or type(y) is not int or not 0 <= y < len(z) or not all(math.isfinite(v) for v in z):
            raise ValueError(f"invalid prediction: {r['id']}")
        if not r.get("group_id"):
            raise ValueError("group_id required for calibration/test separation")
    return rows


def paired(a, b):
    aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if aa.keys() != bb.keys():
        raise ValueError("CE and AURC must be evaluated on exactly the same example IDs")
    for key, r in aa.items():
        other = bb[key]
        for field in ("target_index", "group_id", "source", "type", "options"):
            if r.get(field) != other.get(field):
                raise ValueError(f"paired metadata differ: {key}/{field}")


def probabilities(logits, temperature):
    z = [v / temperature for v in logits]
    maximum = max(z)
    log_normalizer = maximum + math.log(sum(math.exp(v - maximum) for v in z))
    logp = [v - log_normalizer for v in z]
    return [math.exp(v) for v in logp], logp


def scored(rows, temperature=1.0):
    out = []
    for row in rows:
        p, lp = probabilities(row["logits"], temperature)
        index = max(range(len(p)), key=p.__getitem__)
        out.append({**row, "probabilities": p, "confidence": max(p), "error": int(index != row["target_index"]),
                    "nll": -lp[row["target_index"]]})
    return out


def fit_temperature(rows):
    # One scalar, bounded log-temperature search; no test labels participate.
    def objective(logt):
        return sum(-probabilities(r["logits"], math.exp(logt))[1][r["target_index"]] for r in rows) / len(rows)
    lo, hi = -3.0, 3.0
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(64):
        left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if objective(left) < objective(right):
            hi = right
        else:
            lo = left
    candidates = [0.0, -3.0, 3.0, (lo + hi) / 2]
    return math.exp(min(candidates, key=objective))


def curve(rows):
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    points, errors, area, start = [], 0, 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end]["confidence"] == ordered[start]["confidence"]:
            end += 1
        group_errors = sum(r["error"] for r in ordered[start:end])
        # Expected risk under random ordering inside exact ties, matching trainer AURC.
        for k in range(1, end - start + 1):
            area += (errors + k * group_errors / (end - start)) / (start + k)
        errors += group_errors
        points.append(dict(threshold=ordered[start]["confidence"], accepted=end,
                           coverage=end / len(rows), risk=errors / end))
        start = end
    return points, area / len(rows)


def reliability(rows, count=10):
    bins = [[] for _ in range(count)]
    for row in rows:
        bins[min(count - 1, int(row["confidence"] * count))].append(row)
    return [dict(bin=i, n=len(items), confidence=sum(r["confidence"] for r in items) / len(items),
                 accuracy=1 - sum(r["error"] for r in items) / len(items)) for i, items in enumerate(bins) if items]


def summary(rows):
    bins = reliability(rows)
    return dict(n=len(rows), accuracy=1 - sum(r["error"] for r in rows) / len(rows),
                nll=sum(r["nll"] for r in rows) / len(rows), aurc=curve(rows)[1],
                brier=sum(sum((p - (j == r["target_index"])) ** 2 for j, p in enumerate(r["probabilities"])) for r in rows) / len(rows),
                ece=sum(b["n"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / len(rows))


def threshold_result(calibration, test, risk_budget, minimum):
    candidates = [p for p in curve(calibration)[0] if p["risk"] <= risk_budget and p["accepted"] >= minimum]
    if not candidates:
        return dict(risk_budget=risk_budget, threshold=None, accepted=0, coverage=0, risk=None,
                    reason="No calibration operating point meets empirical risk and minimum-count requirements")
    chosen = max(candidates, key=lambda p: p["coverage"])
    accepted = [r for r in test if r["confidence"] >= chosen["threshold"]]
    return dict(risk_budget=risk_budget, threshold=chosen["threshold"], calibration=chosen,
                accepted=len(accepted), coverage=len(accepted) / len(test),
                risk=sum(r["error"] for r in accepted) / len(accepted) if accepted else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ce", required=True, help="CE test predictions.jsonl")
    ap.add_argument("--aurc", required=True, help="AURC test predictions.jsonl")
    ap.add_argument("--ce-calibration")
    ap.add_argument("--aurc-calibration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-accepted", type=int, default=30)
    ap.add_argument("--illustrative", action="store_true", help="Label fixture/random-model reports visibly")
    args = ap.parse_args()
    if bool(args.ce_calibration) != bool(args.aurc_calibration):
        ap.error("provide both calibration files or neither")
    if args.min_accepted < 1:
        ap.error("min-accepted must be positive")
    test = {"CE": load(args.ce), "CE+AURC": load(args.aurc)}
    paired(*test.values())
    calibration = {}
    if args.ce_calibration:
        calibration = {"CE": load(args.ce_calibration), "CE+AURC": load(args.aurc_calibration)}
        paired(*calibration.values())
        test_groups = {r["group_id"] for rows in test.values() for r in rows}
        if test_groups & {r["group_id"] for rows in calibration.values() for r in rows}:
            raise ValueError("calibration and test groups overlap")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("report output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = {}
    for name, rows in test.items():
        variants[name] = (scored(rows), scored(calibration[name]) if calibration else None, 1.0)
        if calibration:
            temp = fit_temperature(calibration[name])
            variants[name + " + temperature"] = (scored(rows, temp), scored(calibration[name], temp), temp)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    results, csv_rows = {}, []
    for name, (rows, cal, temp) in variants.items():
        points, _ = curve(rows)
        line, = axes[0].plot([p["coverage"] for p in points], [p["risk"] for p in points], marker=".", label=name)
        color = line.get_color()
        bins = reliability(rows)
        axes[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", color=color, label=name)
        thresholds = [i / 100 for i in range(101)]
        coverage, risk = [], []
        for threshold in thresholds:
            accepted = [r for r in rows if r["confidence"] >= threshold]
            coverage.append(len(accepted) / len(rows))
            risk.append(sum(r["error"] for r in accepted) / len(accepted) if accepted else float("nan"))
        axes[2].plot(thresholds, coverage, color=color, label=name + " coverage")
        axes[2].plot(thresholds, risk, "--", color=color, label=name + " risk")
        results[name] = dict(temperature=temp, overall=summary(rows), reliability_bins=bins,
                            by_source={s: summary([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})},
                            by_type={t: summary([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})})
        if cal:
            results[name]["fixed_threshold_test"] = [threshold_result(cal, rows, budget, args.min_accepted) for budget in (.01, .05, .1)]
        csv_rows.extend(dict(variant=name, **p) for p in points)
    axes[0].set(xlabel="Coverage (fraction answered)", ylabel="Selective error rate", title="Risk–coverage: lower is better")
    for budget in (.01, .05, .1):
        axes[0].axhline(budget, color="gray", lw=.5, alpha=.5)
    axes[1].plot([0, 1], [0, 1], ":", color="gray")
    axes[1].set(xlabel="Mean confidence", ylabel="Observed accuracy", title="Reliability (10 equal-width bins)")
    axes[2].set(xlabel="Confidence threshold", ylabel="Fraction", title="Solid: coverage / dashed: risk")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
    prefix = "ILLUSTRATIVE SMOKE — NOT MODEL-QUALITY EVIDENCE" if args.illustrative else "Paired held-out prediction comparison"
    fig.suptitle(prefix + "\nPoint estimates; no confidence bands. Counts and fixed-threshold outcomes in JSON.", fontsize=12)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    report = dict(illustrative=args.illustrative, files=vars(args), results=results,
                  definitions={"aurc": "discrete mean 0/1 risk; expected within-tie ordering",
                               "curve": "only attainable whole-tie acceptance points; lines guide the eye",
                               "ece": "10 equal-width confidence bins; counts saved",
                               "thresholds": "max coverage meeting empirical calibration risk budget, minimum accepted count; unchanged on test; no population guarantee",
                               "temperature": "scalar NLL fit on calibration only; bounded exp(-3)..exp(3)",
                               "uncertainty": "no confidence intervals here; grouped uncertainty analysis required before research claims"})
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "risk-coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "threshold", "accepted", "coverage", "risk"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output / "comparison.png")


if __name__ == "__main__":
    main()
