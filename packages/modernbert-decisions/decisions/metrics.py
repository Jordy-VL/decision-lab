"""Selective-classification metrics with tie-aware AURC and AUGRC."""
import math


def augrc(rows):
    """Compute AUGRC for 0/1 failures using confidence-ranked pair comparisons."""
    correct = [r["confidence"] for r in rows if r["index"] == r["target_index"]]
    failures = [r["confidence"] for r in rows if r["index"] != r["target_index"]]
    accuracy = len(correct) / len(rows)
    if not correct or not failures:
        return 0.5 * (1 - accuracy) ** 2
    failure_auc = sum(
        failure < success or 0.5 * (failure == success)
        for failure in failures
        for success in correct
    ) / (len(failures) * len(correct))
    return (1 - failure_auc) * accuracy * (1 - accuracy) + 0.5 * (1 - accuracy) ** 2


def report(rows):
    if not rows:
        raise ValueError("cannot evaluate empty rows")
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    cumulative, total_risk, start = 0.0, 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end]["confidence"] == ordered[start]["confidence"]:
            end += 1
        errors = sum(r["index"] != r["target_index"] for r in ordered[start:end])
        for k in range(1, end - start + 1):
            total_risk += (cumulative + k * errors / (end - start)) / (start + k)
        cumulative += errors
        start = end
    n = len(rows)
    result = dict(n=n, accuracy=1 - cumulative / n,
                  nll=sum(r["nll"] for r in rows) / n,
                  brier=sum(sum((p - (j == r["target_index"])) ** 2 for j, p in enumerate(r["probabilities"])) for r in rows) / n,
                  aurc=total_risk / n,
                  augrc=augrc(rows))
    ordinal = [abs(r["expected_value"] - r["values"][r["target_index"]]) for r in rows if r["type"] == "ordinal"]
    if ordinal:
        result.update(ordinal_n=len(ordinal), ordinal_mae=sum(ordinal) / len(ordinal))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite metric")
    return result


def reports(rows):
    return {"overall": report(rows), "by_type": {t: report([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})},
            "by_source": {s: report([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})},
            "definitions": {"aurc": "mean error risk at coverages 1/n..1; exact confidence ties averaged over all within-tie permutations",
                            "augrc": "area under generalized risk-coverage curve; pairwise failure-vs-correct confidence comparison, bounded to [0, 0.5]",
                            "brier": "sum of squared class-probability errors, then mean over rows",
                            "ordinal_mae": "absolute error of expected value, averaged over ordinal rows"}}
