"""Reproduce the September 30 supplied-confidence review without model calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
from typing import List, Optional, Tuple

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.source / "confidence/evaluation.py"
    names = {"geifman_AURC", "_residuals_and_confidence", "auroc_failure",
             "augrc", "augrc_closed_form", "calculate_coverage_at_risk",
             "derive_per_field_tau", "compute_ecuas"}
    tree = ast.parse(evaluation.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in functions} == names
    namespace = dict(np=np, List=List, Optional=Optional, Tuple=Tuple)
    # Execute only inspected pure functions: no tracker imports, ECE downloads or APIs.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(evaluation), "exec"), namespace)
    opt = runpy.run_path(str(args.source / "confidence/threshold_optimization.py"))
    project = Path(__file__).resolve().parents[1]
    current = runpy.run_path(str(project / "packages/modernbert-decisions/decisions/metrics.py"))

    rng = np.random.default_rng(20260930)
    max_delta = 0.0
    for _ in range(1000):
        n = int(rng.integers(1, 65))
        correct = rng.integers(0, 2, n).tolist()
        confidence = (rng.integers(0, 6, n) / 5).tolist()
        rows = [dict(confidence=q, index=c, target_index=1) for c, q in zip(correct, confidence)]
        values = [namespace["augrc"](correct, confidence),
                  namespace["augrc_closed_form"](correct, confidence), current["augrc"](rows)]
        max_delta = max(max_delta, max(values) - min(values))
    assert max_delta < 1e-12

    aurc = namespace["geifman_AURC"]([0, 0], [0.9, 0.8])
    coverage, threshold = namespace["calculate_coverage_at_risk"](
        np.linspace(1, 0.05, 20), np.arange(1, 21), 0.05)
    derived = namespace["derive_per_field_tau"]([0] * 20, np.linspace(1, 0.05, 20).tolist())
    assert aurc == 0.75 and coverage == 0.05 and derived == 1.0

    field = opt["FieldInstance"]
    tied = {"good": {"f": [field(0.9, 1)]}, "bad": {"f": [field(0.9, 0)]}}
    chosen, (cov, risk), winner = opt["optimize_best"](tied, 0.0)
    assert cov == 1.0 and risk == 0.5 and winner == "doc-ranked"
    _, (greedy_cov, _) = opt["optimize_thresholds"](
        {"good": {"a": [field(0.9, 1)], "b": [field(0.8, 1)]}}, 0.0)
    assert greedy_cov == 0.0
    missing = opt["evaluate"]({"bad": {"f": [field(0.9, 0)]}}, {})
    assert missing[:2] == (1.0, 0.0)

    result = {
        "source": str(args.source),
        "sha256": {str(f.relative_to(args.source)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted(args.source.rglob("*.py")) if "__MACOSX" not in f.parts},
        "augrc_parity": {"cases": 1000, "seed": 20260930, "maximum_absolute_difference": max_delta},
        "legacy_aurc_all_wrong": {"observed": aurc, "correct": 1.0},
        "legacy_coverage_all_wrong": {"reported_coverage": coverage, "threshold": threshold,
                                      "actual_accepted_risk": 1.0},
        "derived_threshold_all_wrong": {"threshold": derived, "actual_accepted_risk": 1.0},
        "optimizer_tie_counterexample": {"target_risk": 0.0, "winner": winner,
                                         "thresholds": chosen, "coverage": cov, "risk": risk},
        "greedy_two_field_good_document": {"observed_coverage": greedy_cov, "feasible_coverage": 1.0},
        "missing_threshold_wrong_document": {"observed_coverage": missing[0], "observed_risk": missing[1]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
