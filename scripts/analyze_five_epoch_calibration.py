"""Analyze historical NLL-selected five-epoch exports without model inference."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from compare_runs import load, paired, scored, fit_temperature, summary, threshold_result

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/modernbert-decisions"))
from decisions.metrics import augrc

ARMS = {"CE": "x2-ce-5epoch", "AURC-only": "x2-aurc-only-5epoch",
        "CE+AURC": "x2-ce-aurc-mix-5epoch"}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(rows):
    # Recompute indexes from scored probabilities, rather than trusting the export.
    normalized = [{**r, "index": max(range(len(r["probabilities"])), key=r["probabilities"].__getitem__)} for r in rows]
    return {**summary(normalized), "augrc": augrc(normalized)}


def run(base, minimum, fit_path):
    require(minimum > 0, "minimum accepted count must be positive")
    inputs, provenance, hashes = {}, {}, {}
    for name, stem in ARMS.items():
        train, test_dir = base / stem, base / (stem + "-test-raw")
        cal_dir = train / "evaluations/best/calibration"
        cm, tm, state = read(cal_dir / "metadata.json"), read(test_dir / "metadata.json"), read(train / "trainer_state.json")
        parent = read(train / "metadata.json")
        require(cm["checkpoint"] == tm["initialized_from"], name + ": different checkpoint export paths")
        require(tm["parent_metadata"] == parent, name + ": test parent metadata differs")
        require(cm["best_model_checkpoint"] == state["best_model_checkpoint"], name + ": selected checkpoint differs")
        best_step = int(state["best_model_checkpoint"].rsplit("-", 1)[1])
        evaluations = [x for x in state["log_history"] if "eval_nll" in x]
        selected = [x for x in evaluations if x["step"] == best_step]
        require(len(selected) == 1 and math.isclose(selected[0]["eval_nll"], state["best_metric"], abs_tol=1e-8), name + ": selector not identified as NLL")
        require(math.isclose(min(x["eval_nll"] for x in evaluations), state["best_metric"], abs_tol=1e-8), name + ": selected NLL is not minimum")
        splits = {}
        for split, directory in (("calibration", cal_dir), ("test", test_dir)):
            path = directory / "predictions.jsonl"
            rows = load(path)
            require(all(r["split"] == split and len(r["options"]) == len(r["logits"]) for r in rows), name + ": split/options invalid")
            require(len(rows) == (cm["rows"] if split == "calibration" else tm["selected_rows"]), name + ": metadata row count differs")
            splits[split] = rows
            hashes[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(not ({r["id"] for r in splits["calibration"]} & {r["id"] for r in splits["test"]}), name + ": calibration/test IDs overlap")
        require(not ({r["group_id"] for r in splits["calibration"]} & {r["group_id"] for r in splits["test"]}), name + ": calibration/test groups overlap")
        inputs[name] = splits
        provenance[name] = {"best_checkpoint": state["best_model_checkpoint"], "export": cm["checkpoint"],
                            "best_development_nll": state["best_metric"], "seed": parent["seed"],
                            "training_code_revision": parent["code_revision"], "test_code_revision": tm["code_revision"],
                            "data_sha256": parent["data_sha256"], "test_data_sha256": tm["data_sha256"],
                            "calibration_data_sha256": parent["additional_data_sha256"]["calibration"]}
    reference = inputs["CE"]
    for name, splits in inputs.items():
        for split in ("calibration", "test"):
            paired(reference[split], splits[split])
        for field in ("seed", "data_sha256", "test_data_sha256", "calibration_data_sha256"):
            require(provenance[name][field] == provenance["CE"][field], name + ": unmatched " + field)
    temperatures = {name: fit_temperature(splits["calibration"]) for name, splits in inputs.items()}
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist frozen calibration-only parameters before scoring any test outcomes.
    fit_path.write_text(json.dumps({"fit_split": "calibration", "temperatures": temperatures,
                                  "calibration_prediction_sha256": {p: h for p, h in hashes.items() if "/calibration/" in p}}, indent=2) + "\n")
    results = {}
    for name, splits in inputs.items():
        temperature = temperatures[name]
        for variant, temp in (("raw", 1.0), ("temperature", temperature)):
            cal, test = scored(splits["calibration"], temp), scored(splits["test"], temp)
            thresholds = []
            for risk in (.01, .05, .10):
                result = threshold_result(cal, test, risk, minimum)
                accepted = [r for r in test if result["threshold"] is not None and r["confidence"] >= result["threshold"]]
                result["errors"] = sum(r["error"] for r in accepted)
                if "calibration" in result:
                    result["calibration"]["errors"] = round(result["calibration"]["risk"] * result["calibration"]["accepted"])
                thresholds.append(result)
            results[name + "/" + variant] = {"temperature": temp, "calibration": aggregate(cal), "test": aggregate(test),
                                              "fixed_threshold_test": thresholds}
    return {"selection": "historical development-NLL-selected, seed 17, five-epoch budget; not five-epoch terminal weights",
            "provenance_status": "Export paths, parent metadata, trainer best checkpoint and minimum logged development NLL agree; weights were not downloaded or content-hashed.",
            "definitions": {"temperature": "One positive scalar per arm; calibration NLL only, log(T) bounded [-3,3], existing 64-iteration search",
                            "ece": "10 equal-width confidence bins", "brier": "mean sum of squared class errors",
                            "aurc": "mean selective risk at ranks 1/n..1; exact ties averaged",
                            "augrc": "repository pairwise failure/correct formula; exact ties receive half credit",
                            "threshold": "Maximum calibration coverage at empirical risk <= budget and accepted >= minimum; freeze confidence >= threshold on test; no statistical risk guarantee"},
            "minimum_calibration_accepted": minimum, "provenance": provenance, "prediction_sha256": hashes, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/remote-review-20260930/runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-accepted", type=int, default=30)
    args = parser.parse_args()
    report = run(args.runs, args.min_accepted, args.out.with_name(args.out.stem + "-temperatures.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, result in report["results"].items():
        print(name, "T=", round(result["temperature"], 6), result["test"])


if __name__ == "__main__":
    main()
