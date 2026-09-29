"""Plot evaluation metrics for the fixed-slot linear 10-epoch runs."""
import argparse
import csv
import json
from pathlib import Path

RUNS = {
    "AUGRC / fixed-slot": "x2-augrc-only-10epoch-linear-1e5",
    "CE / fixed-slot": "x2-ce-10epoch-linear-1e5",
    "AURC / fixed-slot": "x2-aurc-only-10epoch-linear-1e5",
}
METRICS = (
    ("eval_loss", "Evaluation loss"),
    ("eval_nll", "NLL"),
    ("eval_accuracy", "Accuracy"),
    ("eval_brier", "Brier score"),
    ("eval_aurc", "AURC"),
    ("eval_augrc", "AUGRC"),
    ("eval_runtime", "Evaluation runtime (s)"),
    ("eval_samples_per_second", "Evaluation samples / second"),
    ("eval_steps_per_second", "Evaluation steps / second"),
)


def latest_trainer_state(run_dir: Path) -> Path:
    checkpoints = []
    for path in (run_dir / "trainer").glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(path.parent.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"no saved Trainer history found under {run_dir / 'trainer'}")
    return max(checkpoints, key=lambda item: item[0])[1]


def load_history(root: Path) -> list[dict]:
    rows = []
    for label, run_name in RUNS.items():
        run_dir = root / "runs" / run_name
        state_path = latest_trainer_state(run_dir)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        evaluations = [
            item for item in state["log_history"]
            if "eval_augrc" in item and "step" in item and "epoch" in item
        ]
        if not evaluations:
            raise ValueError(f"no evaluation metrics in {state_path}")
        status_path = run_dir / "metadata.json"
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown")
        for evaluation in evaluations:
            row = {
                "run": label,
                "run_name": run_name,
                "status": status,
                "checkpoint_history_step": int(state["global_step"]),
                "step": int(evaluation["step"]),
                "epoch": float(evaluation["epoch"]),
            }
            row.update({
                metric: float(evaluation[metric])
                for metric, _ in METRICS
                if metric in evaluation
            })
            missing = [metric for metric, _ in METRICS if metric not in row]
            if missing:
                raise ValueError(f"{state_path}: evaluation at step {row['step']} missing {missing}")
            rows.append(row)
        print(
            f"{label}: {len(evaluations)} evaluations through step "
            f"{max(row['step'] for row in rows if row['run'] == label)} "
            f"(run status: {status}; source: {state_path.relative_to(root)})"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/linear-10epoch-1e5-fixed-slot-evaluation-curves.png"),
        help="PNG path, relative to --root unless absolute",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    csv_path = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_history(root)
    statuses = {label: next(row["status"] for row in rows if row["run"] == label) for label in RUNS}
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(RUNS, plt.get_cmap("tab10").colors))
    figure, axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, METRICS):
        for label in RUNS:
            series = sorted(
                (row for row in rows if row["run"] == label),
                key=lambda row: row["step"],
            )
            axis.plot(
                [row["step"] for row in series],
                [row[metric] for row in series],
                label=f"{label} ({statuses[label]})",
                color=colors[label],
                linewidth=1.5,
            )
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.grid(True, alpha=0.25)
    axes[0, 2].set_ylabel("Fraction")
    axes[1, 0].set_ylabel("Score")
    figure.suptitle(
        "10-epoch linear 1e-5 runs — saved development evaluations\n"
        "eval_loss is objective-specific; compare NLL, Brier, AURC and AUGRC across objectives",
        fontsize=14,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Plot: {output}")
    print(f"Data: {csv_path}")


if __name__ == "__main__":
    main()
