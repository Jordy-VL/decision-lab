"""Plot saved development evaluations as an offline, interactive Plotly report."""
import argparse
import csv
import json
from pathlib import Path

RUNS = {
    "AUGRC / fixed-slot": "x2-augrc-only-10epoch-linear-1e5",
    "CE / fixed-slot": "x2-ce-10epoch-linear-1e5",
    "AURC / fixed-slot": "x2-aurc-only-10epoch-linear-1e5",
}
CANDIDATE_RUNS = {
    "CE / candidate-masks": "x2-candidate-masks-ce-10epoch-linear-1e5",
    "AURC / candidate-masks": "x2-candidate-masks-aurc-only-10epoch-linear-1e5",
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


def load_history(root: Path, include_candidates: bool = False) -> list[dict]:
    rows = []
    runs = {**RUNS, **(CANDIDATE_RUNS if include_candidates else {})}
    for label, run_name in runs.items():
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


def load_csv(path: Path) -> list[dict]:
    """Reuse exported evaluations when the original checkpoints are remote."""
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no evaluations in {path}")
    for row in rows:
        for key in ("step", "checkpoint_history_step"):
            row[key] = int(row[key])
        for key in ("epoch", *(metric for metric, _ in METRICS)):
            row[key] = float(row[key])
    return rows


def build_figure(rows: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = list(dict.fromkeys(row["run"] for row in rows))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728")
    figure = make_subplots(rows=3, cols=3, subplot_titles=[title for _, title in METRICS],
                           horizontal_spacing=0.07, vertical_spacing=0.11)
    for index, (metric, title) in enumerate(METRICS):
        for run_index, label in enumerate(labels):
            series = sorted((row for row in rows if row["run"] == label),
                            key=lambda row: row["step"])
            statuses = ", ".join(dict.fromkeys(row["status"] for row in series))
            figure.add_trace(go.Scatter(
                x=[row["step"] for row in series],
                y=[row[metric] for row in series],
                customdata=[[row["epoch"], row["checkpoint_history_step"]] for row in series],
                mode="lines", name=f"{label} ({statuses})", legendgroup=label,
                showlegend=index == 0, line=dict(color=colors[run_index % len(colors)], width=2),
                hovertemplate=("Step %{x:,}<br>Epoch %{customdata[0]:.3f}<br>"
                               + title + ": %{y:.6g}<br>Saved history step %{customdata[1]:,}"
                               + "<extra>%{fullData.name}</extra>"),
            ), row=index // 3 + 1, col=index % 3 + 1)
    figure.update_xaxes(title_text="Training step", matches="x", showgrid=True)
    figure.update_yaxes(showgrid=True)
    figure.update_layout(
        template="plotly_white", height=1050, hovermode="x unified",
        title=dict(text="10-epoch linear 1e-5 runs — saved development evaluations"
                   "<br><sup>eval_loss is objective-specific; compare NLL, Brier, AURC and AUGRC. "
                   "Drag to zoom; double-click to reset; click a legend entry to toggle a run.</sup>",
                   font=dict(size=18)),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", groupclick="togglegroup"),
        margin=dict(t=110, b=110),
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-candidates", action="store_true",
                        help="Include candidate-mask CE and AURC runs from saved histories")
    parser.add_argument("--input-csv", type=Path,
                        help="Reuse an exported CSV instead of reading run checkpoints")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="HTML path relative to --root; PNG/SVG/PDF also writes an HTML sibling")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.input_csv and args.include_candidates:
        parser.error("--include-candidates selects histories; --input-csv uses the runs already in the CSV")
    if args.output is None:
        name = "linear-10epoch-1e5" + ("" if args.include_candidates else "-fixed-slot")
        args.output = Path("reports") / f"{name}-evaluation-curves.html"
    output = args.output if args.output.is_absolute() else root / args.output
    if output.suffix.lower() not in (".html", ".png", ".svg", ".pdf"):
        parser.error("--output must end in .html, .png, .svg or .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.input_csv:
        source = args.input_csv if args.input_csv.is_absolute() else root / args.input_csv
        rows = load_csv(source)
    else:
        rows = load_history(root, include_candidates=args.include_candidates)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure = build_figure(rows)
    html_path = output.with_suffix(".html")
    figure.write_html(html_path, include_plotlyjs=True, full_html=True,
                      config={"responsive": True, "displaylogo": False,
                              "toImageButtonOptions": {"format": "png", "scale": 2}})
    print(f"Interactive plot: {html_path}")
    print(f"Data: {csv_path}")
    if output.suffix.lower() != ".html":
        try:
            figure.write_image(output, width=1800, height=1050, scale=2)
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f"HTML and CSV saved; static export requires Kaleido and Chrome: {exc}\n")
        print(f"Static plot: {output}")


if __name__ == "__main__":
    main()
