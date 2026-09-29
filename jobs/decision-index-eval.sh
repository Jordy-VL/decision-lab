#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    cat <<'EOF'
Usage:
  jobs/decision-index-eval.sh CHECKPOINT [sample|full] [RUN_NAME]

Runs the local Decision Index engine and scorer. Defaults to a 20-request
diagnostic sample; use "full" to evaluate the complete installed suite.

Environment:
  PYTHON                         Python with project dependencies and decision_index
  DECISION_INDEX_ROOT            Optional Decision Index source checkout
  DECISION_INDEX_SUITE_DIR       Suite directory (default: ./suite-0.2)
  DECISION_INDEX_EDITION         Suite edition (default: 0.2.1)
  DECISION_INDEX_SAMPLE_N        Diagnostic sample size (default: 20)
  DECISION_INDEX_DEVICE          auto, cuda or cpu (default: auto)
  DECISION_INDEX_MAX_LENGTH      Tokenizer limit (default: 2048)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

CHECKPOINT="$(realpath -- "${1%/}")"
MODE="${2:-sample}"
if [[ "$MODE" != "sample" && "$MODE" != "full" ]]; then
    echo "mode must be 'sample' or 'full'" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT/model.json" ]]; then
    echo "project-format checkpoint metadata not found: $CHECKPOINT/model.json" >&2
    exit 1
fi

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON (set PYTHON to the project/kit environment)" >&2
    exit 1
fi

PYTHONPATH_ENTRIES=(
    "$ROOT/scripts"
    "$ROOT/packages/modernbert-decisions"
)
if [[ -n "${DECISION_INDEX_ROOT:-}" ]]; then
    PYTHONPATH_ENTRIES+=("$DECISION_INDEX_ROOT")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:$PYTHONPATH}"

EDITION="${DECISION_INDEX_EDITION:-0.2.1}"
SUITE_DIR="${DECISION_INDEX_SUITE_DIR:-$ROOT/suite-0.2}"
SAMPLE_N="${DECISION_INDEX_SAMPLE_N:-20}"
DEVICE="${DECISION_INDEX_DEVICE:-auto}"
MAX_LENGTH="${DECISION_INDEX_MAX_LENGTH:-2048}"

if [[ "$MODE" == "sample" && ! "$SAMPLE_N" =~ ^[1-9][0-9]*$ ]]; then
    echo "DECISION_INDEX_SAMPLE_N must be a positive integer" >&2
    exit 2
fi
if [[ ! "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "DECISION_INDEX_MAX_LENGTH must be a positive integer" >&2
    exit 2
fi

if ! "$PYTHON" -c 'import decision_index, torch, transformers' >/dev/null 2>&1; then
    echo "Could not import decision_index, torch and transformers using: $PYTHON" >&2
    echo "Install the Decision Index kit into this environment or set DECISION_INDEX_ROOT to its source checkout." >&2
    exit 1
fi
if [[ ! -d "$SUITE_DIR" ]]; then
    echo "Decision Index suite directory not found: $SUITE_DIR" >&2
    echo "Download it with '$PYTHON -m decision_index suite download --edition $EDITION --dir \"$SUITE_DIR\"' or set DECISION_INDEX_SUITE_DIR." >&2
    exit 1
fi

RUN_NAME="${3:-$(basename "$CHECKPOINT")-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, '.', '_' and '-'" >&2
    exit 2
fi
OUT="$ROOT/runs/decision-index-$MODE-$RUN_NAME"
if [[ -e "$OUT" ]]; then
    echo "output directory already exists: $OUT" >&2
    echo "Choose a new RUN_NAME to avoid mixing results from different checkpoints." >&2
    exit 1
fi
mkdir -p "$OUT"
exec > >(tee "$OUT/console.log") 2>&1

COMMON_ARGS=(
    --engine jev_decision_index_engine:ModernBERTDecisionEngine
    --option "checkpoint=$CHECKPOINT"
    --option "device=$DEVICE"
    --option "max_length=$MAX_LENGTH"
    --edition "$EDITION"
    --suite-dir "$SUITE_DIR"
    --out "$OUT"
    --compact
)

if [[ "$MODE" == "sample" ]]; then
    ROWS="$OUT/sample-rows.jsonl.gz"
    "$PYTHON" -m decision_index suite sample \
        --edition "$EDITION" \
        --suite-dir "$SUITE_DIR" \
        --n "$SAMPLE_N" \
        --out "$ROWS"
    "$PYTHON" -m decision_index pipeline "${COMMON_ARGS[@]}" --rows "$ROWS"
else
    "$PYTHON" -m decision_index pipeline "${COMMON_ARGS[@]}"
fi

printf '\nDecision Index run finished.\n'
printf 'Output: %s\n' "$OUT"
printf 'Metrics: %s/scores.json\n' "$OUT"
printf 'Per-benchmark summary: %s/benchmark-summary.json\n' "$OUT"
printf 'Index details: %s/index.json\n' "$OUT"
if [[ "$MODE" == "sample" ]]; then
    printf 'This is a diagnostic sample, not a full-suite leaderboard score.\n'
fi
