#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
LOG="$ROOT/runs/ce-aurc-mix-watcher.log"

status_of() {
    "$PYTHON" - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        print(json.load(handle).get("status", ""))
except FileNotFoundError:
    print("")
PY
}

printf '%s watcher started\n' "$(date -Is)" >> "$LOG"
while :; do
    all_done=1
    for spec in "2:0" "5:1" "10:3"; do
        epoch="${spec%%:*}"
        gpu="${spec##*:}"
        ce="$ROOT/runs/x2-ce-${epoch}epoch"
        aurc="$ROOT/runs/x2-aurc-only-${epoch}epoch"
        mix="$ROOT/runs/x2-ce-aurc-mix-${epoch}epoch"
        if test -e "$mix" || test -e "$mix.failed-to-launch"; then
            continue
        fi
        all_done=0
        ce_status="$(status_of "$ce/metadata.json")"
        aurc_status="$(status_of "$aurc/metadata.json")"
        if [[ "$ce_status" == failed || "$aurc_status" == failed ]]; then
            printf '%s %sepoch prerequisite failed; not launching mix\n' \
                "$(date -Is)" "$epoch" >> "$LOG"
            touch "$mix.failed-to-launch"
            continue
        fi
        if [[ "$ce_status" == completed && "$aurc_status" == completed ]]; then
            printf '%s launching mix %sepoch on GPU %s\n' \
                "$(date -Is)" "$epoch" "$gpu" >> "$LOG"
            (
                cd "$ROOT"
                CUDA_VISIBLE_DEVICES="$gpu" jobs/quickstart.sh \
                    "train-ce-aurc-mix-${epoch}epoch"
            ) >> "$LOG" 2>&1 &
        fi
    done
    if test "$all_done" -eq 1; then
        printf '%s watcher finished\n' "$(date -Is)" >> "$LOG"
        break
    fi
    sleep 60
done
