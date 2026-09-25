#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-/home-local/sbiswas/experiments/qwen-webarena/.venv/bin/uv}"
LOG="$ROOT/runs/eval-duration-sweep.log"

run_eval() {
    local arm="$1"
    local epoch="$2"
    local gpu="$3"
    local config="x2-${arm}-${epoch}epoch.yaml"
    local checkpoint="runs/x2-${arm}-${epoch}epoch/checkpoint"
    local output="runs/x2-${arm}-${epoch}epoch-test-raw"
    local log="runs/x2-${arm}-${epoch}epoch-test-console.log"
    if [[ -f "$output/report.json" ]]; then
        printf '%s already evaluated: %s\n' "$(date -Is)" "$output" >> "$LOG"
        return
    fi
    if [[ ! -d "$checkpoint" ]]; then
        printf '%s missing checkpoint: %s\n' "$(date -Is)" "$checkpoint" >> "$LOG"
        return 1
    fi
    printf '%s evaluating %s on GPU %s\n' "$(date -Is)" "$output" "$gpu" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$gpu" "$UV" run --env-file .env --no-sync decisions evaluate \
        --config "packages/modernbert-decisions/configs/$config" \
        --data data/kev-decision-v7/test.jsonl \
        --split test \
        --checkpoint "$checkpoint" \
        --output "$output" >> "$log" 2>&1
    printf '%s completed %s\n' "$(date -Is)" "$output" >> "$LOG"
}

printf '%s evaluation sweep started\n' "$(date -Is)" >> "$LOG"
for spec in "0:2" "1:5" "3:10"; do
    (
        gpu="${spec%%:*}"
        epoch="${spec##*:}"
        for arm in ce aurc-only ce-aurc-mix; do
            run_eval "$arm" "$epoch" "$gpu"
        done
    ) &
done
wait
printf '%s evaluation sweep finished\n' "$(date -Is)" >> "$LOG"
