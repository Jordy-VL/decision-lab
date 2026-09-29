#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
UV="${UV:-/home-local/sbiswas/experiments/qwen-webarena/.venv/bin/uv}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"

if [[ ! -x "$UV" ]]; then
    echo "uv executable not found: $UV" >&2
    exit 1
fi

if [[ ! -f .env ]]; then
    echo "missing project .env file" >&2
    exit 1
fi

IFS=, read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
    echo "GPU_LIST must contain exactly four comma-separated device indices" >&2
    exit 2
fi
declare -A seen_gpus=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_gpus[$gpu]:-}" ]]; then
        echo "GPU_LIST must contain four distinct numeric device indices" >&2
        exit 2
    fi
    seen_gpus["$gpu"]=1
done

RUNS=(
    x2-ce-10epoch-linear-1e5
    x2-aurc-only-10epoch-linear-1e5
    x2-candidate-masks-ce-10epoch-linear-1e5
    x2-candidate-masks-aurc-only-10epoch-linear-1e5
)
CONFIGS=(
    packages/modernbert-decisions/configs/x2-ce-10epoch-linear-1e5.yaml
    packages/modernbert-decisions/configs/x2-aurc-only-10epoch-linear-1e5.yaml
    packages/modernbert-decisions/configs/x2-candidate-masks-ce-10epoch-linear-1e5.yaml
    packages/modernbert-decisions/configs/x2-candidate-masks-aurc-only-10epoch-linear-1e5.yaml
)

for i in "${!RUNS[@]}"; do
    run="runs/${RUNS[$i]}"
    config="${CONFIGS[$i]}"
    if [[ ! -d "$run" || ! -f "$run/metadata.json" ]]; then
        echo "expected completed run directory not found: $run" >&2
        exit 1
    fi
    status="$(jq -r '.status // "unknown"' "$run/metadata.json")"
    if [[ "$status" != "completed" ]]; then
        echo "refusing to archive $run with status=$status" >&2
        exit 1
    fi
    if [[ ! -f "$config" ]]; then
        echo "config not found: $config" >&2
        exit 1
    fi
done

archive="runs/archive/linear-augrc-reselection-$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "$archive" ]]; then
    echo "archive destination already exists: $archive" >&2
    exit 1
fi
mkdir -p "$archive/logs"
for run in "${RUNS[@]}"; do
    mv "runs/$run" "$archive/$run"
    log="runs/$run-console.log"
    if [[ -f "$log" ]]; then
        mv "$log" "$archive/logs/"
    fi
done

printf 'archive=%s\n' "$archive" > "$archive/launch-manifest.txt"
printf 'checkpoint_selection=development_augrc\nretained_trainer_checkpoints=4_per_run\n' \
    >> "$archive/launch-manifest.txt"
for i in "${!RUNS[@]}"; do
    gpu="${GPUS[$i]}"
    config="${CONFIGS[$i]}"
    log="runs/${RUNS[$i]}-console.log"
    nohup env -u CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="$gpu" "$UV" run \
        --env-file .env --no-sync decisions train --config "$config" \
        > "$log" 2>&1 < /dev/null &
    pid=$!
    printf 'run=%s gpu=%s pid=%s config=%s log=%s\n' \
        "${RUNS[$i]}" "$gpu" "$pid" "$config" "$log" \
        | tee -a "$archive/launch-manifest.txt"
done

printf 'Archived prior outputs under %s and launched four 10-epoch runs.\n' "$archive"
