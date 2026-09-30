#!/usr/bin/env bash
# Run this coordinator inside a scheduler allocation; use nohup for durable SSH sessions.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
usage() { echo "Usage: $0 --gpus INDEX[,INDEX...] [--dry-run]"; }
GPU_LIST=""
DRY_RUN=0
while (($#)); do
    case "$1" in
        --gpus) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; GPU_LIST="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Explicit numeric --gpus is required.' >&2; exit 2; }
IFS=, read -r -a GPUS <<< "$GPU_LIST"
((${#GPUS[@]} <= 6)) || { echo 'At most six GPUs for six runs.' >&2; exit 2; }
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
    [[ -z "${SEEN[$gpu]:-}" ]] || { echo 'Duplicate GPU index.' >&2; exit 2; }
    SEEN[$gpu]=1
done
RUNS=()
CONFIGS=()
for seed in 17 29; do
    for arm in ce aurc-only ce-aurc-mix; do
        name="x2-${arm}-5epoch-seed${seed}-augrc-selection"
        RUNS+=("$name")
        CONFIGS+=("packages/modernbert-decisions/configs/$name.yaml")
        [[ -f "${CONFIGS[-1]}" ]] || { echo "Missing config: ${CONFIGS[-1]}" >&2; exit 1; }
        [[ ! -e "runs/$name" ]] || { echo "Refusing existing output: runs/$name" >&2; exit 1; }
    done
done
for i in "${!RUNS[@]}"; do
    printf 'planned run=%s gpu=%s config=%s\n' "${RUNS[$i]}" "${GPUS[$((i % ${#GPUS[@]}))]}" "${CONFIGS[$i]}"
done
((DRY_RUN == 0)) || exit 0
UV="${UV:-uv}"
command -v "$UV" >/dev/null || { echo "uv not found: $UV" >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock is required.' >&2; exit 1; }
for split in train development calibration; do
    [[ -f "data/kev-decision-v7/$split.jsonl" ]] || { echo "Missing $split data." >&2; exit 1; }
done
# Lock all six outputs as one queue; no overwrites, archival, or automatic resume.
mkdir -p runs
# Reserve 16 GiB per run plus 20 GiB for transient checkpoint/export peaks.
required_kib=$(( (${#RUNS[@]} * 16 + 20) * 1024 * 1024 ))
available_kib="$(df -Pk runs | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || { echo 'Cannot determine available run storage.' >&2; exit 1; }
((available_kib >= required_kib)) || {
    echo "Insufficient run storage: need $required_kib KiB, available $available_kib KiB. No outputs removed." >&2
    exit 1
}
exec 9>runs/replication-five-epoch.lock
flock -n 9 || { echo 'A replication coordinator is already running.' >&2; exit 1; }
for name in "${RUNS[@]}"; do
    [[ ! -e "runs/$name" ]] || { echo "Refusing existing output: runs/$name" >&2; exit 1; }
done
LAUNCH="runs/replication-five-epoch-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$LAUNCH"
printf '%s\n' "$$" > "$LAUNCH/coordinator.pid"
printf 'checkpoint_selection=development_augrc\nseeds=17,29\nepochs=5\ngpus=%s\n' "$GPU_LIST" > "$LAUNCH/manifest.txt"
git rev-parse HEAD > "$LAUNCH/git-head.txt"
git status --short > "$LAUNCH/git-status.txt"
{
    printf '%s\0' "${CONFIGS[@]}" jobs/replicate-five-epoch.sh uv.lock pyproject.toml
    find packages/modernbert-decisions/decisions -maxdepth 1 -type f -name '*.py' -print0
    printf '%s\0' data/kev-decision-v7/{train,development,calibration}.jsonl
} | sort -z | xargs -0 sha256sum > "$LAUNCH/source-and-data.sha256"
ENV_ARGS=()
[[ ! -f .env ]] || ENV_ARGS=(--env-file .env)
worker() {
    local slot="$1" gpu="${GPUS[$1]}" i name pid code
    printf '%s\n' "$BASHPID" > "$LAUNCH/worker-$gpu.pid"
    for ((i=slot; i<${#RUNS[@]}; i+=${#GPUS[@]})); do
        name="${RUNS[$i]}"
        sha256sum --check --status "$LAUNCH/source-and-data.sha256" || { echo 'Source/data changed; queue stopped.' >&2; return 1; }
        [[ ! -e "runs/$name" ]] || { echo "Refusing existing output: runs/$name" >&2; return 1; }
        env CUDA_VISIBLE_DEVICES="$gpu" "$UV" run "${ENV_ARGS[@]}" --no-sync decisions train --config "${CONFIGS[$i]}" > "$LAUNCH/$name.log" 2>&1 < /dev/null &
        pid=$!
        printf '%s\n' "$pid" > "$LAUNCH/$name.pid"
        printf 'run=%s gpu=%s pid=%s started=%s\n' "$name" "$gpu" "$pid" "$(date -u +%FT%TZ)" >> "$LAUNCH/worker-$gpu.status"
        code=0
        wait "$pid" || code=$?
        printf 'run=%s exit=%s finished=%s\n' "$name" "$code" "$(date -u +%FT%TZ)" >> "$LAUNCH/worker-$gpu.status"
        ((code == 0)) || return "$code"
    done
}
PIDS=()
for slot in "${!GPUS[@]}"; do
    worker "$slot" > "$LAUNCH/worker-${GPUS[$slot]}.log" 2>&1 &
    PIDS+=("$!")
done
printf 'Queue metadata and logs: %s\n' "$LAUNCH"
code=0
for pid in "${PIDS[@]}"; do wait "$pid" || code=1; done
printf 'exit=%s finished=%s\n' "$code" "$(date -u +%FT%TZ)" >> "$LAUNCH/manifest.txt"
exit "$code"
