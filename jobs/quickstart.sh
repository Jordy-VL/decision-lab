#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV="${UV:-/home-local/sbiswas/experiments/qwen-webarena/.venv/bin/uv}"
HF_HOME="${HF_HOME:-/home-local/sbiswas/.cache/huggingface-decision-lab}"
export HF_HOME

if [[ ! -x "$UV" ]]; then
    echo "uv executable not found: $UV" >&2
    exit 1
fi

usage() {
    cat <<'EOF'
Usage:
  jobs/quickstart.sh hf-whoami
  jobs/quickstart.sh prepare
  jobs/quickstart.sh check
  jobs/quickstart.sh smoke-upload
  jobs/quickstart.sh train-ce
  jobs/quickstart.sh train-aurc-only
  jobs/quickstart.sh train-ce-aurc-mix

Set CUDA_VISIBLE_DEVICES per invocation to select a physical GPU, for example:
  CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce
EOF
}

case "${1:-}" in
    hf-whoami)
        "$UV" run --env-file .env --no-sync python -c \
            'from huggingface_hub import whoami; print(whoami())'
        ;;
    prepare)
        "$UV" run --env-file .env --no-sync python scripts/prepare_kev.py
        ;;
    check)
        "$UV" run --env-file .env --no-sync decisions check \
            --config packages/modernbert-decisions/configs/x2-ce.yaml
        ;;
    smoke-upload)
        exec "$UV" run --env-file .env --no-sync python scripts/smoke_hf_trainer.py \
            --single-batch --device auto --require-cuda --upload-hf
        ;;
    train-ce|train-aurc-only|train-ce-aurc-mix)
        case "$1" in
            train-ce) config="x2-ce.yaml"; log="x2-ce-console.log" ;;
            train-aurc-only) config="x2-aurc-only.yaml"; log="x2-aurc-only-console.log" ;;
            train-ce-aurc-mix) config="x2-ce-aurc-mix.yaml"; log="x2-ce-aurc-mix-console.log" ;;
        esac
        mkdir -p runs
        exec "$UV" run --env-file .env --no-sync decisions train \
            --config "packages/modernbert-decisions/configs/$config" \
            2>&1 | tee "runs/$log"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
