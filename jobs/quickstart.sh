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
  jobs/quickstart.sh prepare-test
  jobs/quickstart.sh check
  jobs/quickstart.sh smoke-upload
  jobs/quickstart.sh train-ce
  jobs/quickstart.sh train-ce-2epoch
  jobs/quickstart.sh train-ce-5epoch
  jobs/quickstart.sh train-ce-10epoch
  jobs/quickstart.sh train-ce-10epoch-linear-1e5
  jobs/quickstart.sh train-aurc-only-2epoch
  jobs/quickstart.sh train-aurc-only-5epoch
  jobs/quickstart.sh train-aurc-only-10epoch
  jobs/quickstart.sh train-aurc-only-10epoch-linear-1e5
  jobs/quickstart.sh train-augrc-only-2epoch
  jobs/quickstart.sh train-augrc-only-5epoch
  jobs/quickstart.sh train-augrc-only-10epoch
  jobs/quickstart.sh train-augrc-only-2epoch-linear-1e5
  jobs/quickstart.sh train-augrc-only-5epoch-linear-1e5
  jobs/quickstart.sh train-augrc-only-10epoch-linear-1e5
  jobs/quickstart.sh train-ce-aurc-mix-2epoch
  jobs/quickstart.sh train-ce-aurc-mix-5epoch
  jobs/quickstart.sh train-ce-aurc-mix-10epoch
  jobs/quickstart.sh train-aurc-only
  jobs/quickstart.sh train-ce-aurc-mix
  jobs/quickstart.sh eval-ce
  jobs/quickstart.sh eval-aurc-only
  jobs/quickstart.sh eval-ce-aurc-mix

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
    prepare-test)
        "$UV" run --env-file .env --no-sync python scripts/prepare_kev.py --allow-test
        ;;
    check)
        "$UV" run --env-file .env --no-sync decisions check \
            --config packages/modernbert-decisions/configs/x2-ce.yaml
        ;;
    smoke-upload)
        exec "$UV" run --env-file .env --no-sync python scripts/smoke_hf_trainer.py \
            --single-batch --device auto --require-cuda --upload-hf
        ;;
    train-ce|train-ce-2epoch|train-ce-5epoch|train-ce-10epoch|train-ce-10epoch-linear-1e5|train-aurc-only|train-aurc-only-2epoch|train-aurc-only-5epoch|train-aurc-only-10epoch|train-aurc-only-10epoch-linear-1e5|train-augrc-only-2epoch|train-augrc-only-5epoch|train-augrc-only-10epoch|train-augrc-only-2epoch-linear-1e5|train-augrc-only-5epoch-linear-1e5|train-augrc-only-10epoch-linear-1e5|train-ce-aurc-mix|train-ce-aurc-mix-2epoch|train-ce-aurc-mix-5epoch|train-ce-aurc-mix-10epoch)
        case "$1" in
            train-ce) config="x2-ce.yaml"; log="x2-ce-console.log" ;;
            train-ce-2epoch) config="x2-ce-2epoch.yaml"; log="x2-ce-2epoch-console.log" ;;
            train-ce-5epoch) config="x2-ce-5epoch.yaml"; log="x2-ce-5epoch-console.log" ;;
            train-ce-10epoch) config="x2-ce-10epoch.yaml"; log="x2-ce-10epoch-console.log" ;;
            train-ce-10epoch-linear-1e5) config="x2-ce-10epoch-linear-1e5.yaml"; log="x2-ce-10epoch-linear-1e5-console.log" ;;
            train-aurc-only) config="x2-aurc-only.yaml"; log="x2-aurc-only-console.log" ;;
            train-aurc-only-2epoch) config="x2-aurc-only-2epoch.yaml"; log="x2-aurc-only-2epoch-console.log" ;;
            train-aurc-only-5epoch) config="x2-aurc-only-5epoch.yaml"; log="x2-aurc-only-5epoch-console.log" ;;
            train-aurc-only-10epoch) config="x2-aurc-only-10epoch.yaml"; log="x2-aurc-only-10epoch-console.log" ;;
            train-aurc-only-10epoch-linear-1e5) config="x2-aurc-only-10epoch-linear-1e5.yaml"; log="x2-aurc-only-10epoch-linear-1e5-console.log" ;;
            train-augrc-only-2epoch) config="x2-augrc-only-2epoch.yaml"; log="x2-augrc-only-2epoch-console.log" ;;
            train-augrc-only-5epoch) config="x2-augrc-only-5epoch.yaml"; log="x2-augrc-only-5epoch-console.log" ;;
            train-augrc-only-10epoch) config="x2-augrc-only-10epoch.yaml"; log="x2-augrc-only-10epoch-console.log" ;;
            train-augrc-only-2epoch-linear-1e5) config="x2-augrc-only-2epoch-linear-1e5.yaml"; log="x2-augrc-only-2epoch-linear-1e5-console.log" ;;
            train-augrc-only-5epoch-linear-1e5) config="x2-augrc-only-5epoch-linear-1e5.yaml"; log="x2-augrc-only-5epoch-linear-1e5-console.log" ;;
            train-augrc-only-10epoch-linear-1e5) config="x2-augrc-only-10epoch-linear-1e5.yaml"; log="x2-augrc-only-10epoch-linear-1e5-console.log" ;;
            train-ce-aurc-mix) config="x2-ce-aurc-mix.yaml"; log="x2-ce-aurc-mix-console.log" ;;
            train-ce-aurc-mix-2epoch) config="x2-ce-aurc-mix-2epoch.yaml"; log="x2-ce-aurc-mix-2epoch-console.log" ;;
            train-ce-aurc-mix-5epoch) config="x2-ce-aurc-mix-5epoch.yaml"; log="x2-ce-aurc-mix-5epoch-console.log" ;;
            train-ce-aurc-mix-10epoch) config="x2-ce-aurc-mix-10epoch.yaml"; log="x2-ce-aurc-mix-10epoch-console.log" ;;
        esac
        mkdir -p runs
        exec "$UV" run --env-file .env --no-sync decisions train \
            --config "packages/modernbert-decisions/configs/$config" \
            2>&1 | tee "runs/$log"
        ;;
    eval-ce|eval-aurc-only|eval-ce-aurc-mix)
        case "$1" in
            eval-ce)
                config="x2-ce.yaml"
                checkpoint="runs/x2-ce/checkpoint"
                output="runs/x2-ce-test-raw"
                log="x2-ce-test-console.log"
                ;;
            eval-aurc-only)
                config="x2-aurc-only.yaml"
                checkpoint="runs/x2-aurc-only/checkpoint"
                output="runs/x2-aurc-only-test-raw"
                log="x2-aurc-only-test-console.log"
                ;;
            eval-ce-aurc-mix)
                config="x2-ce-aurc-mix.yaml"
                checkpoint="runs/x2-ce-aurc-mix/checkpoint"
                output="runs/x2-ce-aurc-mix-test-raw"
                log="x2-ce-aurc-mix-test-console.log"
                ;;
        esac
        if [[ ! -d "$checkpoint" ]]; then
            echo "checkpoint not found: $checkpoint" >&2
            exit 1
        fi
        if [[ ! -f data/kev-decision-v7/test.jsonl ]]; then
            echo "test data not found; run jobs/quickstart.sh prepare-test first" >&2
            exit 1
        fi
        mkdir -p runs
        exec "$UV" run --env-file .env --no-sync decisions evaluate \
            --config "packages/modernbert-decisions/configs/$config" \
            --data data/kev-decision-v7/test.jsonl \
            --split test \
            --checkpoint "$checkpoint" \
            --output "$output" \
            2>&1 | tee "runs/$log"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
