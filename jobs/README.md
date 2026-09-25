# Job quickstart

Research training runs use `uab-gpu`, with one experiment per CUDA device. From
the repository root, obtain the scheduler allocation first, then use:

```sh
jobs/quickstart.sh prepare
jobs/quickstart.sh check
```

Prepare the locked test partition only after checkpoint and calibration
choices are frozen:

```sh
jobs/quickstart.sh prepare-test
```

Verify the project-scoped Hugging Face login:

```sh
jobs/quickstart.sh hf-whoami
```

This uses:

```text
HF_HOME=/home-local/sbiswas/.cache/huggingface-decision-lab
```

Training does not upload artifacts to Hugging Face. Outputs remain in the
local `runs/` directory unless a separate upload step is performed.

Run the one-update GPU smoke and upload its verified temporary artifacts
explicitly:

```sh
CUDA_VISIBLE_DEVICES=0 \
jobs/quickstart.sh smoke-upload
```

This uploads to the private `jordyvl/decision-lab-smoke` model repository.
Upload is not part of the full training commands below.

Start two matched arms concurrently in separate terminals:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce
CUDA_VISIBLE_DEVICES=1 jobs/quickstart.sh train-aurc-only
```

For the initial longer-run duration sweep, run the CE arm for 2, 5 and
10 epochs in separate output directories. Start two at a time on available
GPUs and queue the third after one device is free:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce-2epoch
CUDA_VISIBLE_DEVICES=1 jobs/quickstart.sh train-ce-5epoch
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce-10epoch
```

Select the duration using development AURC before launching matched
multi-epoch AURC arms. These configs keep the v7 data, `cls-index-v1`
architecture, optimizer and scheduler fixed.

The original sweep uses `2e-5` peak learning rate, 5% warmup and cosine
decay. Because the historical 10-epoch runs reached their best development
NLL early,
matched slower linear-decay probes are also prepared at `1e-5` with 10%
warmup:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce-10epoch-linear-1e5
CUDA_VISIBLE_DEVICES=1 jobs/quickstart.sh train-aurc-only-10epoch-linear-1e5
```

These probes are separate from the completed sweep and should use fresh
output directories. They are intended to test schedule sensitivity, not to
replace the original results.

The matched AURC-only duration configs are also ready, but intentionally not
started yet:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-aurc-only-2epoch
CUDA_VISIBLE_DEVICES=1 jobs/quickstart.sh train-aurc-only-5epoch
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-aurc-only-10epoch
```

The matched CE+AURC-mix duration configs are also ready. They use
`aurc_lambda: 0.5` and are launched only after both the corresponding CE and
AURC-only runs have completed:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce-aurc-mix-2epoch
CUDA_VISIBLE_DEVICES=1 jobs/quickstart.sh train-ce-aurc-mix-5epoch
CUDA_VISIBLE_DEVICES=3 jobs/quickstart.sh train-ce-aurc-mix-10epoch
```

Run the remaining arm when either device is free:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh train-ce-aurc-mix
```

Evaluate the selected best checkpoints on the locked test partition:

```sh
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh eval-ce
CUDA_VISIBLE_DEVICES=1 jobs/quickstart.sh eval-aurc-only
CUDA_VISIBLE_DEVICES=0 jobs/quickstart.sh eval-ce-aurc-mix
```

Evaluation writes raw test predictions and reports to separate directories:

```text
runs/x2-ce-test-raw/
runs/x2-aurc-only-test-raw/
runs/x2-ce-aurc-mix-test-raw/
```

The console logs are:

```text
runs/x2-ce-test-console.log
runs/x2-aurc-only-test-console.log
runs/x2-ce-aurc-mix-test-console.log
```

These commands do not fit calibration or thresholds. Fit those on the
calibration partition first, then apply the frozen transformations to the
saved test logits. Evaluation output directories must be new or empty.

Each command displays output and writes a console log:

```sh
tail -f runs/x2-ce-console.log
tail -f runs/x2-aurc-only-console.log
tail -f runs/x2-ce-aurc-mix-console.log
```

Structured artifacts are written below each run directory, including
`metadata.json`, `trainer_log_history.json`, `training.json`, checkpoints, and
development/calibration predictions and logits. `metadata.json` reports
`running`, `completed`, or `failed`.

## Historical Modal job status

From the repository root in WSL/Linux:

```sh
./jobs/modal-status.sh
```

This read-only command lists apps and running containers as JSON using the active Modal profile and environment. It does not start or stop jobs. Stopped apps with zero tasks and an empty container list indicate no active compute in that environment. Persistent Volumes remain stored separately.

## Verified findings — 2026-09-22

- Profiling app `ap-xqLBFwGmHPi6oq9PAnzbgQ`: stopped at 18:45:27 Europe/Paris, zero tasks.
- Training app `ap-OO22BxiK87M5jisChJeEOI`: stopped at 19:31:30 Europe/Paris, zero tasks.
- The running-container query returned `[]`.
- The `decision-lab-profile` Volume still contains `hf`, `runs`, and `profile.json`. The profiling report also exists locally at `runs/modal-profile/profile.json`; the profiling job intentionally saved no trained checkpoint.
- Profiling logs are retained: steps 31 and 32 appear at 18:45:21 Europe/Paris, followed by completion at 18:45:24. A dashboard time filter ending at 18:44:37 excludes those final logs. Disable Live and select a historical range covering the run.
- The full training attempt was cancelled before its final checkpoint. See [the retained outcome and evidence](../docs/results/ce-20260922-170803.md); its cancellation cause remains unresolved.

Retrieve the profiling logs without dashboard time filters:

```sh
uv run modal app logs ap-xqLBFwGmHPi6oq9PAnzbgQ --timestamps
```

These observations describe the check on this date; rerun the status script for current state.
