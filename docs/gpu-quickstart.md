# First GPU setup and CE baseline

Run these commands in a Linux shell on an allocated GPU. Remote authentication is paused; the earlier SSH chat was located, but GPU access and capacity are not verified. No credentials belong in this repository.

## 1. Connect and transfer the code

From the Windows machine with the existing SSH config:

```powershell
ssh uab-gpu
```

Use the normal SSH password prompts if key authentication is unavailable. On a shared cluster, obtain a GPU through its scheduler/allocation process before training. The commands below do not request an allocation or choose a scheduler partition.

This repository has no published remote. A convenient transfer from local WSL is:

```sh
cd /home/jvl/tinker/decision-lab
git archive --format=tar.gz --output=/mnt/c/Users/jvl/Downloads/decision-lab.tar.gz HEAD
```

Then in Windows PowerShell (uses the Windows SSH configuration):

```powershell
scp C:\Users\jvl\Downloads\decision-lab.tar.gz uab-gpu:decision-lab.tar.gz
ssh uab-gpu
```

On the server, unpack into a new directory, not an existing checkout:

```sh
mkdir decision-lab-pilot
tar -xzf decision-lab.tar.gz -C decision-lab-pilot
cd decision-lab-pilot
nvidia-smi
```

The archive excludes ignored data, checkpoints, environments and secrets. It is a source snapshot, not a Git clone. Record the local `git rev-parse HEAD` alongside the remote run.

## 2. Install and check CUDA

Use an existing uv installation, or install uv in a private bootstrap environment:

```sh
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv
export PATH="$PWD/.bootstrap/bin:$PATH"
uv sync --locked --python 3.12 --extra train --extra report
uv run --no-sync python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check GPU allocation and driver/wheel compatibility'
print('GPU:', torch.cuda.get_device_name(0))
print('Free/total bytes:', torch.cuda.mem_get_info())
PY
```

The lock currently resolves a CUDA 13 PyTorch build on Linux. If it is incompatible with the server driver, stop and choose a compatible PyTorch build, then record that environment explicitly. Do not change a shared server driver. This trainer uses float32 and gradient checkpointing; available memory has not been profiled. A small batch is a starting point, not a memory guarantee. Respect scheduler-provided CUDA_VISIBLE_DEVICES.

## 3. Smoke-check the real backbone

This downloads ModernBERT-large and performs a few updates on tiny illustrative data. It is a hardware/pipeline check, not a baseline result. Pin the currently selected Hub revision:

```sh
MODEL_REVISION=$(uv run --no-sync python -c 'from huggingface_hub import model_info; print(model_info("answerdotai/ModernBERT-large").sha)')
uv run --no-sync decisions check --data packages/modernbert-decisions/examples/illustrative.jsonl
uv run --no-sync decisions train \
  --config packages/modernbert-decisions/configs/warmup.yaml \
  --data packages/modernbert-decisions/examples/illustrative.jsonl \
  --revision "$MODEL_REVISION" --device cuda \
  --batch-size 1 --accumulation 1 --max-length 512 \
  --output runs/gpu-smoke
```

If memory is insufficient at batch one, do not launch the pilot: memory/precision changes need to be made and verified first. Overlong inputs fail explicitly; don't silently truncate the eventual research data.

## 4. Prepare the actual pilot

The real mixed dataset is not assembled yet. Supply `data/pilot.jsonl` using the [trainer schema](../packages/modernbert-decisions/README.md). Use audited categorical, Boolean and ordinal labels; preserve official splits and document groups. Reserve a calibration split without using test labels. Existing dataset converters are described in the [experiment register](experiment-register.md), but automatic downloads/conversion of the full supervised mixture are not implemented here.

Required split names for the commands below: train, calibration, test. If your reserved split is called validation, use that name consistently. Run `decisions check` and inspect counts/grouping before training. Synthetic mock examples cannot substitute for this pilot.

```sh
uv run --no-sync decisions check --data data/pilot.jsonl
uv run --no-sync decisions train \
  --config packages/modernbert-decisions/configs/warmup.yaml \
  --data data/pilot.jsonl --data-revision pilot-v1 \
  --revision "$MODEL_REVISION" --device cuda \
  --batch-size 2 --accumulation 4 --output runs/warmup
```

Choose the length limit and batch size after the audit and GPU smoke. Use a stable terminal/session or scheduler job for longer runs. The trainer saves at completion; it does not implement mid-epoch recovery. Each output directory must be fresh.

## 5. Matched CE and AURC continuations

Both arms start from the same warm-up, with the same data and batch settings. Optimizers restart identically. AURC uses the actual microbatch for ranking; accumulation does not enlarge that ranking pool. Batch two may be weak for studying the loss: increase the matched microbatch if measured memory permits, and record it.

```sh
for arm in ce aurc; do
  uv run --no-sync decisions train \
    --config "packages/modernbert-decisions/configs/continue-${arm}.yaml" \
    --data data/pilot.jsonl --data-revision pilot-v1 --device cuda \
    --batch-size 2 --accumulation 4
  for split in calibration test; do
    uv run --no-sync decisions evaluate \
      --checkpoint "runs/continue-${arm}/checkpoint" \
      --data data/pilot.jsonl --split "$split" --device cuda --batch-size 2 \
      --output "runs/${arm}-${split}"
  done
done
uv run --no-sync python scripts/compare_runs.py \
  --ce runs/ce-test/predictions.jsonl --aurc runs/aurc-test/predictions.jsonl \
  --ce-calibration runs/ce-calibration/predictions.jsonl \
  --aurc-calibration runs/aurc-calibration/predictions.jsonl \
  --out runs/comparison
```

For CE only, run the loop with `for arm in ce` and retain its raw metrics/logits. The paired report requires both arms. Once both exist, it fits temperatures and selects thresholds using calibration rows only, then reports unchanged thresholds on test.

Review resolved YAML, prediction counts, accuracy/NLL, risk-coverage, reliability and threshold outcomes. Update the [run ledger](run-ledger.csv) with hardware, source snapshot, data manifest, time, conclusion and next action. GPU commands above are instructions; they have not been executed on the CVC server.
