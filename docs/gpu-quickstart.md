# GPU quickstart — X2 redo

Updated 2026-09-23. This guide is for the Hugging Face Trainer redo workflow. Do not use the old illustrative-data training commands or launch research arms until the prerequisites in [training-infrastructure-plan](training-infrastructure-plan.md) are complete.

## Clone on `uab-gpu`

From the Windows machine, use the existing SSH alias:

```powershell
ssh uab-gpu
```

On the server, after obtaining a GPU allocation through its scheduler if required:

```sh
git clone git@github.com:Jordy-VL/decision-lab.git
cd decision-lab
nvidia-smi
```

If GitHub SSH authentication is not configured on the server, use `git clone https://github.com/Jordy-VL/decision-lab.git` if the repository is public; otherwise configure the server's own approved GitHub authentication. Do not copy personal SSH keys or tokens into the repository.

## Environment and pinned data

Install `uv` in the user's environment if it is not available, then install the lockfile-pinned project:

```sh
uv sync --locked --python 3.12 --extra train --extra report
```

Verify the allocated device before running any GPU work:

```sh
uv run --no-sync python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: check allocation and driver/wheel compatibility'
print('GPU:', torch.cuda.get_device_name(0))
print('Free/total bytes:', torch.cuda.mem_get_info())
PY
```

Prepare only the frozen train, development and calibration partitions; test remains untouched:

```sh
uv run --no-sync python scripts/prepare_kev.py
uv run --no-sync decisions check --config packages/modernbert-decisions/configs/ce-baseline.yaml
```

## Before X2

1. Confirm the verified frozen CE checkpoint has been copied to `runs/ce-baseline/checkpoint`.
2. Run the small Trainer/checkpoint-resume smoke described in [training-infrastructure-plan](training-infrastructure-plan.md); confirm best/final checkpoints and development/calibration logits artifacts.
3. Review each arm config in `packages/modernbert-decisions/configs/x2-experiment-group.json` and ensure the checkpoint/data paths resolve.
4. Launch the three arms only after the run bundle, GPU constraints and recovery path are confirmed. Each output directory must be new and unique.

The three configs are `x2-ce.yaml`, `x2-aurc-only.yaml`, and `x2-ce-aurc-mix.yaml`. They share the same data, pinned pretrained ModernBERT revision, seed, optimizer, schedule and update budget; they differ only in `aurc_lambda` and output directory. Each arm starts from the pretrained checkpoint with a fresh optimizer and writes best and final-budget development/calibration logits to compressed NumPy files, along with paired JSONL predictions and reports.
