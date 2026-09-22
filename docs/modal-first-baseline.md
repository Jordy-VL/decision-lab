# Run the first CE baseline on Modal

2026-09-22. Planning/setup guide, not a completed cloud run. The trainer, sectioned YAML and exact Kev data adapter are implemented. **The Modal launch wrapper is the remaining implementation step.** No account access, credit balance, cloud image build or GPU run has been checked here.

## What to review first

1. [Kev baseline data](kev-baseline.md): exact reproduction-source attribution, pinned files and preserved partitions.
2. [CE config](../packages/modernbert-decisions/configs/ce-baseline.yaml): ModernBERT-large, one index head, full fine-tuning, CE, one warm-up epoch. This is our baseline on Kev data, not Kev's LoRA recipe.
3. Your Modal workspace's credits and billing. You mentioned existing credits; their amount and expiry remain unverified. Nebius credits are separate.

The initial task is one GPU smoke/profile run, then one CE warm-up. No synthetic additions, sweep or public deployment. Matched CE/AURC continuations follow the frozen warm-up. Our evaluation extension waits for your supplied code.

## Local setup

From the repository root in WSL/Linux, install the CLI separately from the trainer environment:

```sh
uv sync --locked
uv run modal setup
uv run modal token info
python3 scripts/prepare_kev.py
uv run --extra train decisions check --config packages/modernbert-decisions/configs/ce-baseline.yaml
```

Modal is a project development dependency, version-pinned in `uv.lock`; run these commands from the repository root. `uv run modal setup` opens browser authentication. Verify the intended workspace and inspect available credits in its billing page. GPU usage requires a payment method even when credits are available. Keep credentials in Modal's local configuration; never place them in the repository. [Official setup](https://modal.com/docs/guide/getting-started), [GPU requirements](https://modal.com/docs/guide/gpu).

The preparation command downloads only train/calibration/development. Test remains untouched. If local PyTorch installation is undesirable, the schema check can be performed inside the eventual Modal image instead; data preparation itself uses standard Python only.

## Minimal wrapper to implement

Use a one-shot Modal Function, not a persistent web service. Select one A100 40GB, two CPU cores and 16 GiB host RAM for the initial attempt. Available GPU memory and throughput must still be measured.

- Build a Python 3.12 image with a verified GPU-compatible PyTorch build and the pinned trainer dependencies. The workspace lock currently resolves CUDA 13 PyTorch; verify the Modal driver/image combination before relying on that build. Record any deliberate dependency change.
- Upload only the trainer package, baseline configuration, preparation script and pinned source config. Do not upload the entire workspace/vault, local environments, SSH material or unrelated data. Modal can include selected local files directly; a public Git repository is unnecessary. [Image and local-file documentation](https://modal.com/docs/guide/images).
- Prepare the hash-verified suite on a CPU function or before GPU execution. Persist data, Hugging Face cache and output checkpoints using a named Volume. Place each run in a unique output directory and commit the Volume after writes. [Volumes](https://modal.com/docs/guide/volumes).
- Resolve and record an exact ModernBERT Hub revision. Invoke the existing CLI as a subprocess; preserve its stdout, resolved YAML, metadata, split manifest and checkpoints.
- Use a short timeout and no automatic retries for the first profile. A timeout is a duration bound, not an exact dollar budget. The trainer currently saves its model only at completion, so preemption/timeouts can lose the unfinished run; add periodic recovery separately before long jobs if necessary.
- Download or inspect the completed artifacts before starting another arm. No background service needs to remain running after a one-shot job.

The wrapper should pass the existing command, with paths resolved inside the container:

```sh
python -m decisions train \
  --config packages/modernbert-decisions/configs/ce-baseline.yaml \
  --revision MODEL_COMMIT --device cuda \
  --output /artifacts/RUN_ID/ce-baseline
```

`MODEL_COMMIT` and `RUN_ID` are placeholders that must be supplied. There is intentionally no `modal run` command for a nonexistent wrapper in this guide. Once implemented and smoke-checked, record its exact invocation here.

## First profile and budget

Use a small explicitly labeled training subset solely for the infrastructure profile; do not report its results as a baseline. Measure GPU allocation/download overhead, examples per second and peak memory. Then estimate the full 15,576-question epoch from measured throughput, accounting for the full length distribution. The verified training inputs reach 1,140 tokenizer tokens; shorter toy fixtures would underestimate cost.

The current trainer is float32 with gradient checkpointing. Start conservatively and measure batch size; do not assume it has BF16/autocast, a scheduler or mid-epoch resume. If we add performance features, commit them separately and keep both loss arms matched.

At the checked [Modal standard rates](https://modal.com/pricing), A100 40GB costs $2.0988/hour for GPU alone. Two CPU cores plus 16 GiB RAM add approximately $0.2222/hour, giving **about $2.32/hour** before other charges/credits:

| Compute duration | Approximate cost |
|---|---:|
| 10 minutes | $0.39 |
| 30 minutes | $1.16 |
| 1 hour | $2.32 |
| 2 hours | $4.64 |

These are duration scenarios, not measured training forecasts. Downloads/preparation, storage, repeated failed runs and other resources may add cost. Region-specific or non-preemptible settings can carry premiums. Starter currently advertises $30/month included compute, but inspect your actual account. A $5–10 pilot allowance is a proposed cap for planning, not proof that all experiments fit or authorization to spend it.

## After the CE warm-up

Evaluate development and calibration separately, using their preserved names and files. Save raw logits/probabilities. Freeze the checkpoint, then run matched CE and CE+AURC continuations from it; do not train one arm from the other. Test is prepared only after model/threshold decisions are frozen. Our custom mixtures and external full Decision Index run come later.

Ready now: data, config, trainer, local smoke evidence. Remaining before paid execution: Modal wrapper/image validation, account/credit check, bounded GPU profile, exact run budget and pinned model revision.
