# Smoke verification — 2026-09-22

Passed in an isolated Windows CPU environment: Python 3.12, PyTorch 2.14.0+cpu, Transformers 4.57.6, PyYAML 6.0.3.

- Editable package build/install, package compilation, and generated CLI help.
- All seven illustrative rows pass schema/split validation (3 train, 1 validation, 3 test).
- The sibling generator's six-row `examples/mock/trainer.jsonl` loads directly (two rows per primitive).
- Missing-split assignment is deterministic and keeps document groups together; cross-split groups and out-of-range targets are rejected.
- YAML config plus CLI override and boolean flag parsing.
- Kev labeled requests adapt correctly for categorical, boolean and explicit-valued ordinal tasks, retaining a shared document group.
- Tiny randomly initialized ModernBERT (2 layers, hidden size 32) runs forward/backward with a bounded index head.
- Masked option slots have zero probability; unpadded probability vectors sum to one.
- Encoder, tokenizer and head save/load preserve evaluation logits and token IDs.
- Harmonic blended loss agrees with a hand-computed three-row example; singleton microbatches and singleton type groups reduce to CE; gradients are finite.
- Two tied-confidence examples with one error produce tie-averaged AURC 0.5.
- Two weights-only continuation arms from the same tiny checkpoint run through train, save, evaluate and predict, using batch size 2 and accumulation 2 (including a partial final microbatch/window).
- Excessive token lengths and excessive option counts raise errors.
- Official `answerdotai/ModernBERT-large` tokenizer (tokenizer files only) correctly produces CLS/SEP and exact total counts for all seven examples: 35–49 tokens.

No pretrained encoder weights, real training dataset, GPU run, unit-test suite, or performance benchmark was used. Tiny random-model metrics are not research results. The one-off smoke script/environment and temporary checkpoints live under the task's `work/` directory, outside this deliverable. Parent repository integration and its dependency lock are separate work.

## Hugging Face Trainer migration smoke — 2026-09-23

Passed locally on CPU with Python 3.14.4, PyTorch 2.14.0+cpu, Transformers 4.57.6, Accelerate 1.15.0, and NumPy 2.5.3. Run from the repository root after installing the training dependencies:

```bash
PYTHONPATH=packages/modernbert-decisions python scripts/smoke_hf_trainer.py
```

The smoke creates a tiny random BERT and synthetic, group-disjoint train/development/calibration files. It performs four optimizer steps, evaluates and saves each step, and verifies:

- best-checkpoint selection and reload from an earlier development-NLL minimum (step 1) while retaining the final-budget model (step 4);
- development/calibration `.npz` IDs, labels, option counts, active logits and padding against JSONL, plus logits reproduced by the saved best model;
- exact mid-run recovery from Trainer checkpoint 2 to step 4, with final-budget weights bitwise identical to an uninterrupted run.

All generated smoke artifacts are temporary by default; `--keep-output runs/hf-trainer-smoke-local` retains them in an ignored run directory. This validates the local CPU Trainer/artifact/recovery path only. It uses no pretrained weights, Kev/research data, GPU, or CVC allocation and is not evidence of model quality. No new research arm was launched.

The single-update local smoke can be run with:

```bash
PYTHONPATH=packages/modernbert-decisions python scripts/smoke_hf_trainer.py --single-batch --device cpu
```

After that passes, the same single-update fixture can verify GPU execution on an allocated CUDA host with `--device auto --require-cuda`.

To upload a completed smoke run explicitly to the private model repository
`jordyvl/decision-lab-smoke`, set the project-scoped `HF_HOME` and add
`--upload-hf`:

```bash
HF_HOME=/home-local/sbiswas/.cache/huggingface-decision-lab \
PYTHONPATH=packages/modernbert-decisions \
python scripts/smoke_hf_trainer.py \
  --single-batch --device auto --require-cuda --upload-hf
```

Uploading is disabled by default. The command creates a timestamped folder in
the private repository and prints the Hub commit URL only after all local smoke
checks pass.
