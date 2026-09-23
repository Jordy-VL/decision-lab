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

## Scope note — 2026-09-23 Trainer migration

The checks above predate the Hugging Face `Trainer` migration and do **not** verify the new `DecisionTrainer`, exact resume, checkpoint selection, or `.npz` development/calibration logit exports. Run the synthetic-fixture Trainer smoke and recovery check in [training-infrastructure-plan](../../docs/training-infrastructure-plan.md) before any new Kev research training. No new research run has been launched for this migration.
