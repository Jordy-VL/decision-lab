# Selective prediction with indexed decision models

*Working manuscript baseline. Updated 2026-09-24.*

## Scope

The experiment tests whether an AURC-motivated loss improves confidence-based
selective prediction over ordinary cross-entropy. The model maps a state,
question and ordered options to an option index and probability distribution.
This is an independent experiment, not a reproduction of Jev or the Decision
Models systems.

The [Decision Models paper](https://vllm-sr.ai/decision-paper.pdf) is related
work. Its ModernBERT-style systems use candidate-aware scoring,
candidate-distribution supervision and, in some systems, staged adaptation.
Our controlled runs use the public `answerdotai/ModernBERT-large`, a
CLS-pooled fixed-slot head, full encoder/head fine-tuning and a loss ablation.
The systems are not directly comparable training recipes.

## Model and data

The encoder is `answerdotai/ModernBERT-large`, revision
`45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`. State, question and numbered
options are serialized into one sequence. A two-layer MLP maps the first-token
representation to 128 option logits; unused slots are masked. Inputs longer
than 2,048 tokens and examples with more than 128 options are rejected.

Data is the pinned Kev decision-v7 reproduction suite, adapted to one row per
question:

| Partition | Examples | Role |
|---|---:|---|
| Train | 15,576 | Parameter updates |
| Development | 1,468 | Checkpoint selection by NLL |
| Calibration | 1,148 | Post-training calibration and threshold fitting |
| Test | 1,440 | Frozen final evaluation |

Test data is prepared only after the model and calibration protocol are
frozen. These are Kev reproduction records, not TypeSafe's original Jev
training mixture.

The Hub now provides a newer `v8/decision-v8` suite and a separate
evaluation-only `v9/transfer-v9` suite. We retain v7 for this baseline:
v8 changes train/calibration exposure, so using it would require a fresh
matched rerun rather than a silent data update.

## Matched one-epoch runs

Both completed runs used the same pinned model and data, seed 17, batch size 4,
one epoch, 3,894 AdamW updates, learning rate `2e-5`, weight decay `0.01`,
5% warmup, cosine decay, gradient checkpointing and float32 precision. They
started directly from the same pretrained checkpoint; neither used the
historical Modal checkpoint.

| Arm | Loss | Output | Status |
|---|---|---|---|
| CE | `aurc_lambda=0.0` | `runs/x2-ce` | completed |
| AURC-only | `aurc_lambda=1.0` | `runs/x2-aurc-only` | completed |
| CE+AURC mix | `aurc_lambda=0.5` | `runs/x2-ce-aurc-mix` | not yet run |

The AURC surrogate uses detached harmonic confidence ranks within each actual
microbatch. It is not direct optimization of discrete AURC.

## Development and calibration results

These are raw, pooled per-question metrics from each arm's best checkpoint.
The best checkpoint is selected by development NLL. No temperature or
threshold was fitted, and no test result is included.

| Arm | Split | Accuracy | NLL | Brier | AURC | Ordinal MAE |
|---|---|---:|---:|---:|---:|---:|
| CE | Development | 43.39% | 1.3858 | 0.6225 | 0.3292 | 0.9918 |
| CE | Calibration | 48.26% | 1.1789 | 0.5894 | 0.3334 | 0.9421 |
| AURC-only | Development | 60.49% | 1.2264 | 0.5067 | 0.1923 | 0.5960 |
| AURC-only | Calibration | 65.51% | 0.8044 | 0.4054 | 0.1505 | 0.5317 |

These results are an initial single-seed observation, not evidence of a causal
AURC benefit. The arms have not yet been compared on the frozen test set, and
the CE+AURC mix arm is pending.

## Evaluation protocol

After all arms and calibration choices are frozen:

1. Select checkpoints using development NLL; report development Brier as a
   secondary diagnostic, never accuracy as the selection metric.
2. Fit raw-to-calibrated mappings on calibration only.
3. Report raw, temperature-scaled, isotonic and pre-specified spline results.
4. Freeze calibrators and confidence thresholds.
5. Evaluate once on test and report accuracy, NLL, Brier, ECE, AURC,
   risk-coverage and coverage at fixed risk.

Temperature scaling should change probability sharpness but preserve confidence
ordering. Flexible calibrators may re-rank examples; any resulting AURC change
must be identified separately from numerical calibration.

Run metadata, hashes, checkpoints, logits and predictions are retained under
each `runs/` directory. Full test evaluation remains a separate frozen action.
