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

## Matched duration sweep

All runs used the same pinned model and data, seed 17, batch size 4,
learning rate `2e-5`, weight decay `0.01`, 5% warmup, cosine decay, gradient
checkpointing and float32 precision. Each started directly from the same
pretrained checkpoint; none used the historical Modal checkpoint.

| Arm | Loss | Epochs | Updates |
|---|---:|---:|---:|
| CE | `aurc_lambda=0.0` | 2, 5, 10 | 7,788 / 19,470 / 38,940 |
| AURC-only | `aurc_lambda=1.0` | 2, 5, 10 | 7,788 / 19,470 / 38,940 |
| CE+AURC mix | `aurc_lambda=0.5` | 2, 5, 10 | 7,788 / 19,470 / 38,940 |

The AURC surrogate uses detached harmonic confidence ranks within each actual
microbatch. It is not direct optimization of discrete AURC.

## Duration selection and raw test results

Checkpoint selection remained development-NLL based. The best development-NLL
duration was 5 epochs for CE (1.2033), 5 epochs for AURC-only (1.2756), and 2
epochs for the mix (1.2447). These per-arm minima are diagnostic; they are not
a matched-budget comparison because they use different durations.

The following are raw, pooled per-question test metrics from the selected
development checkpoints at each duration. No temperature or threshold was
fitted.

| Epochs | Arm | Dev NLL | Dev Brier | Test accuracy | Test NLL | Test Brier | Test AURC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | CE | 1.4216 | 0.6480 | 39.65% | 1.4674 | 0.6680 | 0.3926 |
| 2 | AURC-only | 1.4173 | 0.6381 | 40.21% | 1.4755 | 0.6612 | 0.3637 |
| 2 | CE+AURC mix | 1.2447 | 0.5460 | 54.17% | 1.3350 | 0.5871 | 0.2533 |
| 5 | CE | 1.2033 | 0.5206 | 56.32% | 1.2415 | 0.5378 | 0.2554 |
| 5 | AURC-only | 1.2756 | 0.5001 | 66.60% | 1.2722 | 0.4979 | 0.1697 |
| 5 | CE+AURC mix | 1.2468 | 0.5528 | 52.85% | 1.2893 | 0.5702 | 0.2746 |
| 10 | CE | 1.2352 | 0.5530 | 53.68% | 1.2646 | 0.5582 | 0.2660 |
| 10 | AURC-only | 1.2966 | 0.5878 | 48.19% | 1.3633 | 0.6058 | 0.2936 |
| 10 | CE+AURC mix | 1.2457 | 0.5394 | 56.88% | 1.3171 | 0.5567 | 0.2404 |

The 5-epoch matched comparison is the strongest current pilot result:
AURC-only has lower test NLL, Brier and AURC than CE, while the mix does not
improve on either at that budget. These are single-seed results and should not
be treated as causal evidence without repeats and calibration analysis.

## Evaluation protocol

After all arms and calibration choices are frozen:

1. Select checkpoints using development NLL; report development Brier and
   development AURC as secondary diagnostics, never accuracy as the selection
   metric.
2. Fit raw-to-calibrated mappings on calibration only.
3. Report raw, temperature-scaled, isotonic and pre-specified spline results.
4. Freeze calibrators and confidence thresholds.
5. Evaluate once on test and report accuracy, NLL, Brier, ECE, AURC,
   risk-coverage, and coverage at fixed selective risk levels, including
   `Cov@5% risk` and the corresponding confidence threshold.

Temperature scaling should change probability sharpness but preserve confidence
ordering. Flexible calibrators may re-rank examples; any resulting AURC change
must be identified separately from numerical calibration.

Run metadata, hashes, checkpoints, logits and predictions are retained under
each `runs/` directory. Full test evaluation remains a separate frozen action.
