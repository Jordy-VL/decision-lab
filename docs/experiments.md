# Minimal experimental protocol

No research results have been obtained. Smoke fixtures prove that code runs, not that AURC helps.

## Related reference

The [Decision Models paper](https://vllm-sr.ai/decision-paper.pdf) is saved as
background reference for typed decision interfaces, candidate scoring,
calibration and abstention/selective-prediction comparisons. It is related
work, not a specification of this experiment or evidence for our results.

Its ModernBERT-style encoder systems use candidate-aware shared scoring,
candidate-distribution cross-entropy and, in some systems, staged adaptation
with head warmup or frozen embeddings. Their released systems also fit a
positive temperature after training. Our controlled X2 study instead uses the
public pinned `answerdotai/ModernBERT-large`, one CLS-pooled fixed-slot head,
full encoder/head fine-tuning from the first step, and a CE versus
higher-confidence-weighted CE loss ablation. The paper's recipe combines
architecture, data, and training differences, so it is related work rather
than a directly matched baseline.

## Data and setup

Use official training/test partitions and reserve development/calibration data from permitted training data where necessary. If no official partitions exist, use one fixed document-grouped split. Split before generating derived questions; deduplicate across sources. Record source revisions, licenses, preprocessing, sample counts and hashes. Do not train on Decision Index or the public RVL-CDIP-N test set.

Each JSONL example has state, question, option descriptions, zero-based target_index, type, id, group_id, source and split. Ordinal rows also contain ordered numeric values. Evidence/provenance are stored but not fed as answer hints. Audit label semantics, length, class balance and unsupported evidence. The trainer rejects overlong examples rather than silently truncating.

## Matched training

Use ModernBERT-large, full fine-tuning and a fixed-capacity index head. Start each arm from the same pinned pretrained checkpoint with fresh matched optimizers, seed, data order, microbatch size and update budget. The initial objective matrix is lambda 0 (CE-only), lambda 1 (AURC-weighted CE surrogate only), and lambda 0.5 (blend). Lambda 1 is not direct optimization of discrete AURC. Do not pick lambda on test results.

The detached confidence ranks use the harmonic weighting described in [AsymptoticAURC](https://github.com/han678/AsymptoticAURC). Gradient accumulation does not enlarge the ranking microbatch. Track actual microbatch sizes and task composition. Report results by type/source because confidence is not directly comparable across option counts. We independently implement the formula; this is not a guarantee of out-of-domain calibration.

## Validation artifacts, measurements and plots

For every arm and selected checkpoint, save raw logits, labels, IDs, group IDs, source/type and complete probabilities for the development (validation) and calibration partitions. Keep these as immutable, paired artifacts alongside the checkpoint and resolved config. Select the best checkpoint by development NLL; use development Brier as a secondary probability-quality diagnostic, not as an accuracy substitute or a post hoc selection rule. Accuracy must not determine checkpoint selection because it ignores probability quality and confidence ordering. The current Trainer configuration implements development-NLL selection and retains the final-budget checkpoint separately. Never use final-test labels for checkpoint selection.

After all arms complete under the equal compute budget, evaluate both raw and calibrated outputs on the untouched test partition. For each arm, fit calibration only on the calibration partition, freeze the calibrator and any confidence thresholds, and apply them unchanged to test. The initial calibration comparison is raw probabilities, temperature scaling, isotonic regression and a pre-specified spline calibrator. Report every method rather than selecting one from test results; if a calibrator must be selected, use a declared calibration-only rule or a fit/selection split within calibration.

Temperature scaling is the low-variance primary calibration baseline: it can improve NLL, Brier and ECE but preserves confidence ordering, so it should not materially change AURC. Isotonic and spline calibration are flexible secondary methods; report their overfitting controls and whether they change example ordering. Any AURC change after a flexible calibrator must be separated from probability calibration because it may reflect re-ranking.

Report accuracy, NLL, Brier, ECE with bin counts and binning rules, maximum/adaptive calibration error where available, ordinal MAE by scale/source, AURC, risk-coverage curves, risk at fixed coverage, coverage at fixed risk and the thresholds used. Include raw-versus-calibrated reliability plots and calibration-curve data. Save logits, raw probabilities, calibrated probabilities and calibrator parameters to permit independent analysis. The test partition is evaluated only after checkpoint selection, calibration fitting, method/threshold freezing and all implementation choices are fixed.

Primary hypothesis: **more answered coverage at a given low error risk**, without an unacceptable loss of ordinary accuracy. Lower ECE alone does not confirm it. Lower AURC can partly reflect improved accuracy, so show full-coverage accuracy and the curves together. Threshold spacing is not an invariant utility measure. Exact confidence ties must be treated consistently; the report averages AURC over within-tie orders and plots whole-tie acceptance points.

The root script `scripts/compare_runs.py` produces PNG/SVG plots and JSON/CSV summaries. It rejects mismatched paired examples and overlapping calibration/test groups. Empirical risk targets of 1%, 5% and 10% are descriptive; the default minimum of 30 accepted calibration examples does not establish a population guarantee. If no eligible threshold exists, record that rather than inventing one.

### Frozen test evaluation

After the models and calibration protocol are frozen, prepare the locked test
partition exactly once:

```bash
uv run --env-file .env --no-sync python scripts/prepare_kev.py --allow-test
```

Evaluate each selected best checkpoint into a new directory. This produces
raw, uncalibrated test scores; it must not be used to choose a checkpoint,
temperature, spline, isotonic mapping, or threshold:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --env-file .env --no-sync decisions evaluate \
  --config packages/modernbert-decisions/configs/x2-ce.yaml \
  --data data/kev-decision-v7/test.jsonl \
  --split test \
  --checkpoint runs/x2-ce/checkpoint \
  --output runs/x2-ce-test-raw

CUDA_VISIBLE_DEVICES=1 uv run --env-file .env --no-sync decisions evaluate \
  --config packages/modernbert-decisions/configs/x2-aurc-only.yaml \
  --data data/kev-decision-v7/test.jsonl \
  --split test \
  --checkpoint runs/x2-aurc-only/checkpoint \
  --output runs/x2-aurc-only-test-raw
```

The command prints overall metrics and writes `predictions.jsonl` and
`report.json` under each output directory. Fit calibration on the existing
calibration partition first, apply the frozen mapping to saved test logits,
and keep raw and calibrated reports separate. The current CLI evaluation
command reports raw scores; temperature, isotonic and spline evaluation
belongs in the planned post-processing analysis and must not be fitted on
test.

Initial runs use one seed. Before publication or a strong effectiveness claim, add document-grouped uncertainty analysis and targeted repeat runs if the pilot merits it. The report currently supplies point estimates, not confidence bands. Freeze costs before any utility experiment; evaluate expected utility under explicit wrong-answer and deferral costs.

## Decisions after results

Pending user-supplied AURC evaluation code: preserve the distinction between (1) validation/calibration-selected thresholds applied unchanged to test and (2) an exploratory, retrospective test-distribution coverage/risk frontier selected using test labels. The second is an oracle diagnostic, not a deployable held-out guarantee or a basis for tuning the final model. Do not extend the existing threshold implementation until that code is reviewed.

If selective coverage improves robustly, test fixed out-of-domain sources with unchanged thresholds. If only ECE improves, describe a calibration result. If AURC improves while accuracy falls, inspect the tradeoff before claiming benefit. If there is no useful gain, check data semantics, confidence ranking and microbatch composition before scaling. RL abstention, vision and multi-turn rollouts remain separate follow-up hypotheses.

See the [experiment register](experiment-register.md) for all experiments and original sources. Every real run should record commit, full resolved YAML, data manifest, model revision, checkpoint lineage, hardware, duration, metrics, artifacts and conclusion. No trial should disappear because its outcome is negative.
