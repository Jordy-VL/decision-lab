# Minimal experimental protocol

No research results have been obtained. Smoke fixtures prove that code runs, not that AURC helps.

## Data and setup

Use official training/test partitions and reserve development/calibration data from permitted training data where necessary. If no official partitions exist, use one fixed document-grouped split. Split before generating derived questions; deduplicate across sources. Record source revisions, licenses, preprocessing, sample counts and hashes. Do not train on Decision Index or the public RVL-CDIP-N test set.

Each JSONL example has state, question, option descriptions, zero-based target_index, type, id, group_id, source and split. Ordinal rows also contain ordered numeric values. Evidence/provenance are stored but not fed as answer hints. Audit label semantics, length, class balance and unsupported evidence. The trainer rejects overlong examples rather than silently truncating.

## Matched training

Use ModernBERT-large, full fine-tuning and a fixed-capacity index head. Fit one CE warm-up; start both continuation arms from that same checkpoint with fresh matched optimizers, seed, data order, microbatch size and update budget. Lambda is 0 for CE and 0.5 for the initial AURC blend. Do not pick lambda on test results.

The detached confidence ranks use the harmonic weighting described in [AsymptoticAURC](https://github.com/han678/AsymptoticAURC). Gradient accumulation does not enlarge the ranking microbatch. Track actual microbatch sizes and task composition. Report results by type/source because confidence is not directly comparable across option counts. We independently implement the formula; this is not a guarantee of out-of-domain calibration.

## Measurements and plots

Compare CE, CE+AURC, CE with temperature scaling, and CE+AURC with temperature scaling. Fit temperature on calibration only; choose confidence thresholds there and apply unchanged on test. Report accuracy, NLL, Brier, ECE with bin counts, ordinal MAE by scale/source, AURC, risk-coverage curves, and coverage/risk at fixed thresholds. Save logits and complete probabilities to permit independent analysis.

Primary hypothesis: **more answered coverage at a given low error risk**, without an unacceptable loss of ordinary accuracy. Lower ECE alone does not confirm it. Lower AURC can partly reflect improved accuracy, so show full-coverage accuracy and the curves together. Threshold spacing is not an invariant utility measure. Exact confidence ties must be treated consistently; the report averages AURC over within-tie orders and plots whole-tie acceptance points.

The root script `scripts/compare_runs.py` produces PNG/SVG plots and JSON/CSV summaries. It rejects mismatched paired examples and overlapping calibration/test groups. Empirical risk targets of 1%, 5% and 10% are descriptive; the default minimum of 30 accepted calibration examples does not establish a population guarantee. If no eligible threshold exists, record that rather than inventing one.

Initial runs use one seed. Before publication or a strong effectiveness claim, add document-grouped uncertainty analysis and targeted repeat runs if the pilot merits it. The report currently supplies point estimates, not confidence bands. Freeze costs before any utility experiment; evaluate expected utility under explicit wrong-answer and deferral costs.

## Decisions after results

If selective coverage improves robustly, test fixed out-of-domain sources with unchanged thresholds. If only ECE improves, describe a calibration result. If AURC improves while accuracy falls, inspect the tradeoff before claiming benefit. If there is no useful gain, check data semantics, confidence ranking and microbatch composition before scaling. RL abstention, vision and multi-turn rollouts remain separate follow-up hypotheses.

See the [experiment register](experiment-register.md) for all experiments and original sources. Every real run should record commit, full resolved YAML, data manifest, model revision, checkpoint lineage, hardware, duration, metrics, artifacts and conclusion. No trial should disappear because its outcome is negative.
