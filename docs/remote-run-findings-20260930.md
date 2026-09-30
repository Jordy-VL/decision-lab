# Remote linear-run review — 2026-09-30

All five reviewed 10-epoch linear runs completed 38,940 steps. The four new
CE/AURC runs have already performed the first matched V1/V2 training comparison.
Candidate V2 performs substantially worse on development data. The three V1
loss arms have close development AUGRC values; this single-seed result does not
establish an objective winner.

Evidence is the read-only local snapshot under
`runs/remote-review-20260930/runs/`. Run names below abbreviate the common
`x2-` prefix and `-10epoch-linear-1e5` suffix. Reviewed each run's `config.yaml`,
`metadata.json`, `training.json`, `trainer_state.json` and
`evaluations/{best,final_budget}/{development,calibration}/{report,metadata}.json`.
Archived and `development-checkpoint-evaluations-shard0` results belong to
different checkpoint histories and must not be mixed with these current runs.

## Matching and provenance

All five use seed 17, ModernBERT-large revision
`45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`, initialized directly from that
pretrained encoder (not a CE parent checkpoint), 15,576 training rows, batch 4,
accumulation 1, FP32, 10 epochs, LR `1e-5`, linear decay, 10% warmup, weight
decay .01, max length 2,048, hidden head width 128 and capacity 128. Evaluation
and saving occur every 250 steps. Development has 1,468 rows and calibration
1,148. Common recorded SHA-256 hashes:

- Train: `39fc904150b9fc5265ae17ac38723027d49067cb3a660183af6be97896b42b70`
- Development: `2c886deeae75d8e475b20cf0511ed7e59d81da9ca0b55f6e31a9cdce45a4630d`
- Calibration: `2a61065d05cc0b5a52a9d536a4a60c9ec800791a7e407284959a73156608ad87`

CE and AURC runs for both heads record code revision
`0e5198dc3ca4e37139c928e091e3724c756b135a` and September 29 starts. V1 AUGRC
records `44578da5597f36d6b21bb30f051fc171b9983e87` and a September 25 start.
The source diff changes checkpoint retention from a fixed two to configurable
four in the newer trainer, plus config/job support; model and loss files do not
change between these revisions. Record this provenance difference rather than
claiming an identical code revision. Recorded torch/Transformers/Accelerate
versions match. Metadata records CUDA but does not establish identical hardware
or a clean working tree at launch.

## Checkpoint selector verified from actual history

For each run, `best_metric` equals the minimum `eval_augrc` in its own
`log_history`, and `best_model_checkpoint` points to that minimum's step.
Best-development reports reproduce the same AUGRC. All runs ended at epoch 10.

| Run | AUGRC-selected step | Best AUGRC | Minimum-AURC step | Minimum-NLL step |
|---|---:|---:|---:|---:|
| ce | 15,500 | .092748 | 15,500 | 2,750 |
| aurc-only | 20,500 | .092379 | 20,500 | 3,750 |
| augrc-only | 29,750 | .091493 | 29,750 | 4,000 |
| candidate-masks-ce | 28,250 | .214126 | 14,000 | 11,250 |
| candidate-masks-aurc-only | 25,250 | .236376 | 13,000 | 15,500 |

The differing candidate minimum-AURC steps independently demonstrate why AURC
and AUGRC selection cannot be described interchangeably. No retrospective
reselection was performed in this review.

## Development results at the selected checkpoint

| Run | Accuracy | NLL | Brier | AURC | AUGRC |
|---|---:|---:|---:|---:|---:|
| ce | .719346 | 1.684259 | .489735 | .133189 | .092748 |
| aurc-only | .716621 | 1.893751 | .500470 | .133048 | .092379 |
| augrc-only | .715940 | 2.160205 | .519371 | .133814 | .091493 |
| candidate-masks-ce | .463896 | 2.386847 | .783758 | .370597 | .214126 |
| candidate-masks-aurc-only | .402589 | 1.595249 | .711331 | .426137 | .236376 |

V2 CE loses 25.5 percentage points of accuracy relative to V1 CE. This weakness
appears across types: Boolean .7076 versus .8983, categorical .3558 versus
.6561, ordinal .3250 versus .5667. The local implementation audit did not find
a normal-path label/position alignment bug; these metrics do not identify the
cause. AURC training does not rescue this candidate configuration.

Final-budget development AUGRC is worse than selected AUGRC for all five:
CE .097378, AURC .095256, AUGRC .093466, candidate CE .223848 and candidate
AURC .250800. Their corresponding final NLL values are 2.505909, 2.516333,
2.361266, 3.347405 and 1.901452. Keep final and selected results separate.

## Calibration split inventory

These are **raw predictions on the calibration split**, not calibrated models.
They establish that calibration inputs exist; do not use this table to change
checkpoint selection.

| Selected run | Accuracy | Raw NLL | Raw Brier | Raw AUGRC |
|---|---:|---:|---:|---:|
| ce | .767422 | .841346 | .345918 | .063525 |
| aurc-only | .760453 | .921573 | .342761 | .061725 |
| augrc-only | .777003 | 1.130482 | .381635 | .060596 |
| candidate-masks-ce | .503484 | 1.909775 | .720301 | .204640 |
| candidate-masks-aurc-only | .412892 | 1.444766 | .702951 | .241355 |

## Remaining actions

1. Preserve these five selected checkpoints and provenance before further work.
   Freeze them for raw and calibrated test evaluation; fit temperatures and
   other declared mappings only on calibration data, with operating thresholds
   also fixed there. No test metrics were inspected or used in this review.
2. Repeat the predeclared matched five-epoch CE/AURC/mix comparison with seed 29;
   the ten-epoch linear runs do not substitute for that replication.
3. Investigate candidate learning on development/training diagnostics before
   spending another full training budget. Check tiny-set overfitting and actual
   pretrained marker gradients, then isolate marker initialization and sequence
   layout if needed. Log each change as a new experiment; do not tune on test.
4. Report the V2 result as a negative result for this simple shared-MLP head and
   prompt format. It is not evidence against Laya's materially different head
   or against all option-aware architectures; see the candidate-head audit.
