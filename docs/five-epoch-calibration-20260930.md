# Historical five-epoch temperature comparison — 2026-09-30

**Temperature scaling improves the AURC-only arm, but calibration-derived risk thresholds do not transfer at their nominal budgets.** This is seed 17, V1, historical **development-NLL-selected** checkpoints from five-epoch training budgets, not final five-epoch weights or AUGRC-selected reruns. It is not seed-29 replication.

## Provenance and procedure

Compared `x2-ce-5epoch`, `x2-aurc-only-5epoch`, and `x2-ce-aurc-mix-5epoch`: selected steps **1750, 7250, 1750**, respectively. Calibration exports name the same checkpoint export paths used by the test evaluations. Test parent metadata exactly matches training metadata; selected checkpoints match trainer states and minimum logged development NLL. This establishes metadata consistency, **not weight-content identity**: weights were not downloaded or hashed.

All arms match on IDs, ordered options, target indices, groups, source and question type within each split. Calibration has **1,148** rows; test has **1,440**. Calibration/test IDs and groups are disjoint. Seed and training/calibration/test data hashes match across arms. Prediction-file SHA-256 hashes and code revisions are retained in the [aggregate report](../reports/five-epoch-calibration-20260930.json).

One scalar temperature per arm minimizes calibration NLL using the existing bounded `log(T) ∈ [-3,3]` search. [Frozen temperatures](../reports/five-epoch-calibration-20260930-temperatures.json) are written before test scoring. Every raw and temperature arm is reported; no test-based temperature or arm selection was performed. Calibration NLL decreases for every arm (CE 0.84899→0.84600; AURC 0.69762→0.69025; mix 0.92053→0.92028). Isotonic/spline fitting was not added without a fully specified protocol.

## Frozen test results

Lower is better for all metrics below. ECE uses ten equal-width confidence bins; Brier sums squared class errors per example. AURC handles confidence ties by expected within-tie ordering; AUGRC uses the repository's pairwise definition.

| Arm | T | NLL | Brier | ECE | AURC | AUGRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE raw | 1 | 1.24148 | 0.53782 | 0.07686 | 0.25537 | 0.15317 |
| CE temperature | 0.872712 | 1.28229 | 0.54335 | 0.09907 | 0.25943 | 0.15343 |
| AURC-only raw | 1 | 1.27215 | 0.49790 | 0.15517 | 0.16966 | 0.10993 |
| AURC-only temperature | 1.212316 | 1.16790 | 0.48576 | 0.12912 | 0.16764 | 0.10946 |
| CE+AURC raw | 1 | 1.28929 | 0.57025 | 0.08559 | 0.27463 | 0.16870 |
| CE+AURC temperature | 0.963824 | 1.29826 | 0.57215 | 0.09244 | 0.27547 | 0.16876 |

Accuracy is unchanged by positive temperature scaling: CE **56.32%**, AURC-only **66.60%**, mix **52.85%**. Across examples with different multiclass logit vectors, temperature can change confidence ordering, hence AURC/AUGRC can change.

**Correction to older summary prose:** these exports do not support lower *raw* AURC-only NLL than CE. AURC-only has better raw Brier and AURC, but raw NLL is 1.27215 versus CE's 1.24148. After calibration its NLL is lower. Retain that distinction rather than generalizing one objective's advantage.

## Calibration-fitted thresholds, applied unchanged to test

Choose maximum calibration coverage with empirical risk at or below budget and at least **30** accepted calibration examples. This is an empirical operating-point procedure, not a statistical risk guarantee. Entries show **threshold; accepted test count / 1,440; test error count; test risk**. `None` means no eligible calibration operating point, so no test examples are accepted. Full calibration counts and coverage are in JSON.

| Arm | 1% budget | 5% budget | 10% budget |
| --- | --- | --- | --- |
| CE raw | None | 0.840481; 478; 95; 19.87% | 0.620455; 680; 155; 22.79% |
| CE temperature | None | 0.879972; 466; 93; 19.96% | 0.637797; 698; 161; 23.07% |
| AURC-only raw | 0.998294; 108; 3; 2.78% | 0.945241; 742; 134; 18.06% | 0.821092; 909; 181; 19.91% |
| AURC-only temperature | 0.994810; 108; 3; 2.78% | 0.905065; 737; 132; 17.91% | 0.766734; 903; 178; 19.71% |
| CE+AURC raw | None | 0.902061; 275; 45; 16.36% | 0.704576; 571; 133; 23.29% |
| CE+AURC temperature | None | 0.909157; 277; 45; 16.25% | 0.718104; 569; 131; 23.02% |

At the calibrated 5% operating point, coverage is CE **32.36%**, AURC-only **51.18%**, mix **19.24%**, but **none meets 5% test risk**. Do not present these as coverage achieved at 5% test risk. Investigate calibration/test composition and per-source reliability before deployment claims; do not retune against this test set. Matched seed-29 replication and the current AUGRC-selection protocol remain separate experiments.

## Reproduce

From repository root, using the downloaded metadata/prediction exports (ignored raw artifacts):

```sh
python3 scripts/analyze_five_epoch_calibration.py \
  --runs runs/remote-review-20260930/runs \
  --out reports/five-epoch-calibration-20260930.json
```

The script uses the standard library and existing comparison/metric functions. It validates provenance and pairing before fitting, preserves raw exports, and emits aggregate JSON plus frozen temperatures. No training or model inference was run.
