# Frozen flexible confidence protocol — 2026-09-30

Written before executing these flexible mappings. Historical test results have already been inspected in the temperature analysis, so this is **exploratory**, not confirmatory evidence.

Scope: three historical seed-17, development-NLL-selected five-epoch arms (CE, AURC-only, mix). Input score is the raw maximum softmax probability; target is whether its top-label prediction is correct. Predicted class and full class probabilities are never changed. These are **scalar correctness-confidence** maps, not multiclass distribution calibration. Temperature remains a separate experiment.

Partition calibration groups deterministically using SHA-256 of `confidence-v1:` plus group ID: first eight hex digits modulo 10 below 6 goes to mapping fit; remaining groups go to held-out calibration assessment and operating-point selection. All arms share the partition. Require nonempty partitions and disjoint group/row IDs from test. Match paired IDs, ordered options, labels and groups between arms; reuse the historical provenance checks conceptually without refitting temperature.

Report every method; no winner or hyperparameter selection based on held-out calibration or test:

1. Raw confidence identity baseline.
2. Isotonic regression: least-squares PAVA, aggregate exact input ties first with count weights; piecewise-constant right-continuous prediction, flat extrapolation. No smoothing or tuned regularizer.
3. Monotone cubic spline: ten fixed equal-width input bins on [0,1]; omit empty bins, use bin's mean confidence as its x coordinate and Jeffreys-smoothed correctness `(correct+0.5)/(n+1)` as y, weight `n+1`; PAVA enforces monotonicity before shape-preserving cubic Hermite interpolation (PCHIP), with flat extrapolation. One occupied bin yields a constant. No knot/count/smoothing search.

Fit mappings using fit partition only. Serialize mappings before assessing held-out calibration or test. Report scalar correctness Brier, binary correctness log loss (clip only for log scoring at 1e-12), ten-bin correctness ECE, AURC and AUGRC; do not report fabricated multiclass NLL/Brier for these maps. Accuracy is unchanged. Report fit and held-out counts and metrics, but distinguish fit scores as in-sample diagnostics.

For each method and each 1%, 5%, 10% risk budget, select maximum held-out-calibration coverage with empirical selective risk <= budget and >=30 accepted rows. Include all exact-score ties. Freeze thresholds before applying to test. Report threshold, calibration accepted/errors, test accepted/errors/coverage/risk. No eligible calibration point means abstain on all test rows. Empirical thresholds do not guarantee population risk, and reuse of assessment data to choose thresholds makes its selected risk optimistic.

Use standard-library implementations with meaningful synthetic validation of PAVA pooling/ties, monotonic bounded interpolation and flat extrapolation, group partition invariance, and separation of fit versus assessment. This avoids new package dependencies. Keep raw exports untouched; publish aggregate results and serialized mappings only.
