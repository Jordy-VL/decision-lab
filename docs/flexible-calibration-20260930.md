# Flexible scalar confidence calibration — 2026-09-30

**Exploratory result:** flexible confidence mappings do not solve threshold transfer. AURC-only benefits on scalar correctness Brier, but its spline mapping worsens test AURC by creating ties. No method was selected as a winner, and no test-based tuning was performed.

The [protocol](flexible-calibration-protocol-20260930.md) was written before fitting/execution. Historical test results were already inspected in the temperature analysis, so these results are not a fresh confirmatory experiment. Three seed-17, historical development-NLL-selected five-epoch arms were analyzed. Metadata agrees on exported/selected checkpoint provenance; weight content identity remains unverified.

Calibration groups were deterministically partitioned into **727 mapping-fit rows** and **421 held-out assessment rows**. Maps were fit only on the former; thresholds were chosen only on the latter. IDs, ordered options, labels and groups match across arms, and calibration/test IDs and groups are disjoint. Serialized maps were saved before held-out scoring; thresholds were saved before test application. The aggregate JSON records prediction hashes and frozen protocol hash.

Raw identity, weighted isotonic PAVA and a ten-bin, Jeffreys-smoothed, monotone cubic Hermite spline are all reported. No dependencies were added. Scalar confidence predicts **whether the unchanged top-label decision is correct**; it does not replace class probabilities. Brier/log loss below are therefore binary correctness scores, **not multiclass Brier/NLL**, and cannot be compared directly with the temperature report's multiclass values.

## Interpretation

- Held-out calibration scalar Brier (raw → isotonic → spline): CE **0.18205 → 0.18309 → 0.18101**; AURC-only **0.16457 → 0.16393 → 0.16335**; mix **0.17996 → 0.18167 → 0.18120**. Benefits are small and inconsistent across arms.
- AURC-only spline improves test correctness Brier **0.20931 → 0.19416** and ECE **0.15517 → 0.08968**, while test AURC worsens **0.16966 → 0.19019**. Monotone maps cannot reverse confidence order, but plateaus and flat extrapolation introduce ties; our metrics average within ties.
- Unsmoothed isotonic maps can produce exact 0/1 confidences. Held-out errors at those endpoints cause large log penalties; scoring clips at 1e-12 as prespecified, without changing the map.
- No 5% assessment-selected operating point meets 5% test risk: observed risk is approximately **13–20%**. These are empirical thresholds, not guarantees. Dataset/variant composition shift is separately audited by the integration task.
- Raw threshold numbers differ from the temperature report because this experiment uses only the 421-row assessment partition for threshold selection. Do not interpret that difference as a mapping effect.

## Reproduce and outputs

```sh
python3 scripts/test_calibrate_confidence.py
python3 scripts/calibrate_confidence.py --out reports/flexible-calibration-20260930
```

Six synthetic checks pass: weighted pooling, tied inputs and step boundaries, monotonic bounded spline/flat extrapolation, constant spline, stable grouped partition and fit isolation, and AUGRC ties/order. The full pipeline completed successfully with provenance/pairing/separation assertions. [Aggregate metrics](../reports/flexible-calibration-20260930/aggregate.json), [frozen maps](../reports/flexible-calibration-20260930/frozen-mappings.json), and [frozen thresholds](../reports/flexible-calibration-20260930/frozen-thresholds.json) contain all nine variants, fit/assessment/test metrics and all 1/5/10% operating points. Raw exports remain untouched.
