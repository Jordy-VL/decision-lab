# Is a numeric output a useful fourth primitive?

Status: proposed research direction, not implemented. Keep the first experiment focused on Choice, Boolean, and Score.

## Three different problems

| Problem | Example | Minimal mechanism | Does it need a new primitive? |
|---|---|---|---|
| Select a number present in evidence | Which amount is the invoice total? | Extract candidates in code; choose an index | No: existing Choice, with candidate-recall evaluation |
| Apply a bounded numerical rubric | How many of three required fields are present? | Ordered levels 0, 1, 2, 3 | No: existing Score |
| Predict an unrestricted/continuous quantity | Estimate delivery time | Regression or a predictive distribution with units/range constraints | Potentially, but a different modeling/evaluation problem |

Returning an expected ordinal value already produces a number, but it does not establish calibrated continuous regression. An average between ordinal levels can also have weak semantic meaning if level spacing is not meaningful.

## Recommendation

First add numeric-candidate tasks to the existing data format. Store candidate values and units as metadata and return the selected value through deterministic postprocessing. Include “not present” only when supported by the task semantics. Report missing-candidate errors rather than hiding them behind classification accuracy. Exact arithmetic and deterministic normalization stay in code.

If a later use case requires continuous estimation, compare bounded bins with a regression/distribution head. This may violate the single fixed index-head simplicity goal. Evaluate MAE or a domain loss, selective risk using that same loss, and predictive-interval coverage/width if intervals are produced. Do not describe an arbitrary scalar confidence as a calibrated predictive distribution.

AURC can rank examples by a confidence score while measuring nonbinary loss, but the ranking score, units, and error costs need explicit definitions. Neither low ECE for classification nor low 0/1 AURC establishes reliable numerical predictions.

No additional numeric head is planned until a concrete task, labels, and measurable benefit justify it.
