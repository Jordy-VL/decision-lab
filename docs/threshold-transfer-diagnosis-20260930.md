# Threshold transfer diagnosis

The historical five-epoch fixed-slot models fail primarily on the test suite's
perturbed option sets among accepted predictions. This is descriptive evidence
from already inspected test outputs, not a new checkpoint-selection criterion
or a causal proof about the architecture.

Applied the existing calibration-fitted temperature and 5%-risk threshold for
each arm unchanged. No thresholds or models were refitted. Reproduce with
`python3 scripts/audit_threshold_transfer.py`; complete counts by source and
variant are in `reports/threshold-transfer-slices-20260930.json`.

| Test variant | Rows | CE accepted/errors/risk | AURC-only accepted/errors/risk | Mix accepted/errors/risk |
| --- | ---: | --- | --- | --- |
| clean | 1200 | 369 / 21 / 5.69% | 604 / 41 / 6.79% | 235 / 18 / 7.66% |
| permuted | 96 | 46 / 26 / 56.52% | 62 / 32 / 51.61% | 26 / 12 / 46.15% |
| none_absent | 72 | 27 / 25 / 92.59% | 38 / 32 / 84.21% | 10 / 10 / 100% |
| none_present | 72 | 24 / 21 / 87.50% | 33 / 27 / 81.82% | 6 / 5 / 83.33% |

The variant names above are copied from prediction provenance. Together the
240 perturbed examples contribute 72/93 accepted CE errors, 91/132 accepted
AURC-only errors and 27/45 accepted mix errors. Even the clean slice exceeds
5% empirical risk, so removing perturbations would not establish a guarantee.
The clean results must not replace the originally declared full-suite result.

Calibration exports have no `variant` field, which is reported as unspecified
rather than assumed clean. Source mixtures also differ: calibration includes
240 composition examples absent from this test export. Several test option
sets have an extra candidate (for example 78 rather than 77 Banking77 options).
These observations justify a robustness investigation; they do not isolate
the effects of source mix, option order, additional candidates or sample size.

## Consequences for the remaining work

- Keep scalar calibration separate from semantic robustness. A monotone map
  cannot repair a model choosing the wrong option after a permutation; it can
  only remap confidence and merge scores into ties.
- Use training/development perturbations or a separately declared future
  robustness suite for debugging. Do not optimize against these test errors.
- Audit option-order exposure during training and label remapping, then test
  paired permutations on development data. The candidate-head implementation
  audit found no normal-path index mismatch; its weak aggregate development
  result is a separate finding, not proof that it fixes this failure mode.
- Report source/variant counts and risk with the pooled metrics in future
  comparisons. Do not call calibration-target coverage a test-risk guarantee.
