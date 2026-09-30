# Development AUGRC selection, AURC and Cov@5% risk

Reviewed 2026-09-25 against the supplied evaluator, found locally at
`../src/app_benchmarks/confidence/evaluation.py`; the supplied
`src/app/_benchmarks/` path is absent.
SHA256: `b312327a97cf1f8c558ab9871191b9c790614c44665c5003b69b2dc20721c552`.

Re-reviewed 2026-09-30 against the expanded user-supplied folder
`C:\Users\jvl\Downloads\confidence` (WSL: `/mnt/c/Users/jvl/Downloads/confidence`).
Its `confidence/evaluation.py` has SHA256
`7602ff65d100b1838a0cb9e6b74e206c300a92dcaa12020807454c69d369f62c`.
It is a distinct, expanded snapshot; no version manifest establishes its release
date. The original review remains applicable to the unchanged legacy functions.

## September 30 expanded-reference review

Useful additions in the supplied snapshot:

- Tie-aware `augrc`, a separate closed form, and `auroc_failure` with half credit
  for ties. Both AUGRC forms match Decision Lab's existing implementation across
  1,000 seeded binary-correctness examples including ties and degenerate classes
  (maximum difference 3.33e-16). No AUGRC selection change is needed.
- Field/document auto-accept rates, accepted-error mass and conditional
  selective risk. Keep the denominators explicit:
  `coverage = accepted / total`, `error_mass = accepted_wrong / total`, and
  `selective_risk = accepted_wrong / accepted`. For nonzero coverage,
  `error_mass = coverage * selective_risk`. The source calls the first two
  validated-STP and validated-FP; neither means all accepted outputs are correct.
- Document gates accept only when every **observed** field clears the threshold.
  Before adapting this for extraction, define required-field completeness and
  missing-confidence handling; observed fields alone do not establish completion.
- Judge-confidence support, post-hoc logprob aggregation comparison and
  field/document confidence-method plots. `logprobs_utils.py` is byte-identical
  to the earlier supplied copy; these are additional surrounding tools.
- A standalone `compute_ecuas` helper, with n=0/1/4 names registered as scorers.
  The inspected field/overall reporting methods do not call or emit it. Its
  source describes a generative-extraction assumption; it is not adopted as a
  finite-choice Decision Lab metric by this review.

Confirmed limitations, with small numerical reproductions:

1. The original `geifman_AURC` denominator error remains: two wrong examples
   return 0.75 rather than 1.0. The original `calculate_coverage_at_risk` still
   divides by total N: 20 wrong examples produce 5% reported coverage at a 5%
   budget, but 100% error among accepted examples.
2. New `derive_per_field_tau` also uses `wrong / total`, not `wrong / accepted`,
   and can stop within a confidence tie. It must not set our selective-risk
   thresholds. On 20 wrong examples it accepts the highest-confidence one at
   the nominal 5% target, with 100% accepted error.
3. `optimize_thresholds_doc_ranked` can choose a safe partial prefix inside a
   tie, then derive a threshold that admits additional erroneous documents.
   It recomputes risk but does not enforce the requested budget afterward.
   `optimize_best` sorts by coverage without filtering infeasible candidates.
   Reproduction: one correct and one wrong document, both confidence 0.9,
   requested risk 0%; it selects both, reporting coverage 100%, risk 50%.
4. The greedy optimizer can stay at reject-all when admitting one good document
   requires loosening two field thresholds together. The subset-search helper
   tries three ranking heuristics; it is not exhaustive optimization.
5. Threshold evaluation skips a field completely when its threshold is missing,
   including its error. An empty threshold map thus reports 100% coverage and
   zero risk for a wrong document. No-acceptance risk is also encoded as zero,
   whereas our reporting convention is null. Require complete thresholds and
   explicit missing-data rules before reuse.
6. The optimizer CLI fits and reports on the same supplied metrics file. Treat
   its frontier as an in-sample diagnostic. Fit on calibration, serialize the
   threshold map, then evaluate unchanged on held-out documents for our protocol.

Other integration details: correctness exactly equal to `tau` is treated as a
failure by AURC/AUGRC (`c > tau`), but as correct by acceptance metrics/ECUAS
(`c < tau` flags failures). This does not affect our 0/1 labels at tau=0.5,
but needs normalization for fuzzy scores. The comparison plot keeps the newest
file per method label and takes its ground-truth ceiling from the first run;
match document IDs, extraction outputs and model configurations before treating
its curves as a controlled confidence-method comparison.

Evidence: [audit output and source hashes](../reports/supplied-confidence-audit-20260930.json).
Reproduce with `.venv/bin/python scripts/audit_supplied_confidence.py --source
/mnt/c/Users/jvl/Downloads/confidence --out reports/supplied-confidence-audit-20260930.json`.
The audit executes selected pure evaluator functions and the inspected optimizer
without its CLI. It does not initialize the tracker, fetch ECE modules, call
models or modify the supplied folder. This is targeted metric/threshold review,
not full judge/API or plotting integration validation.

## Reference review

Base the interface on `geifman_AURC`, `calculate_coverage_at_risk` and
`compute_field_metrics`: correctness/confidence pairs, AURC, coverage and its
confidence threshold. For decision tasks use exact index correctness (0 or 1)
and maximum option probability. Correct two calculation issues:

- `geifman_AURC` removes an example but divides remaining errors by `m - i`
  instead of `m - i - 1`. Two incorrect examples give 0.75 instead of 1.
  Default NumPy sorting also leaves tie treatment unspecified.
- `calculate_coverage_at_risk` divides cumulative errors by total `N`, not
  accepted count. This is accepted-error mass, not selective risk. It selects
  the first point reaching the budget, potentially exceeding it. For 20
  incorrect examples it reports 5% coverage at a 5% budget, despite 100%
  risk among accepted examples.

If exact reference parity is needed, label those outputs legacy metrics.
Do not overwrite existing Decision Lab results with the reference defects.

## Development checkpoint selection and AURC

Updated 2026-09-30: select checkpoints by development AUGRC (lower is better),
matching `make_training_arguments` in `decisions/hf_trainer.py`. Report NLL,
Brier and AURC alongside it at every development evaluation and for best/final
checkpoints. Preserve the original selection rule on historical results; this
correction does not retrospectively relabel NLL-selected checkpoints.
Compute on the full development set, not by averaging minibatch metrics.

For 0/1 error, the implementation defines AUGRC as
`(1 - AUROC_correctness) * accuracy * error_rate + 0.5 * error_rate**2`,
with half credit for confidence ties. It measures generalized risk, distinct
from selective-risk AURC below. Both are reported; only AUGRC selects new
checkpoints. AURC/AUGRC loss names do not change this shared selection rule.

Sort confidence descending. With `E_k` errors in the first `k` examples,
coverage is `k/N`, risk is `E_k/k`, and AURC is the mean of these risks for
`k=1..N`. Average over within-tie permutations for exact confidence ties,
matching `decisions/metrics.py` and `scripts/compare_runs.py`. This is not the
right-step integral over whole tie groups. Record the convention and report
pooled and per-task/source metrics with counts. Lower AURC is better.

## Coverage at a 5% selective-risk target

Fit calibrators and thresholds on calibration only. For each arm and each
raw/calibrated variant, evaluate all unique thresholds, accepting
`confidence >= threshold` and keeping whole tie groups. Compute risk as
`accepted_errors / accepted` and coverage as `accepted / total`.
Choose maximum calibration coverage subject to risk <= 0.05 and at least 30
accepted examples (the existing comparison-script default; freeze alternatives
before evaluation). Inspect every threshold: selective risk is not necessarily
monotonic, so do not use `searchsorted` or stop at the first violation.
Use the same procedure for the declared 1% and 10% targets.

Freeze the threshold and apply it unchanged to test. Report target risk,
threshold, calibration coverage/risk/count, test coverage and achieved risk,
and accepted/error/total counts. Label this **coverage at a calibration-selected
5% risk target**; achieved test risk can exceed 5%. A retrospective test
`Cov@5% risk` frontier is a separately labeled oracle diagnostic, never a
source of deployed thresholds or checkpoint selection.

If no calibration threshold qualifies, reject all: threshold null, coverage 0,
accepted count 0, risk null. If a frozen threshold accepts no test examples,
retain the threshold and report test risk null. Empirical feasibility is not a
statistical risk guarantee; include counts and uncertainty in low-risk claims.

Temperature scaling preserves winning classes but can reorder maximum-softmax
confidence across multiclass examples. Recompute AURC and thresholds after
calibration. Specify whether isotonic/spline mappings calibrate scalar
correctness confidence or full class probabilities; scalar confidence alone
cannot supply multiclass NLL or Brier.

## Implementation status

Presentation preference: [operational, interactive findings narrative](findings-presentation.md),
based on the LlamaIndex article the user selected September 30. Keep frozen
operating-point results distinct from exploratory threshold controls.

This reference update records the protocol and numerical audit without rerunning
training or model inference.
`decisions/metrics.py` implements tie-aware AURC and AUGRC, and Trainer
`metrics_from_logits` emits both alongside NLL, accuracy and Brier.
`scripts/compare_runs.py` implements temperature scaling and whole-tie
calibration thresholds. The September 30 [isotonic/spline comparison](flexible-calibration-20260930.md)
and [threshold-transfer diagnosis](threshold-transfer-diagnosis-20260930.md) are
now complete and include frozen operating-point counts. Those exploratory
results do not meet the 5% test-risk target. This source review changes neither
their numerical results nor the checkpoint-selection rule. Document-level
multi-field optimization remains a separate future extension; the supplied
optimizer is not installed as our evaluator.
