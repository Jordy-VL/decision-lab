# Decision Index: execution, submission and cost

Checked 2026-09-22 against the [published reproduction/submission instructions](https://github.com/apolinario/decision-index). This concerns the community dashboard, not TypeSafe's workflow evaluation.

The documented route is self-run evaluation, upload of complete result artifacts, then a PR for review/re-scoring. Maintainers do not promise to run uploaded weights for free. There are 132,422 requests across 37 benchmarks; 19 feed five equally weighted areas. Unsupported and unanswered requests are penalized; inputs/options cannot be silently removed.

## Proposed sequence

Establish the matched Kev-data baseline first. Next implement a harness adapter that maps our index/probabilities back to original option keys and declares capacity. Run a small fixed diagnostic sample preserving linked cases, measure throughput and capacity failures, then decide whether to fund a full sweep. Keep the external suite out of training. A sample is not a full leaderboard result; submission is a later publishing action.

Our current head defaults to 128 options and the pilot length is smaller than the encoder context limit. If 255 slots are needed, select that capacity before training. One request may contain several questions, each separately encoded by our current model. Requests/second and questions/second are not interchangeable. Preserve model, code, data and hardware provenance.

## Conditional costs

An A100 40GB with two CPU cores and 16 GiB RAM is about $2.32/hour at the checked [Modal base rates](https://modal.com/pricing), before other charges/credits. If throughput includes all question processing and runner overhead:

| Whole-request throughput | Full-suite time | Approximate compute |
|---|---:|---:|
| 1 per second | 36.8 hours | $85 |
| 10 per second | 3.68 hours | $8.5 |
| 50 per second | 44 minutes | $1.7 |

These are arithmetic scenarios, not measured ModernBERT forecasts. Include setup, downloads, scoring/storage and any external API calls in the budget. An available CVC allocation may avoid cloud charges; we still need to measure resource use.

The headline index measures task performance. Our CE/AURC claim separately needs risk-coverage, calibration and frozen-threshold domain-shift results. Do not infer selective utility from leaderboard placement.
