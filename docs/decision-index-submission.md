# Decision Index: execution, submission and cost

Checked 2026-09-22 against the [published reproduction/submission instructions](https://github.com/apolinario/decision-index). This concerns the community dashboard, not TypeSafe's workflow evaluation.

The linked [Hugging Face Space](https://huggingface.co/spaces/multimodalart/jev-decision-index) is a static leaderboard/tracker, not an interactive endpoint for uploading a checkpoint and running inference. Its README points contributors to the separate [Decision Index kit](https://github.com/apolinario/decision-index): self-run evaluation, publication of complete result artifacts, then a PR for review/re-scoring. Maintainers do not promise to run uploaded weights for free. The current Space describes Decision Index 0.2.1; pin the kit's suite edition and revisions before launching, since its frozen benchmark bundle is distinct from our Kev v7 data.

For this workspace's custom ModernBERT wrapper, the operational route is:

1. Choose a completed checkpoint and export/package the matching `checkpoint/`
   folder with its tokenizer, `model.json`, and `head.pt`; the intermediate HF
   `trainer/checkpoint-*` folders alone do not contain the project's
   `DecisionModel` wrapper metadata.
2. Implement an adapter from the kit's typed `state`/`questions` requests to
   our one-question model inputs. Return the answer under the original option
   key and a probability for every option. Preserve one result per question,
   request/question IDs, and original options; do not drop inputs exceeding
   our 128-option capacity.
3. Run the kit locally against the frozen suite and generate its required
   result artifacts. Start with a small fixed diagnostic sample and throughput/
   capacity audit; a sample is not a leaderboard score.
4. Upload the model and reproducibility/artifact links as appropriate, then
   submit the result bundle through the kit's documented contribution process
   (PR/re-scoring). Do not expect the static Space itself to execute weights.

The hosted page currently lists models as tracker/leaderboard entries, but a
listing is not a model-execution API or proof that a result was evaluated
under our configuration. Follow the kit's current submission schema rather
than inventing a direct Space upload flow.

### Minimal sample run with an existing checkpoint

The project provides a custom Decision Index engine in
[`scripts/jev_decision_index_engine.py`](../scripts/jev_decision_index_engine.py).
It loads one project-format checkpoint, converts every typed question into
the project's existing model input, returns probabilities under the original
option keys, and declares option/token-capacity overflows unsupported.

Install the Decision Index kit in the same Python environment as this project
so its `decision_index` package and the project's PyTorch/Transformers
dependencies are importable. Then, from the kit checkout:

```sh
python -m decision_index suite sample --n 20 --out /tmp/decision-index-sample.jsonl.gz

PYTHONPATH=/data/133-1/users/sbiswas/experiments/decision-lab/scripts:/data/133-1/users/sbiswas/experiments/decision-lab/packages/modernbert-decisions \
python -m decision_index run \
  --engine jev_decision_index_engine:ModernBERTDecisionEngine \
  --option checkpoint=/data/133-1/users/sbiswas/experiments/decision-lab/runs/x2-augrc-only-5epoch-linear-1e5/checkpoint \
  --option device=cuda \
  --rows /tmp/decision-index-sample.jsonl.gz \
  --out /tmp/modernbert-decision-index-sample \
  --limit 20 --compact
```

This is a smoke/sample evaluation, not a leaderboard score. Review
`status.json`, `environment.json`, `results.jsonl` and the suite scorer output;
check unsupported/error counts before scaling to the full suite. Do not run
the full suite until the kit revision/edition, hardware budget and capacity
gaps are explicitly accepted.

The current kit documentation describes 132,422 requests across 37 benchmarks
for its tracked suite; the live Space currently advertises Decision Index 0.2.1
with an edition-specific benchmark panel. Verify counts and evaluation rules
from the exact pinned kit edition used for the run before budgeting.

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
