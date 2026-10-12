# Decision Index: execution, submission and cost

## Completed October 1: fixed-slot AURC

The full edition 0.2.1 run completed at 14:36 CEST. Official Decision Index:
**1.48/100**, balanced raw score **22.62/100**, breadth skill **1.45/100**.
All 150,317 scoreable requests are accounted for: 138,814 successful responses,
11,503 unsupported, zero runtime errors or missing requests. Unsupported cases
comprise 7,500 exceeding 128 options and 4,003 exceeding 2,048 tokens; they remain
in the official scoring denominator. Inference attempted 150,759 requests,
including 442 official exclusions, and took 9,183.7 seconds (2h 33m).

Official base/added suite hashes, exclusions and subsets verified successfully.
The [retrieved result bundle](../reports/decision-index-results-20261001.json)
contains complete scores, per-benchmark breakdowns, environment, verification,
checkpoint/source provenance and final status. Transfer SHA-256:
`2022dc68ce8b14fe0f1ecff8c94bcf6723a406a7b8e9c1705a131da8940c48df`.
Detailed request outputs remain on the server in the job's `full/` directory.

This is a poor external-suite result for this checkpoint and adapter. Capacity
limits are only part of it: fully supported MMLU, HellaSwag and WinoGrande scores
are approximately chance (25.41%, 25.21% and 49.8%). Investigate input formatting,
slot behavior and domain transfer before interpreting this as an AURC-loss
effect; there is no matched CE Decision Index run here. The headline is a
chance-adjusted task-performance index, not the AURC metric itself. Nothing
has been uploaded or submitted to the leaderboard.

The [completed transfer diagnosis](decision-index-transfer-diagnosis-20261001.md)
finds strong slot-position dependence in both AURC and CE. Original predictions
reproduce exactly; restoring training-style JSON spacing does not rescue the
diagnostic. See that report before launching a new full evaluation.

## September 30 full fixed-slot AURC job

The user requested full inference and scoring on Decision Index. Selected the
latest completed fixed-slot AURC-only 10-epoch linear-1e-5 checkpoint, seed 17,
development-AUGRC-selected at step 20,500 (best metric 0.0923790176).
This is the AURC-trained arm, distinct from the AUGRC-trained example below.
No calibration is fitted on Decision Index; this job uses the checkpoint's raw
probabilities, 128-option capacity and 2,048-token input limit.

Pinned kit: `apolinario/decision-index` at
`87d4650b42b377c0291a89c1f1a879f9b31082bf`, edition 0.2.1. Its scorer expects
119,898 base scoreable requests plus 30,419 added requests (150,317 total),
with 38 benchmarks contributing to the headline index. This supersedes the
historical request-count estimate below. The public kit reconstructs the suite
from source datasets; it does not supply an unrestricted public suite download.

Four adapter tests pass locally and remotely. A synthetic 19-question check
with the real checkpoint on an A40 GPU passes the kit's response validation;
batch-eight versus batch-one probability difference is at most 2.468e-8.
This is an interface check, not a benchmark result. Questions are processed in
bounded batches of eight; overlength/overcapacity requests remain unsupported
under the official scoring rules rather than being truncated or dropped.

The resumable coordinator is exposed as `jobs/quickstart.sh eval-decision-index-full`;
see [job instructions](../jobs/README.md). It freezes checkpoint/source hashes,
verifies the suite, runs a diagnostic sample and then performs full inference
and official scoring. Existing outputs require explicit `--resume` and identical
input hashes. Results are not uploaded or submitted.

Initial setup encountered a missing Git LFS executable in the server's dataset
checkout. The generated checkout was repaired after verifying its files against
the fetched commit. Job-scoped Git filters leave LFS pointers for the official
builder to replace using its own hash-verified ESCI downloads. Kit source and
suite hashes are not relaxed.

HLE access preflight returned HTTP 403 for Hugging Face account `jordyvl`.
The user was asked to enable access at https://huggingface.co/datasets/cais/hle.
Do not claim a full evaluation or score until the suite build and scorer finish.

The end-to-end job was launched through quickstart at 19:00:17 UTC, then
stopped at 19:01:27 UTC during reconstruction on that confirmed HLE denial.
No official suite inference or score was produced. Verified local
[setup evidence](../reports/decision-index-setup-20260930.json) includes checkpoint
and source hashes, launch details, failure status and the real-checkpoint smoke.
Public-source prefetch continued separately to cache the remaining ingredients.

After access is enabled and any prefetch process has finished, resume from
the remote Decision Lab directory on an authorized idle GPU:

```sh
HF_HOME=/home-local/sbiswas/.cache/huggingface-decision-lab \
CUDA_VISIBLE_DEVICES=5 jobs/quickstart.sh eval-decision-index-full \
  --checkpoint runs/x2-aurc-only-10epoch-linear-1e5/checkpoint \
  --run-name fixed-slot-aurc-10epoch-20260930 --resume
```

Use a fresh external console-log name for each attempt. The coordinator's
`job-status.json` identifies the current phase; `full/scores.json` is the final
official score only after successful completion. GPU 6 was occupied during the
September 30 setup and was not used. These artifacts remain local/private; this request does not
authorize a leaderboard submission.

## October 1 continuation

HLE access for `jordyvl` was verified after the user enabled it. Public-source
prefetch finished successfully. The first retry inherited the server shell's
different `HF_HOME` and failed authorization; the next attempt explicitly set
the verified project cache shown above. No credentials were copied into code.

Resumed through quickstart at 09:43:16 UTC on authorized, idle GPU 5, PID
1691159. Checkpoint/source/configuration hashes matched the existing manifest.
Console: `runs/decision-index-full-aurc-20261001-retry1-console.log`.
Suite reconstruction is in progress; no official score is available yet.
The [11:54 CEST continuation snapshot](../reports/decision-index-continuation-20261001.json)
confirms an active builder at BRIGHT retrieval after completing the earlier
normalizers. Full inference and scoring remain queued behind suite verification.

At 14:23 CEST, official suite verification had passed (base uncompressed hash,
added rows, exclusions and subsets all matched). The sample completed with
17 supported requests, three unsupported requests and no errors. Full inference
started at 12:02 CEST and reached 139,460 of 150,759 attempted requests (92.5%):
127,659 successful, 11,801 unsupported, zero runtime errors. The unsupported
breakdown was 7,500 option-capacity cases and 4,301 token-length cases.
The attempted count includes 442 official exclusions; the scoring denominator
remains 150,317. Final scoring had not yet run. These are progress counts,
not model-quality results.

## Historical September 22 review

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
