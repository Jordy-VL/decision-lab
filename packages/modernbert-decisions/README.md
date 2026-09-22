# ModernBERT decisions

A small text-only fine-tuner: official `answerdotai/ModernBERT-large` encoder, CLS pooling, and one new MLP predicting a bounded option index. State, question, and numbered natural-language options share one sequence. There are **no special option tokens**. Mask unused output slots before softmax and CE. Full encoder fine-tuning is the default.

The output always contains a **zero-based `index` and full `probabilities` over actual options**. One head handles:

| Primitive | Example options | Readout |
| --- | --- | --- |
| Categorical routing | shipping, billing, technical support | index 1 selects billing |
| Boolean / noul | false, true | index and `probability_true = probabilities[1]` |
| Ordinal rating | poor, acceptable, excellent; values `[1,3,5]` | index and expected value `sum(p * value)` |

`max_options: 128` accommodates Banking77. CLINC150 with an OOS class requires at least 151. Too many options are rejected. The checkpoint fixes this head capacity; changing it requires a new initialization. This intentionally simple index classifier can learn option-position biases, and is not permutation invariant. It does not use a dynamic per-option scoring head, generate text, or predict independent confidence. Numeric values are explicit ordinal rubric anchors; nominal class IDs are never interpreted as scores.

## Run

From this directory, with Python 3.10+ and uv:

```sh
uv sync
uv run decisions check --data examples/illustrative.jsonl
uv run decisions train --config configs/warmup.yaml
uv run decisions train --config configs/continue-ce.yaml
uv run decisions train --config configs/continue-aurc.yaml
uv run decisions evaluate --checkpoint runs/continue-ce/checkpoint --data examples/illustrative.jsonl --split test --output runs/eval-ce
uv run decisions evaluate --checkpoint runs/continue-aurc/checkpoint --data examples/illustrative.jsonl --split test --output runs/eval-aurc
uv run decisions predict --checkpoint runs/continue-ce/checkpoint --data examples/illustrative.jsonl --split test --output runs/predict
```

`pip install -e .` and `python -m decisions` also work. Training commands download the official encoder on first use. Example data is **illustrative, not a benchmark**. `check` validates schema/splits/capacity without loading a tokenizer; training and prediction check token lengths before running. Prediction accepts omitted targets. Choose a new output directory per invocation; accidental overwrites are rejected. No validation-based selection or early stopping is hidden in training; evaluate validation explicitly before using test results.

`configs/ce-baseline.yaml` uses optional `data`, `model`, and `training` sections and points to the prepared Kev training partition (run the root `scripts/prepare_kev.py` first). Legacy flat YAML and existing CLI flags remain supported; don't mix flat and nested keys in one file. CLI overrides still win. Saved resolved YAML remains flat for compatibility.

`Config` in `decisions/config.py` is the single field definition. YAML and generated argparse flags share it. CLI values override YAML; unknown fields/types fail. Boolean flags have `--no-...` variants. Paths are relative to the working directory. Every run saves all resolved fields, file SHA-256, supplied data revision, encoder revision when available, seed, package versions, selected split manifest, and initialization lineage. Pin `revision` to a Hub commit for reproducibility. `device: auto` selects CUDA, then MPS, then CPU. Training uses float32, AdamW, gradient checkpointing, and gradient clipping at 1.0; no AMP, scheduler, or trainer framework.

## Data contract

One JSONL row per question:

```json
{"id":"doc1/urgency","group_id":"doc1","source":"my-corpus","split":"train","state":"Customer cannot sign in.","question":"How urgent is this?","options":["routine","soon","immediate"],"values":[1,2,3],"target_index":1,"type":"ordinal"}
```

Types are `categorical`, `boolean`, `ordinal`. Boolean options must be exactly `["false","true"]`. Ordinal `values` must be finite, strictly increasing, and aligned with options. `evidence`/`provenance` are optional and preserved in predictions; they are never fed to the encoder. The sibling generator's `trainer.jsonl` is directly compatible.

Existing split labels are preserved literally (including `development` or `calibration`); select them with `--split`. A standalone pre-split file without row split labels can use `--data-split train` or `--data-split test`. Conflicts and groups crossing splits within the loaded file fail. With no splits, a single seed+group hash assigns train/validation/test (80/10/10 by default); small datasets can have empty partitions, which fail when selected. Mixed missing/assigned splits fail. Supply document-level group IDs across questions/variants; the loader cannot infer duplicate documents across separate input files. It keeps the JSONL in memory, suitable for a pilot.

Kev **labeled requests** (`state`, `questions` mapping, `type`, `instructions`, `criteria`, `label`) are adapted to independent rows. `_meta` IDs/groups/source/split are retained, structured text is serialized as JSON, boolean criteria are appended to the question, choice labels map through dictionary insertion order. Add explicit `values` to every Kev score question, e.g. `[1,2,3,4,5]` for stars: the adapter refuses to guess scales from level indices. Kev internal materialized records lack reliable task semantics and are rejected. This adapter is schema-compatible, not byte-identical to Kev's rendering or trainer.

Length accounting includes state/question headings, numbered option text, and encoder CLS/SEP tokens. No answer, state, or option is silently truncated. An overlong row raises its ID plus section counts; all selected rows are encoded before updates. `max_length` must fit the encoder context. Banking77's 77 descriptions may require a larger context than short boolean tasks. `max_options` is a separate capacity limit.

Preserve official benchmark partitions when preparing Banking77, BoolQ, MNLI, SST-5, Yelp or CLINC data; this package does not download/reshape them for you. Original RVL-CDIP is excluded; RVL-CDIP-N is eligible for evaluation only here (public card has only test). Frozen Decision Index is evaluation-only. Training rejects these named sources and rows marked `evaluation_only`; this is a guard, not a replacement for provenance review. DUDE-derived tasks and vision/fusion remain future work.

## Matched CE vs CE+AURC pilot

Use one CE warm-up checkpoint, then start both provided continuation configs from it with identical seed, data, batch size, epochs and optimizer settings. They differ only in lambda and output path. Continuation is explicitly **weights-only**: encoder/tokenizer/head load, while AdamW, shuffle state and RNG start fresh. There is no exact optimizer resume. This gives matched fresh continuation arms; neither arm starts from the other. No mandatory grid, repeated seeds, or cross-validation.

For actual microbatch size B, ascending detached confidence rank r (1..B), and CE per row:

```text
confidence = max softmax probability
alpha = H_B - H_(B-r)
loss = (1-lambda) * mean(CE) + lambda * mean(alpha * CE)
```

Lambda defaults to 0; the example AURC arm uses 0.5. Confidence ties retain batch order for training. The final partial microbatch uses its actual size. Accumulation averages gradients over actual examples but does **not** enlarge the ranking batch. Optional `--rank-by-type` ranks separately within each microbatch's task types; singleton groups reduce to CE. Maximum probability is not necessarily comparable across binary, 77-way, and ordinal tasks (or across sources); report per-type/source metrics and keep batch composition matched. The harmonic formula follows [AsymptoticAURC's source](https://github.com/han678/AsymptoticAURC/blob/main/utils/loss.py), independently implemented without SciPy.

Evaluation saves `predictions.jsonl` (labels, logits, probabilities, confidence, IDs/groups/source, token counts and ordinal readouts) and `report.json`: accuracy, log-softmax NLL, multiclass Brier, discrete AURC and ordinal expected-value MAE, overall and by type/source. AURC averages 0/1 error risk at coverages 1/n through 1, treating exact confidence ties as the expectation over within-tie permutations. It measures selective prediction, **not calibration**. Teacher confidence is not assumed calibrated; temperature fitting and richer statistics are deferred. Mixed ordinal scales should be interpreted by source.

The research hypothesis is **higher coverage at a fixed low error risk**, not merely improved calibration. Saved logits/labels support later raw and temperature-scaled CE/AURC comparisons, risk–coverage curves, reliability plots, and risk/coverage versus thresholds. Select operating thresholds on validation and apply them unchanged to test. Threshold spacing or the number of numerical thresholds is not invariant utility: a monotonic confidence transformation can change spacing without improving the ranking. No hypothesis is confirmed by this implementation or its smoke checks.

Inspiration: [Dev model card](https://huggingface.co/mpnikhil/dev-0.4b), its [config](https://huggingface.co/mpnikhil/dev-0.4b/blob/main/config.json) and [run metadata](https://huggingface.co/mpnikhil/dev-0.4b/blob/main/run.json), plus [Kev data](https://github.com/jaredpalmer/kev/blob/main/kev/data.py), [API schema](https://github.com/jaredpalmer/kev/blob/main/kev/api.py) and [suite](https://github.com/jaredpalmer/kev/blob/main/kev/suite.py). Dev's linked repository returned 404 on 2026-09-22; this is not a Dev reproduction. The index head is an intentional simplification of its dynamic-choice approach.

No unit-test suite is included. See `SMOKE.md` for the actual lightweight checks performed during implementation.
