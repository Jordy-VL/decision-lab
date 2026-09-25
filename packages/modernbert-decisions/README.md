# ModernBERT decisions

A small text-only fine-tuner: official `answerdotai/ModernBERT-large` encoder, with `decision_head=fixed_slot` selecting from a bounded option-index head by default. State, question, and numbered natural-language options share one sequence. Mask unused output slots before softmax and CE. Full encoder fine-tuning is the default.

Set `decision_head: candidate_masks` to use the alternative parameterized input path. Each option is prefixed with a dedicated candidate marker token; the model gathers the contextual representation at each marker and applies the same scalar scorer to every candidate. Candidate positions are padded and masked by the collator. Candidate checkpoints save the added marker-token embedding and the selected head in `model.json`; older checkpoints without this field continue to load as `fixed_slot`.

The output always contains a **zero-based `index` and full `probabilities` over actual options**. One head handles:

| Primitive | Example options | Readout |
| --- | --- | --- |
| Categorical routing | shipping, billing, technical support | index 1 selects billing |
| Boolean / noul | false, true | index and `probability_true = probabilities[1]` |
| Ordinal rating | poor, acceptable, excellent; values `[1,3,5]` | index and expected value `sum(p * value)` |

`max_options: 128` accommodates Banking77. CLINC150 with an OOS class requires at least 151. Too many options are rejected. The fixed-slot checkpoint fixes this head capacity; changing it requires a new initialization. The candidate-mask scorer shares parameters across options and still uses `max_options` as the batch/output capacity. Numeric values are explicit ordinal rubric anchors; nominal class IDs are never interpreted as scores.

## Run

From this directory, with Python 3.10+ and uv:

```sh
uv sync
uv run decisions check --data examples/illustrative.jsonl
uv run decisions train --config configs/ce-baseline.yaml
uv run decisions train --config configs/x2-ce.yaml
uv run decisions train --config configs/x2-aurc-only.yaml
uv run decisions train --config configs/x2-ce-aurc-mix.yaml
uv run decisions evaluate --checkpoint runs/continue-ce/checkpoint --data examples/illustrative.jsonl --split test --output runs/eval-ce
uv run decisions evaluate --checkpoint runs/continue-aurc/checkpoint --data examples/illustrative.jsonl --split test --output runs/eval-aurc
uv run decisions predict --checkpoint runs/continue-ce/checkpoint --data examples/illustrative.jsonl --split test --output runs/predict
```

`pip install -e .` and `python -m decisions` also work. Training commands download the official encoder on first use. Example data is **illustrative, not a benchmark**, and is not a training fixture. `check` validates schema/splits/capacity without loading a tokenizer; training and prediction check token lengths before running. Prediction accepts omitted targets. Training requires separate train, development and calibration files with explicit partitions. Choose a new output directory per invocation; accidental overwrites are rejected.

`configs/ce-baseline.yaml` uses optional `data`, `model`, and `training` sections and points to the prepared Kev training partition (run the root `scripts/prepare_kev.py` first). Legacy flat YAML and existing CLI flags remain supported; don't mix flat and nested keys in one file. CLI overrides still win. Saved resolved YAML remains flat for compatibility.

`Config` in `decisions/config.py` is the single field definition. YAML and generated argparse flags share it. CLI values override YAML; unknown fields/types fail. Boolean flags have `--no-...` variants. Paths are relative to the working directory. Every run saves resolved fields, data hashes/revisions, encoder revision, seed, package versions and initialization lineage. Pin `revision` to a Hub commit for reproducibility. Training uses Hugging Face `Trainer`/`TrainingArguments` with AdamW, a 5% warm-up then cosine decay to 10% of peak LR, gradient clipping at 1.0, standard resumable checkpoints and local JSON logs (`report_to=none`). `device: auto` lets Accelerate select the available device; `device: cpu` forces CPU. Mixed precision is opt-in.

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

## X2 objective comparison

Use one frozen CE checkpoint, then start the three X2 configs from it with identical seed, data, model and optimizer settings: CE-only (`aurc_lambda: 0`), AURC-weighted CE surrogate only (`1`), and their blend (`0.5`). Each arm starts a fresh optimizer/scheduler; none starts from another arm. Exact interruption recovery within an arm uses the Trainer checkpoint path in `resume`.

The surrogate ranks detached maximum-softmax confidence within each actual microbatch. Gradient accumulation does not enlarge the ranking batch. “AURC-only” means the rank-weighted CE surrogate, not direct optimization of discrete AURC.

Each train run requires `development_path` and `calibration_path`. Trainer selects the best checkpoint by development NLL, saves best and final-budget model folders, and writes `.npz` raw logits/labels plus paired JSONL predictions for both development and calibration. `evaluations/best/` and `evaluations/final_budget/` each contain predictions, logits, report and metadata. Temperature and thresholds are still fitted on calibration only; final test remains a separate, frozen action.

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
