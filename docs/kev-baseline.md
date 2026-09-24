# Frozen Kev baseline data

This is a **format adapter for the Kev-published decision-v7 suite**, not a new dataset, a TypeSafe-official training set, or a claim to reproduce Jev's undisclosed training mixture. Baselines come first; additional synthetic data and mixture weighting come later.

Run from the repository root (standard Python only; no GPU or credentials required):

```bash
python scripts/prepare_kev.py
```

This prepares `data/kev-decision-v7/{train,calibration,development}.jsonl`, keeps the original bytes in `raw/`, and writes hashes and counts to `preparation.json`. No test examples are downloaded by default. Once model and threshold choices are frozen, `python scripts/prepare_kev.py --allow-test` explicitly prepares the locked test as well. Do not use it for tuning.

The adapter pins both the [Kev source commit](https://github.com/jaredpalmer/kev/tree/90990a5fac2995b9faa3190f7d437e84f2067768) and its [published Hub mirror](https://huggingface.co/datasets/jaredpalmer/kev-suites/tree/a88f56db5341397299137cb68775c2ea6e3f68cb/v7/decision-v7). Training data is intentionally omitted from the Git repository; the same pinned Hub mirror used by Kev supplies it. The [upstream manifest](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/evals/v7/decision-v7/manifest.json) is itself SHA-256 pinned, and each downloaded partition must match its manifest checksum and record/question counts.

### Newer Hub suites

The Hub repository now also publishes `v8/decision-v8` and `v9/transfer-v9`.
They must not be silently substituted into the current baseline:

- `v8/decision-v8` is a successor decision-training suite. It adds the
  `shipping_delay` synthetic family and changes train/calibration data
  (15,688/1,164 questions versus v7's 15,576/1,148), while its development
  and test files are byte-identical to v7 (1,468/1,440 questions).
- `v9/transfer-v9` is evaluation-only: it has no training or calibration
  partition and adds holdouts including MMLU-Pro, buried-context and
  unknowable-confidence tests. It is a separate transfer evaluation, not a
  replacement training suite.

The current one-epoch results remain v7 results because changing to v8 would
change the training exposure and invalidate the matched-run comparison.
Future v8 experiments require a new pinned config, preparation directory and
full rerun of every objective arm. The Hub repository's immutable revision is
[`a88f56db5341397299137cb68775c2ea6e3f68cb`](https://huggingface.co/datasets/jaredpalmer/kev-suites/tree/a88f56db5341397299137cb68775c2ea6e3f68cb).

## What is preserved

- Original record/group IDs, source, metadata, state evidence, question instructions, option insertion order, labels and suite partition. Upstream dataset splits remain in provenance; they are not confused with the suite partitions.
- One canonical row per question; a multi-question request therefore becomes multiple training examples sharing a group. Original criteria and labels remain in provenance. This changes model input presentation, not labels or membership; our one-question classifier is not Kev's packed multi-question architecture.
- Boolean options are `false,true`. Categorical indices follow original option order. Choice names and descriptions are rendered together.
- Ordinal values are **zero-based level indices**, as specified by Kev's [API and labeling documentation](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/README.md#api). For example a five-star task retains criteria mentioning 1–5 stars but `values=[0,1,2,3,4]`; a derived expected value is the expected level index, not a star-rating unit. No numeric values are inferred from prose.

No sampling, source balancing, filtering, split recreation, or extra synthetic examples occur. The upstream suite already includes its own generated policy and compositional examples. Its manifest records ten public dataset revisions plus those controls. Source dataset terms still apply; Kev publishing the suite does not relicense all underlying data. Consult the upstream dataset cards before redistribution. Downloaded data stays ignored by Git.

## Verified preparation (2026-09-22)

| Partition | Original records | Canonical questions | Distinct groups | Boolean / categorical / ordinal | Maximum options |
|---|---:|---:|---:|---:|---:|
| Train | 12,576 | 15,576 | 10,868 | 5,224 / 6,904 / 3,448 | 77 |
| Calibration | 968 | 1,148 | 724 | 332 / 572 / 244 | 77 |
| Development | 1,204 | 1,468 | 880 | 472 / 756 / 240 | 78 |

All hashes and counts matched; all rows passed the actual trainer's `decisions.data.validate`. No duplicate question IDs, crossing group IDs, or identical canonical state text across these three partitions were found. Repeated groups *within* a partition are expected and preserved. These checks do not establish absence of fuzzy overlap or pretraining contamination. The manifest itself states that inherited legacy-test overlap with new synthetic controls is not certified; we have not inspected test to resolve that.

Use the default 128-slot index head, which covers the observed 78-option development cases. Audit token lengths with the selected tokenizer before training; do not silently truncate or drop examples just to fit a context limit. This preparation does not download model weights, train a model, or report model performance. It does not yet provide exact Kev benchmark aggregate metrics: raw per-question accuracy from our trainer must not be compared as if it were Kev's macro source aggregate.

Use development for model selection, calibration for temperature/threshold fitting, and locked test only for final evaluation. CE and subsequent AURC comparisons must share prepared data, tokenizer, backbone initialization, seed, updates, and evaluation partitions. Architecture, optimizer and full-fine-tuning differences from Kev remain disclosed comparison factors even with matched data.

## ModernBERT readiness and commands

The local official ModernBERT-large tokenizer audit, using the trainer's exact text rendering and special tokens, found maxima of 1,140 tokens (train), 1,078 (calibration), and 1,124 (development). All fit the configured 2,048-token limit without filtering. The 95th percentiles were 943, 883 and 975 respectively. This is a length audit, not GPU memory/throughput validation. Test was not downloaded or tokenized.

The earlier CE run is a completed preliminary baseline, but its checkpoint lives on the Modal Volume and its launcher predates the Trainer migration. It remains historical context, not the parent for X2. X2 now starts all three matched arms directly from the same pinned pretrained ModernBERT revision with fresh optimizers. From the repository root on `uab-gpu`, after installing the train extra:

```sh
uv run --extra train decisions check --config packages/modernbert-decisions/configs/ce-baseline.yaml
# Run this before the fresh-start X2 arms.
```

The nested config defines the one-epoch CE arm, not an exact replication of Kev's two-epoch LoRA recipe. The X2 objective configs initialize from the same pinned pretrained ModernBERT revision, use fresh matched optimizers, and automatically save best/final development and calibration logits. Threshold selection remains calibration-only; the test split is prepared separately after choices are frozen.
