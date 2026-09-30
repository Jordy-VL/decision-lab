# Candidate-head audit — 2026-09-30

The candidate scorer is connected end to end. This is an implementation audit,
not a trained-quality result or an exact Laya replication. V1 remains the
fixed-slot reference; V2 remains a separate architecture experiment.

## Local path

- `initialize` registers `<decision_candidate>` as an added special token and
  resizes encoder embeddings. Saved checkpoints own the head architecture and
  retain tokenizer/encoder state.
- `encode` inserts one marker before each numbered option, retaining input
  option order. It records token positions, rejects marker collisions in source
  text and refuses overlength sequences rather than truncating candidates.
- `collate` right-pads tokens, preserves positions and masks unused option slots.
  Padding positions are zero but their scores are masked to negative infinity.
- `DecisionModel.forward` gathers each marker hidden state, applies one shared
  Linear/GELU/Linear scalar scorer and masks unused slots. It does not pool CLS
  for candidate decisions. The name `candidate_masks` does not mean literal
  pretrained `[MASK]` tokens.
- `make_collator` preserves `target_index`; `DecisionTrainer` removes labels and
  task types before the forward call, then applies the selected CE/AURC/AUGRC
  objective. There is no label shift or extra confidence head.
- Data validation enforces zero-based targets within available options;
  categorical criteria retain insertion order, Boolean options are false/true,
  and ordinal anchors retain increasing numeric order.

Two defensive fixes prevent malformed inputs from silently selecting wrong
positions: reject an unknown/unregistered marker token, and reject missing,
out-of-range or wrong-count positions within a candidate batch. Normal encoded
inputs and existing checkpoints retain their behavior.

## Pinned primary-source comparison

Inspected Laya [`laya/common.py`](https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/laya/common.py)
at revision `9d955671415fc19f069b9cc998928075c1f255ec`, specifically
`build_sequence`, `DecisionModel`, and `proper_reward`.

| Aspect | Decision Lab V2 | Pinned Laya |
|---|---|---|
| Marker | Added special token | Existing tokenizer mask token |
| Sequence | State, question, numbered options | Typed question, options, state, separated by SEP |
| Length | Reject overlength | Budget and truncate options/state |
| Post-backbone processing | Shared two-linear-layer MLP | Type embedding and normally two TransformerEncoder layers, then LayerNorm and scalar MLP |
| Extra outputs | None | Action head and per-type temperature buffer |
| Option padding | Fixed configured capacity, negative infinity | Marker mask, negative 10,000 |

The [pinned fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
uses soft target distributions and a proper-scoring-rule policy-gradient recipe.
It installs `laya>=0.1.6` and downloads an unpinned model, so pinning that notebook
alone does not pin its runtime implementation or weights. Our hard-label losses
and Kev suite are separate experiments.

## Verification and remaining research work

Passed all 13 package tests on CPU:

```sh
PYTHONPATH=packages/modernbert-decisions .venv/bin/python -m unittest discover -s packages/modernbert-decisions/tests -v
```

New tests exercise an actual fast-tokenizer implementation, variable option
counts, padding, label alignment, shared scorer gathering, zero probability and
gradient for invalid slots, finite CE/AURC/AUGRC losses with correctly directed
target gradients, collision rejection, malformed position rejection, and a
tiny Transformers encoder/tokenizer checkpoint round trip with identical logits.
These use local synthetic fixtures and a randomly initialized tiny BERT; no
pretrained ModernBERT download, GPU training, or quality claim was made.

Before X12 conclusions: run matched V1/V2 CE training and frozen evaluation with
the same data, seed, schedule and checkpoint selector. Report the marker and
post-backbone differences explicitly; do not label V2 a reproduction of Laya.
