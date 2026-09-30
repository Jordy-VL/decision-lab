# Jev article: pinned-source follow-up — 2026-09-30

**Verdict:** useful technical lead, not independent evidence of Jev equivalence or superior model quality. The author's engineering claims have a real source trail. Trust individual claims according to their evidence; do not use the headline as a research conclusion. Disorganized writing is not itself evidence of dishonesty.

The [article](https://martinschroder.substack.com/p/i-learned-how-jev-works-and-went) explicitly disclaims access to Jev's weights, exact architecture, and benchmark equivalence. It distinguishes several training procedures and acknowledges local experimental limitations. Its caution about a clipped scoring reward is technically justified by the independent numerical check below. Performance claims remain first-party reports. Company affiliation establishes provenance, not independent validation; see the [initial review](review-20260929.md).

## Scope and immutable sources

Read-only source inspection used GitHub's API and raw files. The previously inaccessible Brain tree was successfully retrieved. The article was readable through the search index; direct HTTP access returned 403. No remote code was executed, no model weights were downloaded, and no training or benchmark was reproduced.

- Brain: `180a8890c6b8450af2936c7739aa8cdaaab631c1`, the article's pin.
- Laya: `9d955671415fc19f069b9cc998928075c1f255ec`, the reference used in the initial review.
- Binary-fix commit resolved to `57e982923b3063fe8fc5ed10b5185fbd490b8b7a`.

## What the code supports

| Claim / issue | Evidence and assessment |
| --- | --- |
| Brain's original decision head differs from Laya | **Source inspected.** Brain gathers candidate CLS states, projects candidate queries and state keys/values, performs cross-attention, adds a query residual, normalizes and applies a scalar linear scorer. This is a late-interaction head, not the Laya head. [Brain head, lines 368–409](https://github.com/swedishembedded/brain/blob/180a8890c6b8450af2936c7739aa8cdaaab631c1/crates/decide/src/head.rs#L368). |
| Laya uses typed marker readouts | **Source inspected.** Default head depth is two. Type embeddings are added to backbone states before extra pre-normalized transformer layers. Marker states are gathered and scored with LayerNorm/Linear/GELU/Linear; invalid candidates are masked. The action head is a separate output. [Laya common.py, lines 278–340](https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/laya/common.py#L278). |
| Public Laya fine-tuning is pure RL | **False for this notebook.** It combines sampled-logit reward optimization with coefficient-1 soft-target CE. Four Gaussian samples use zero-mean projection across valid options; sigma anneals from 0.4 to 0.1. Rewards use spherical weight 0.75 and ordinal weight 1.0. Inspect the training loop in the [pinned notebook](https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb). This does not establish how original pretraining was performed. |
| Brain implements exactly the same update | **Qualified.** Brain explicitly documents per-question reward standardization, whereas the notebook centers across samples per question then divides by a standard deviation across the microbatch. The distinction changes relative update scaling. [Brain reinforce.rs](https://github.com/swedishembedded/brain/blob/180a8890c6b8450af2936c7739aa8cdaaab631c1/crates/rlcd/src/reinforce.rs#L59). |
| The implemented reward is globally strictly proper | **Contradicted by an independent scalar check.** Both implementations floor the log score at -9.21 while describing the reward as strictly proper. The flooring changes the optimum for sufficiently rare events; see below. [Laya reward](https://github.com/NandhaKishorM/laya/blob/9d955671415fc19f069b9cc998928075c1f255ec/laya/common.py#L410), [Brain reward](https://github.com/swedishembedded/brain/blob/180a8890c6b8450af2936c7739aa8cdaaab631c1/crates/rlcd/src/proper.rs). |
| Binary-softmax repair establishes a general sigmoid advantage | **No.** The [fix commit](https://github.com/swedishembedded/brain/commit/57e982923b3063fe8fc5ed10b5185fbd490b8b7a) records improved measurements and a common-mode cancellation diagnosis in that setup. The measurements were not reproduced here. Algebraically, two-logit softmax probability equals sigmoid of the logit difference; replacing a head changes parameterization and learning behavior, not this identity. |
| Public parity tests establish independently reproduced parity | **No.** The real-weight parity test returns early when model weights or reference fixtures are absent. Inspectable tests are useful, but a passing invocation without fixtures would not establish real-model parity. [Parity test, lines 96–116](https://github.com/swedishembedded/brain/blob/180a8890c6b8450af2936c7739aa8cdaaab631c1/crates/modernbert/tests/laya_real_parity.rs#L96). No such test was run in this review. |

## Independently checked: reward counterexample

Evaluated the source's non-ordinal formula with Python standard-library float64 arithmetic, target `t = [0.999999, 0.000001]`, spherical weight `0.75`, and log floor `-9.21`:

```python
import math
t = [1 - 1e-6, 1e-6]
def reward(q):
    log_score = sum(a * max(math.log(b), -9.21) for a, b in zip(t, q))
    spherical = sum(a * b for a, b in zip(t, q)) / math.sqrt(sum(b*b for b in q))
    return log_score + 0.75 * spherical
print(reward(t))
print(reward([1 - 5e-7, 5e-7]))
```

Truthful report: **0.749989040001**. Halved rare-event report: **0.749989540001**. A non-truthful report earns more, refuting the global properness guarantee. This reproduces the mathematical issue, not either framework's full runtime or model behavior. It does not establish the size of any practical calibration error.

## Decision Lab actions

1. Keep the candidate-head audit separate from objective claims: matching marker extraction alone does not reproduce Laya's typed transformer head or learning algorithm.
2. Preserve matched CE controls, development checkpoint selection, separate calibration fitting and untouched final evaluation. Neither sampled-reward training nor an architecture diagram guarantees calibration.
3. Do not replace categorical binary scoring solely because of Brain's reported failure. First inspect logit differences, candidate-state similarity and learning behavior in our own matched runs.
4. Treat the clipped reward as a distinct empirical objective if explored later; include rare-event checks and record its floor. Do not label it a theorem-backed calibration guarantee.
5. Jev-equivalence, speed, generalization, sales results and actual Brain/Laya numerical parity remain **unverified** here. Their validation would require pinned weights, datasets/splits, hardware, fixtures, and independently executed evaluations. None is needed to finish our current CE/AURC study.

This completes the source-review TODO at an inspection-and-mathematical-check level; it does not complete model reproduction.
