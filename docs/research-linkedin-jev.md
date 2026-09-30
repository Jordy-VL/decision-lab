# Research note: Jev architecture and open baselines

Reviewed 2026-09-22; updated 2026-09-24. [Requested LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7508035616039727104). Source inspection only; no external model benchmark run.

## Additional Jev reproduction lead and `other` experiment

User-proposed [mity-prodgen/jev-test](https://github.com/mity-prodgen/jev-test) (shared via a LinkedIn redirect) is saved as a follow-up lead for Jev reproduction. Before adopting it, inspect the repository's implementation, benchmark/task sources, licenses, revisions, and whether it measures Jev itself or provides a reproduction harness. Record exact commit and dataset hashes; do not treat repository claims or results as independently verified until reproduced.

The user also proposed an experiment to tune for the `other` decision class. Treat this as a separate, exploratory follow-up arm, not a change to the primary CE-only / AURC-only / mixed comparison. First verify that the target dataset and task schema contain a meaningful `other` label, enough examples, and an auditable definition; `other` must not be manufactured by collapsing semantically different labels without justification. If viable, compare the existing matched baseline against an `other`-focused training variant using identical parent checkpoint, data split, seed, and update budget, with class-wise `other` precision/recall, calibration, risk-coverage and p50 end-to-end latency. Keep validation logits and labels. Select any operating threshold on development/calibration only and leave test untouched. Do not launch until the dataset audit and the primary local/GPU smoke prerequisites are complete.

## Evidence boundaries

The post presents community projects as reverse engineering Jev. The [underlying probe study](https://archerhume.com/posts/jevs-architecture-unmasked/) explicitly distinguishes observations from inference: direct probability readout and shared-state behavior are supported more strongly than the exact backbone/head. It does not uniquely recover a causal decoder, MoE or fixed-slot implementation. The 255-option API limit does not prove a 255-unit neural head.

[TypeSafe's launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev) states parallel probability outputs and RLCD, but supplies no reproducible training specification. Treat the LinkedIn assertion that RL is required for calibration as unsupported. [Guo et al.](https://arxiv.org/abs/1706.04599) establishes temperature scaling as an effective alternative in studied settings; neither approach guarantees reliability under arbitrary shift.

## Useful primary implementations

- [SemIf](https://github.com/TheoLeeCJ/SemIf): frozen-model direct option-logit inference with shared-prefix modes. Its [results](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/RESULTS.md) distinguish speed, task quality and distribution agreement; they do not establish Jev equivalence. This is a useful external baseline, with model size/runtime differences reported explicitly.
- [SemIf calibration](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/CALIBRATION.md): adds scalar temperature fitting and published out-of-fold ECE results. Reusable resources include source-fetch/build scripts, pinned selection manifests and saved option logits. We keep our single fixed calibration/test split, not their five-fold protocol.
- [jevlike model code](https://github.com/vinnylarouge/jevlike/blob/main/jevlike/model.py): each option independently attends to context before shared softmax. This offers a dynamic-menu architectural control. Our joint state/question/options encoder can model interactions among options; this code does not include that interaction before normalization.

## Measurement issue

The official [Choice confidence implementation](https://github.com/typesafe-ai/system-one-adapter-python/blob/fb52b1030b7fc1f4f1cf39910afa5da54f9835e3/src/system_one_adapter/_utils/confidence_metrics.py) rescales the maximum probability relative to uniform: `(pmax - 1/K)/(1 - 1/K)`. This is not p(correct). Keep raw probabilities and separately named derived scores. My inference: it preserves rankings for fixed K but can alter rankings when K varies.

## Implications for our register

Keep X1/X2 unchanged: full ModernBERT, bounded indices, matched CE/AURC. Add SemIf as a candidate external control for X6; use its workload assets only after inspecting provenance and leakage. A future X1 architecture ablation can compare a jointly encoded index head with an option-attention scorer under matched data and compute.

Differentiate on fixed-risk coverage, frozen-threshold OOD behavior and explicit deferral costs, not merely calibrated output or no text generation. Temperature scaling preserves each example's argmax but can change confidence ordering across multiclass examples, so recompute risk-coverage after calibration. Shared-prefix efficiency is a separate systems experiment: our current bidirectional joint encoder recomputes context for each question.

No new training, head change, mandatory permutation sweep or RL implementation follows from this post. Small order-sensitivity diagnostics can be considered later without changing official splits or retraining.

## Jev as an RL reward model (saved 2026-09-25)

- Source: Sophia Yang, [Getting started: your first RL run with Jev as the reward model](https://www.linkedin.com/pulse/getting-started-your-first-rl-run-jev-reward-model-sophia-yang-yrmpc/), published 2026-09-23.
- Linked implementation: [Fireworks + Jev RL tutorial](https://github.com/sophiamyang/fireworks-jev-reward-rl/blob/main/docs/TUTORIAL.md) (saved for follow-up; code not reviewed).
- The article describes a toy Qwen3.8 27B rank-8 LoRA run on Fireworks, with 96 training prompts, eight rollouts per prompt, 24 updates and 24 held-out evaluation prompts. Jev probabilities score style, quality and source support; reward combines the mean of style/quality with source support multiplicatively.
- Author-reported results: mean Jev reward increased from 0.583 to 0.759; 896 Jev calls cost $0.061 with 186 ms median latency. Outputs shortened by about 40%. These are reported demo results, not independently reproduced measurements.
- Relevance to DecisionLab: a downstream use case for probabilistic decisions as training feedback. It does not establish how Jev itself was trained or validate probability calibration. Using the same judge for training and evaluation, plus length effects, motivates independent evaluation and length-controlled comparisons before interpreting reward gains as quality gains.

## LlamaIndex document-task comparison (saved 2026-09-30)

- [Jev vs OSS repository](https://github.com/run-llama/jev_vs_oss) and its [comparison notebook](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs_open.ipynb), pinned at `0a3726fabea08cf9766fe04ec25815f0631009be`.
- [Detailed review and saved results](jev-vs-oss-document-tasks-20260930.md): five document-pipeline decisions, backend differences, confidence/latency limitations and implications for our external document suite. Jev leads saved classification/splitting results; Qwen and a simple heuristic outperform it on OCR triage. These are small publisher-reported comparisons, not independent reproduction.
- Reuse the decision-task framing while preserving our dataset exclusions and test-only policy. The notebook measures decisions over extracted text/OCR, not document-to-JSON extraction; it does not establish calibrated risk or uniformly measured end-to-end cost/latency. No benchmark run or primary experiment change follows from saving this reference.
