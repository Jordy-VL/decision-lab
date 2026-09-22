# Research note: Jev architecture and open baselines

Reviewed 2026-09-22. [Requested LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7508035616039727104). Source inspection only; no external model benchmark run.

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
