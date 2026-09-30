# Jev vs OSS: document-task comparison

Reviewed 2026-09-30. Source: LlamaIndex's [repository](https://github.com/run-llama/jev_vs_oss), pinned at `0a3726fabea08cf9766fe04ec25815f0631009be`. Read the [notebook and saved outputs](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs_open.ipynb), task/corpus builders, backend wrappers and evaluation code. These are publisher-reported results, not a benchmark rerun.

## What is being compared

The useful framing is **decisions around document processing**: language, orientation, document type, bundle boundaries and whether to escalate a weak local parse. LiteParse extracts native PDF text or uses Tesseract OCR; the decision models receive text or structured textual state, not page images. Tesseract OSD is a specialized image-based exception. This does not benchmark document-to-schema JSON extraction, tables, visual grounding or a complete PDF parser.

| Task | Saved n | Inputs and target |
| --- | ---: | --- |
| Language | 48 | Up to 1,200 characters of native text from Wikipedia-derived PDFs; 12 language options. |
| Orientation | 32 | Four cached OCR candidates, up to 500 characters each; select readable rotation. Four OCR passes happen before decision timing. |
| Classification | 96 | Six scanned pages per original RVL-CDIP class; English OCR, first 2,500 characters; 16 described class options. |
| Bundle splitting | 82 | Previous page tail and current page head, 700 characters each; new document versus continuation. Synthetic multipage Wikipedia articles concatenated into bundles; 27 true boundaries. |
| Parse triage | 48 | Complexity signals, OCR confidence, character count and excerpt; escalate when normalized OCR/source edit distance exceeds 0.10. |

Sources: [tasks](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs/tasks.py), [corpus construction](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs/corpus.py), [PDF processing](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs/pdf_tools.py).

Triage measures a proxy for OCR quality, not the measured benefit of a higher-tier parser. No expensive-parser output is evaluated. The builder knows the source language and true rotation: it chooses the Tesseract language and uses correctly rotated OCR. The task explicitly assumes orientation was fixed upstream. The reported character count is based on a stored candidate capped at 1,500 characters, with a 900-character excerpt. It is not a complete end-to-end routing experiment.

Fan-out helpers build four questions over shared state, but the pinned notebook has no executed fan-out comparison. All fan-out document-type truths are encyclopedia_article, and its state includes the correctly rotated triage excerpt. Do not interpret this as a varied document-classification or independent orientation benchmark.

## Saved accuracy comparisons

Percentages transcribed from the notebook's final combined table; rounded to one decimal. Small samples, no reported confidence intervals or repeated runs.

| Task | Jev | Qwen3.5-4B | Laya | jeff | Specialized baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| Language | 100.0% | 100.0% | 39.6% | 47.9% | Lingua: 100.0% |
| Orientation | 93.8% | 90.6% | 28.1% | 21.9% | Tesseract OSD: 100.0% |
| Classification | 54.2% | 51.0% | 24.0% | 26.0% | — |
| Bundle splitting | 96.3% | 87.8% | 70.7% | 73.2% | — |
| Parse triage | 85.4% | 95.8% | 52.1% | 64.6% | OCR heuristic: 93.8% |

For new-document boundaries, Jev precision/recall/F1 is **90.0% / 100.0% / 94.7%**; Qwen is **94.7% / 66.7% / 78.3%**. Overall accuracy alone hides missed boundaries.

The classification confidence sweep is especially relevant to Decision Lab:
- Jev at reported confidence >=0.9 accepts 44/96 pages (45.8% coverage), with 36/44 correct (81.8% accuracy).
- Qwen at >=0.9 accepts 36/96 pages (37.5%), with 31/36 correct (86.1%).
- Laya accepts only four pages at that threshold; three are correct.
- These are retrospective subsets of the same sample, not independently calibrated operating thresholds or risk guarantees. In particular, a score of 0.9 does not establish 90% correctness.

## Backend and measurement differences

Source: [backends](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs/baselines.py), [remote decoder](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/modal_app.py), [evaluation](https://github.com/run-llama/jev_vs_oss/blob/0a3726fabea08cf9766fe04ec25815f0631009be/jev_vs/evalh.py), and notebook setup/output.

- **Jev:** hosted service, concurrent requests. The illustrative raw response says jev-1.13.0, while benchmark setup selects jev-latest; do not infer a pinned benchmark model version from that example.
- **Qwen3.5-4B:** Modal L4; a single forward pass reads option-letter logits, then softmaxes over allowed letters. It is not autoregressive JSON generation. The implementation supports at most 26 single-token option letters.
- **Laya:** typed-decisions checkpoint, local MPS in the saved run, max_len=2048 and head_max_len=512. The saved warning says an out-of-range choice:11+ temperature of 0.1006 is clamped and affected confidence must be treated as uncalibrated. This particularly matters for the 12- and 16-option comparisons; it does not by itself explain all accuracy failures. No pinned model revision is recorded.
- **jeff:** self-hosted Jev-compatible service backed by gliformer. The saved endpoint runs on Modal; the benchmark explicitly charges zero API token cost. Zero reported cost is not zero compute cost.
- **Specialized baselines:** Lingua for language, Tesseract OSD for orientation, and an OCR-confidence/character-count heuristic for triage. Their strong results argue for keeping simple baselines.

Confidence columns are not commensurate: Jev and Laya expose their native scores; Qwen uses top-minus-runner-up probability; other baselines use their own scores. Mean confidence and the confidence sweep are not calibration validation. No ECE/NLL/Brier evaluation establishes the notebook's broader calibration claims.

Latency is not measured uniformly. Jev is timed per client request inside the concurrency semaphore; Qwen uses server method elapsed time divided by batch size, excluding remote transport/queue/cold-start; Laya is local per-call. OCR preprocessing is excluded, notably the four orientation passes. The Qwen wrapper's latency prose is broader than the actual field it uses. Do not quote the displayed medians as a controlled end-to-end speed comparison.

Costs similarly mix API token accounting, warm GPU method-time estimates and zero-cost local/self-hosted entries. Idle time, CPU, preprocessing and cold starts are not consistently included. Treat displayed costs as limited attributed estimates, not current total operating prices.

## Trust and reuse assessment

Useful, inspectable primary implementation under the LlamaIndex organization, with MIT-licensed code and saved outputs. It is good evidence of the authors' experimental setup and a useful source of task designs. It is not independent validation of Jev or a conclusive ranking of candidate-head architectures.

The notebook is a demonstration on small samples. Dataset/model revisions and a frozen corpus/prediction bundle are not fully supplied; code defaults also differ from some saved counts. A reproduction must pin code, models, corpus manifests, samples and preprocessing, and retain per-example predictions.

The classification corpus is **original RVL-CDIP**, which our plan excludes. Reuse the task idea, not that dataset by default. Our separate RVL-CDIP-N_MultiPage plan remains test-only, with no tuning/calibration on its test set.

## Implications for Decision Lab

1. Add a separate document-decision evaluation suite after the primary matched CE/AURC/mixed experiment. Prioritize document classification, boundary splitting and parse escalation. Do not silently change the primary benchmark.
2. Differentiate through **coverage at a frozen risk/cost policy**, transfer under OCR noise/source/template shift and explicit escalation costs, not a generic claim that output probabilities are calibrated.
3. Fit calibrators and thresholds on separate document-grouped calibration data. Report accuracy, NLL/Brier, AURC/AUGRC and frozen-threshold test risk/coverage. Keep whole bundles/source articles together to prevent leakage.
4. For parse escalation, measure actual downstream extraction gains and total cost from the expensive parser, alongside the OCR-error proxy.
5. Hold preprocessing constant across decision backends, include specialized baselines, and report both model-only and complete pipeline latency/cost under matched hardware/timer scope.
6. Treat shared-context fan-out as a separate systems experiment. Our current bidirectional joint encoder re-encodes per question; this notebook does not establish that we already match Jev/Laya's shared-state behavior.
7. Keep Laya as a distinct architecture/configuration comparison. Its poor demo scores are not proof that all candidate-aware heads fail.

Related: [experiment register](experiment-register.md), [Jev research references](research-linkedin-jev.md), [candidate-head audit](candidate-head-audit-20260930.md).

No external benchmark was executed, no paid API calls were made, and no dataset was added. The inspected source snapshot is retained under ignored runs/reference-jev-vs-oss-20260930/.
