# Additional experiment tracks — 2 October 2026

Status: research plans, not completed experiments. Keep existing v7 head comparisons and running jobs unchanged.

## Ecosystem reference

- [Han Xiao — All About Jev, filtered for decision-lab](https://hanxiao.io/all-about-jev/#v=table&cat=model&q=decision-lab), supplied 2 October 2026. Preserve the fragment: it requests table view, model category and the `decision-lab` query. Saved as an ecosystem/discovery reference. Page content and any Decision Lab listing were not verified: web retrieval exceeded its size limit and direct retrieval returned HTTP 403. Follow linked primary sources before relying on comparison claims.

## Multilabel and hierarchical selective tagging

Read the [user's LinkedIn post](https://lnkd.in/p/eqkTPeXv), published 1 October, linking [Jordy-VL/AAPD](https://github.com/Jordy-VL/AAPD). It describes arXiv tagging with approximately 150 categories and 1.7 labels per paper, existing encoder/domain/few-shot/generative baselines, and BCE, asymmetric and two-way multilabel losses. These are post-reported properties; verify the repository's actual dataset variant, taxonomy and splits before adopting those counts as experiment metadata.

Proposed controlled experiment:
- Start with independent sigmoid label scores and BCE; compare asymmetric and two-way multilabel losses under matched encoder, split, budget and seeds. A categorical softmax cannot represent multiple simultaneous labels.
- Compare a fixed label head with a shared label-text scorer; permute labels with targets remapped to test order dependence. Exclude padded labels from losses and metrics.
- Report micro/macro F1, per-label PR-AUC, rare/common-label slices, Brier score and reliability. Tune thresholds only on calibration data; compare global thresholds with per-label thresholds and shrinkage for sparse labels.
- Define three separate confidence targets: individual tag correctness, precision of the proposed set, and completeness/exact correctness of the full set. Report missed labels as well as false positives.
- Define the abstention unit before computing risk–coverage/AUGRC: label decision, full document or hierarchy node. Report both automated coverage and human-review workload; single-label AURC is not automatically a valid multilabel metric.
- Test broad-topic acceptance with narrow-topic abstention using a pinned hierarchy and a stated ancestor-consistency rule. Hold out papers/groups across splits and check temporal duplicates.

Next: audit the linked code/report, establish a split/taxonomy manifest and reproduce the BCE baseline before adding confidence objectives.

## Document decisions with ModernVBERT

The supplied [Drive folder](https://drive.google.com/drive/folders/1VDnwRhmguvhKUCZ0_nv54RMGgqfYHGfz) was inspected. Its README identifies **RVL-CDIP-N** as new-source examples within the original 16 classes and **RVL-CDIP-O** as examples outside those classes. These are companion evaluation sets, not interchangeable training splits. Other folders include training, evaluation, output and PDFs. The README states CC BY 3.0; inspect actual files and split membership before ingestion.

Use [ModernVBERT/modernvbert](https://huggingface.co/ModernVBERT/modernvbert) as the requested base. The card describes a multimodal encoder after alignment/MLM, distinct from its retrieval-specialized variants. It needs an image-aware processor and a new decision-head adapter. Pin weights, processor and library versions before training; verify how image-token expansion affects question/option spans.

The [jev_vs_oss comparison](https://github.com/run-llama/jev_vs_oss), reviewed at `0a3726fabea08cf9766fe04ec25815f0631009be`, includes language identification, rotation, RVL-CDIP classification, page-boundary splitting and parse-tier triage. It primarily supplies OCR text/signals to decision models. This is not evidence of general field-extraction quality.

Proposed arms: OCR-only text encoder; page-image ModernVBERT; image+OCR ModernVBERT, with matched labels/splits and separately measured OCR costs. First reproduce classification, then evaluate RVL-CDIP-N distribution shift and RVL-CDIP-O rejection. Never train on those held-out sets. Select thresholds on independent calibration data, then measure accuracy, AUGRC, risk–coverage, OOD AUROC/AUPR and latency. Add page-boundary and triage tasks separately; extraction requires its own schema/field correctness and abstention protocol.

Before launch: reconcile the existing project's original-RVL-CDIP restrictions, audit the available training split, pin image/OCR hashes and preprocessing, and check overlap with model pretraining data. No document training launched.

## KEV suite versions and provenance

[jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) inspected at repository revision **cc4bac803e73112689ec327ffa481c519cbc7a05**, last modified 30 September 2026. There is no README at this revision. Pin both repository revision and suite subdirectory; manifest `version` is a schema/protocol field, not the suite name.

| Suite | Train records / questions | Calibration records / questions | Role |
|---|---:|---:|---|
| v7/decision-v7 | 12,576 / 15,576 | 968 / 1,148 | Current comparison's suite family |
| v8/decision-v8 | 12,688 / 15,688 | 984 / 1,164 | Adds shipping-delay legacy family; changed training/calibration |
| v9/transfer-v9 | 0 / 0 | 0 / 0 | Explicit evaluation-only transfer suite, not v9 training data |

v7 and v8 manifests give identical development (1,204 records / 1,468 questions) and test (1,176 / 1,440) hashes. v9 has 1,264 development and 1,264 test records, including buried evidence, unknowable/control and MMLU-Pro slices. Do not select thresholds on its test set.

Upstream training SHA-256:
- v7: `7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad`
- v8: `0a2199e4a63384a50ae9185a4ff3aa8fd87784cd8e1d74adf846e20309943b2b`

The current local config identifies adapted data as `kev-decision-v7-a88f56db-adapter-v1`; do not relabel it as the newly inspected Hub revision. Future runs must record upstream revision/subdirectory, manifest and raw split hashes, adapter revision, adapted split hashes, source filters, option permutations and seed. Manifest source-selection rules mention public sources plus one synthetic arm; record actual filtering rather than assuming all published rows were used. Upstream checkpoint selection mentions NLL; Decision Lab retains its own development-AUGRC protocol.

Other available families include decision/transfer v1–v6, round3, public-pool v4–v6, smoke, documents-v1, hard-v1, diagnostics/binding-v1 and external scienthoon/semif. Names alone do not establish suitability or independence. Add a separate v7→v8 data ablation after the matched head comparison, rather than silently upgrading current runs.
