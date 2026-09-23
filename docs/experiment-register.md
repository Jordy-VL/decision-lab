# Experiment register

Updated 2026-09-23. This is the review entry point for research decisions. No hypothesis below has been confirmed. Package implementation/smoke checks are not model-quality evidence.

## Shared constraints

- Configuration direction is recorded in [config and loader design](config-design.md): baseline adapter with predefined splits first; data/model/training sections in one experiment config; optional weighted synthetic sources later. Hugging Face model-config subclassing is a packaging direction to verify, not a required refactor before the pilot.

- Updated baseline priority: first reuse a pinned, existing competitor training suite; our custom mixture and synthetic additions come later. Candidate anchor is Kev decision-v7, subject to exact-file/hash and adapter validation. Match examples, labels, option semantics and official partitions, not merely dataset names. Compare against a matching pre-delta Kev checkpoint (v7-base) unless its later generated/replay training is also included. Kev's training-time augmentations must be recorded separately; equal source files alone do not establish identical training exposure. Sources: [Kev training](https://github.com/jaredpalmer/kev#training), [Kev-0.8B recipe](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-0.8b.md). This overrides the initially proposed independently sampled 10k–20k mixture for X1; that mixture belongs in later X3 work.

- Keep V1 as the frozen reference: ModernBERT-large, full fine-tuning, CLS pooling, a fixed 128-slot index head and masked unused outputs. V2 is a planned option-aware candidate scorer; it must not overwrite or relabel V1 results.
- Kev decision-v7 remains the anchor for the first V1/V2 architecture comparison. Add external datasets later as separately named suites with their own provenance, splits, modality and per-suite metrics; do not pool unlike tasks into one headline score.
- Official splits when available; otherwise one fixed document-grouped split. One seed for the pilot; no cross-validation, mandatory permutation sweep, or large hyperparameter search.
- Keep sources/documents and all derived variants in one partition. Reserve development/calibration data without touching final tests.
- X2 objective matrix: CE-only (lambda 0), AURC-weighted CE surrogate only (lambda 1), and blend (lambda 0.5), all initialized from the same frozen CE checkpoint. Follow-up repetitions only after a promising, interpretable signal.
- Every run must retain raw validation/development and calibration logits with labels and example/group IDs, plus probabilities and the resolved configuration. Fit calibration/thresholds on calibration only; validation artifacts support checkpoint diagnostics and paired analysis.
- Training infrastructure is an open choice: compare the existing Modal path with the configured `ssh uab-gpu` cluster using access, scheduler, GPU availability, durable storage, interruption/resume, setup effort, wall time and total cost. See [training infrastructure plan](training-infrastructure-plan.md); no backend is selected by assumption.
- Generator and trainer are separate uv workspace packages. No paid generation before endpoint/model/rates/pilot budget are configured. No assumption that Nebius Cloud credit applies to every inference offering.
- Borrowed code must retain required attribution. Dataset access, provenance, and terms must be recorded. Potential novelty is a research hypothesis, not a literature-complete claim.

## Experiment map

| ID | Question | Dependency | Current state |
|---|---|---|---|
| X0 | Can we obtain and transform the data faithfully? | None | Source/code inspection done; full audit pending |
| X1 | Does the minimal index model learn all three primitives? | X0 | Implemented; tiny CPU smoke passed |
| X2 | Does AURC training improve useful selective operating points? | X1 | Planned; primary research experiment |
| X3 | Does diverse synthetic data help beyond existing labeled tasks? | X0–X1 | Implemented; offline and small source checks passed |
| X4 | Do selective/calibration benefits transfer out of domain? | X2 | Planned fixed holdouts |
| X5 | Can document ground truth yield useful typed decisions? | X0–X2 | DUDE/RVL-CDIP-N_MultiPage inspected; OCR/vision path pending |
| X6 | How close are we to Jev on a common external protocol? | X1–X2 | Harness found; frozen data access unverified |
| X7 | Does vision add enough value to justify complexity? | X5 | Deferred |
| X8 | Do per-step gains improve multi-turn behavior? | X2/X4 | Deferred |
| X9 | Does RL improve abstention beyond supervised objectives? | X2/X4 | Deferred |
| X10 | Do we need a fourth numeric primitive? | Concrete numeric use case | Assessment only |
| X12 | Does an option-aware head improve flexibility over fixed output slots? | X1 | V2 design recorded; V1 remains frozen reference |

## X0 — data and measurement feasibility

**Hypothesis:** existing loaders and a simple canonical schema support a reproducible low-cost pilot.

**Borrow/source:** [Kev data converters](https://github.com/jaredpalmer/kev/blob/main/kev/data.py), [split builder](https://github.com/jaredpalmer/kev/blob/main/kev/suite.py), [v7 manifest](https://github.com/jaredpalmer/kev/blob/main/evals/v7/decision-v7/manifest.json), [AURC estimators](https://github.com/han678/AsymptoticAURC/blob/main/utils/estimators.py).

**Setup:** pin source/code revisions; validate class-to-index mappings, provenance, grouping, licence/access and real token lengths. Hand-check AURC/tie conventions. Public DUDE loader exposes multiple OCR engines; verify actual assets and legacy loader compatibility. RVL-CDIP-N_MultiPage has 991 labeled PDF records, 16 classes and a test-only split; keep it evaluation-only. Audit OCR or multimodal feasibility, document/page grouping and its CC BY-NC 4.0 terms before use. Do not invent train or calibration partitions from its test set.

**Minimum:** small real-data sample for each selected source, one full schema pass, token-count summary; no model training.

**Our contribution:** an auditable common decision representation and measurement protocol; loaders themselves are borrowed infrastructure.

**If supported:** freeze pilot manifests. **If not:** repair/drop the unavailable source and disclose coverage; do not fabricate or silently substitute data.

## X1 — minimal three-primitive index model

**Hypothesis:** one small encoder with a fixed-capacity index head can learn choice, boolean and ordinal decisions under supplied instructions/options.

**Borrow/source:** [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large); [Dev-0.4B card](https://huggingface.co/mpnikhil/dev-0.4b) as architectural inspiration, not verified reusable implementation; [Kev](https://github.com/jaredpalmer/kev) for data/API concepts.

**Setup:** ordinary numbered options, pooled encoder representation, max_options head, slot mask, complete probability output. Proposed capacity 128; CLINC150+OOS needs >=151 if included. YAML config/CLI; save resolved settings, revisions and checkpoint. Initial 512–1,024 token budget; report unsupported inputs explicitly.

**Minimum:** ~10k–20k balanced labeled examples across Banking77, BoolQ, MNLI and accepted ordinal data SST-5/modest Yelp; one CE run. Exact mix follows X0. Verify all primitives over chance and inspect per-source errors.

**Our contribution:** this simple architecture is an experimental vehicle, not the main novelty claim.

**If supported:** freeze warm-up checkpoint for X2. **If not:** debug schema, masking, evidence and optimization before adding data or architecture complexity.

## X2 — selective utility versus calibration (primary)

**Hypothesis:** AURC-weighted continuation yields higher coverage at the same low error rate than matched CE, including calibrated CE, without unacceptable accuracy/probability degradation. Separate whether rank weighting works alone from whether blending it with CE is more robust.

**Borrow/source:** [AsymptoticAURC paper](https://arxiv.org/abs/2410.15361), [weighted loss](https://github.com/han678/AsymptoticAURC/blob/main/utils/loss.py), [temperature scaling](https://arxiv.org/abs/1706.04599).

**Setup:** identical warm-up, data and continuation compute. Compare three arms: CE-only (lambda 0), AURC-weighted CE only (lambda 1), and the mixed objective (lambda 0.5). “AURC-only” means only the detached harmonic rank-weighted CE surrogate, not direct optimization of discrete AURC. Actual ranking batch is distinct from optimizer accumulation. Prespecify grouping, low-risk budgets, accuracy tolerance, and tie handling. Fit each temperature on separate calibration data.

**Minimum:** three matched continuations, one seed/split, raw and calibrated readouts from each. Save every prediction and resolved config. Plot risk–coverage, reliability/sample counts, and risk/coverage versus threshold. Select thresholds before looking at final test outcomes. Bootstrap original groups from saved predictions if feasible; this requires no retraining.

**Our contribution:** testing selective-risk optimization in a small instruction-conditioned typed decision model, beyond reporting ECE alone. Novelty requires further literature verification.

**If supported:** confirm the promising effect with limited additional seeds and X4. **If not:** report the negative result and distinguish weak accuracy, loss implementation, ranking-batch effects and lack of transfer. Do not retune on test or claim every low-risk region improved because average AURC fell.

## X3 — grounded synthetic data

**Hypothesis:** a diverse, evidence-grounded mixture improves unfamiliar decisions more than a size/compute-matched existing-data baseline.

**Borrow/source:** [Kev generation tooling](https://github.com/jaredpalmer/kev/tree/main/skills/kev-finetune/scripts), [Qwen3.8-27B candidate teacher](https://huggingface.co/Qwen/Qwen3.8-27B), [vLLM structured output](https://docs.vllm.ai/en/latest/features/structured_outputs/). Individual HF corpus IDs/revisions go in the generator registry; financial/legal/general/everyday domain coverage is requested.

**Setup:** deterministic bounded source sampling; source grouping before generation; strict facts/evidence/question/options/rubric records; choice, boolean and ordinal quotas. Programmatic/annotation validation where available; teacher-only labels explicitly distinguished. Independent answering is a consistency check, not proof. Credentials local, bounded concurrency, resumable outputs and conservative cost limits.

**Minimum:** a small audited pilot (e.g. 100 source records with capped questions), then a capped training addition if its quality is acceptable. Report rejection rate, actual token/cost usage and domain balance. Compare matched update counts and document sample sizes.

**Our contribution:** task/rubric breadth and grounded uncertainty cases; not simply generating many teacher answers.

**If supported:** scale the useful domains gradually within an agreed budget. **If not:** repair generation/verification or retain only validated subtypes. Never scale a low-quality pilot merely to increase volume.

## X4 — out-of-domain robustness

**Hypothesis:** gains in X2 survive dataset/domain/rule-family shifts, with useful threshold transfer.

**Borrow/source:** [Kev transfer/data policies](https://github.com/jaredpalmer/kev/blob/main/kev/data.py); user research context on domain shift. No claim about TypeSafe's undisclosed training mixture.

**Setup:** choose entire held-out sources/families in advance, keep fixed splits, and distinguish unseen fine-tuning domains from unknown pretraining contamination. Test ID-calibrated thresholds unchanged on OOD data. Recalibration using labeled OOD data is a separately named adaptation experiment.

**Minimum:** reuse X2 checkpoints on one or two fixed held-out sources; no retraining initially. Report ID/OOD curves, accuracy, probability metrics, and actual risk at frozen thresholds.

**Our contribution:** evidence about transfer of selective behavior, not just mixture-wide calibration.

**If supported:** confirm on another shift. **If not:** scope claims to ID, identify failure modes and motivate X9 or targeted mixture changes. Neither AURC nor RL guarantees OOD calibration.

## X5 — document-derived decisions

**Hypothesis:** original document annotations can support useful grounded choice/boolean tasks, with ordinal tasks only where an explicit rubric has valid labels.

**Borrow/source:** [user DUDE loader](https://huggingface.co/datasets/jordyvl/DUDE_loader), [RVL-CDIP-N_MultiPage](https://huggingface.co/datasets/jordyvl/rvl_cdip_n_mp), a 16-class multi-page PDF suite. Original RVL-CDIP is excluded by user preference.

**Setup:** small fixed document subset, official splits or explicitly custom fixed split if needed. Real ModernBERT token audit; no gold-answer-guided retrieval. Same-document distractors, answer-variant checks, explicit multi-answer/unanswerable handling. Preserve OCR/image pairing for later. A mismatch with the gold answer alone does not prove a statement is false.

**Minimum:** bounded audited DUDE-derived multiple-choice subset and a small grounded boolean subset. Keep tasks clearly named as adaptations. RVL-CDIP-N_MultiPage is evaluation-only: its published loader exposes a test split only. Use OCR with a named pipeline for text-only scoring, or defer direct PDF evaluation until multimodal support. Do not derive ordinal labels from nominal class IDs or tune against this test split.

**Our contribution:** verified document decisions and OCR/evidence-availability analysis, not a claim to reproduce original generative DUDE scores.

**If supported:** expand within fixed budgets and test evidence loss. **If not:** fix candidate validity or retrieval; preserve the text baseline rather than immediately adding vision.

## X6 — external Jev comparison

**Hypothesis:** our model achieves useful breadth under a fixed independent decision protocol, with explicit capacity limitations.

**Borrow/source:** [Decision Index kit](https://github.com/apolinario/decision-index), its source adapters, frozen corpus/hashes and native scorers. This is community-run, not a TypeSafe official benchmark.

**Setup:** verify corpus access or pinned rebuild; implement thin adapter mapping internal indices to supplied keys. No truncation/pruning; retain all probabilities. Preserve linked cases, failed-request accounting and actual timing scope.

**Minimum:** a declared small subset to validate adapter/scoring, then full sweep only when affordable. Subset scores are not full leaderboard scores. Frozen data remain evaluation-only.

**Our contribution:** a new model evaluated by an existing protocol; AURC analysis remains separate because index scoring counts abstention as wrong.

**If supported:** full reproducible report/submission can be considered. **If not:** report per-task/capacity gaps; avoid tuning prompts on the final benchmark.

## X7 — vision, deferred

**Hypothesis:** image evidence resolves meaningful OCR-only errors enough to justify extra training/inference cost.

**Borrow/source:** [SigLIP 2 Base](https://huggingface.co/google/siglip2-base-patch16-224) candidate vision tower; alternatively investigate a native multimodal/Gemma-family baseline later. Model/version suitability and total parameter counts must be verified then.

**Setup:** retain current bounded index head; fuse text representations with patch features through a small learned block. This supersedes earlier per-option-marker fusion sketches. Frozen/cached vision first, paired data and same splits. OCR for small text; explicit image resolution/page-selection costs.

**Minimum:** matched OCR-only versus OCR+image on a fixed visually relevant subset, after X5 failure analysis. No vision implementation in milestone one.

**Our contribution:** evidence that visual information improves selective decisions under a constrained model budget.

**If supported:** repeat CE/AURC comparison with vision. **If not:** retain OCR-only and document compute/quality tradeoffs.

## X8 — multi-turn decisions, deferred

**Hypothesis:** useful per-step selective behavior improves end-to-end outcomes once errors compound over a trajectory.

**Borrow/source:** [Decision Index API-Bank/tool adaptations](https://github.com/apolinario/decision-index), [TypeSafe workflow evaluations](https://evals.typesafe.ai/), [tau benchmark family](https://github.com/sierra-research/tau-bench). Check current maintained tau version before use.

**Setup:** history prefixes only; no future outcome leakage. Whole conversations grouped into splits. Candidate action generation, tool arguments, dialogue generation and execution are explicitly outside the index model unless supplied by the harness.

**Minimum:** fixed offline trajectory-prefix evaluation, then a few bounded interactive episodes. Measure episode success, costly errors and deferrals, not only step AURC.

**Our contribution:** linking selective decisions to system-level utility with a fully specified surrounding system.

**If supported:** broader rollout study. **If not:** inspect action candidates, state representation and compounding/handoff costs; do not infer agent reliability from static classification.

## X9 — RL abstention, deferred

**Hypothesis:** reward-based fine-tuning improves answer/defer utility under shift beyond supervised CE/AURC and supervised abstention baselines.

**Borrow/source:** TypeSafe's RLCD is motivation, not a reproducible disclosed method. A concrete open RL implementation and relevant literature must be selected in a later design review; no RL package is currently committed.

**Setup:** explicit answer error penalty, deferral cost, reward validity and coverage constraints; matched data/compute; fixed OOD tasks. Avoid always-abstain policies and reward shortcuts. Action utility and honest probability elicitation are different objectives.

**Minimum:** a small contextual answer/defer task on an established checkpoint, compared with thresholding and a supervised abstention baseline.

**Our contribution:** test whether RL adds anything beyond objective/data changes; do not assume it is inherently less overfitting-prone.

**If supported:** investigate robustness and probability quality. **If not:** retain supervised route and publish scope/negative evidence if useful.

## X10 — numeric output, assessment only

**Hypothesis:** a new numeric primitive adds value that candidate selection and ordered Score cannot provide.

**Borrow/source:** existing Jev Choice/Score concepts; detailed reasoning in [numeric outputs](numeric-outputs.md). No new implementation borrowed yet.

**Setup/minimum:** identify a real use case. First test numerical candidate selection with units/normalization and candidate recall; keep arithmetic in code. Continuous regression needs its own labels, loss and uncertainty evaluation and may require a different output head.

**Our contribution:** only meaningful if a genuinely unmet decision task emerges.

**If supported:** design a separate regression/distribution experiment. **If not:** keep the original three primitives.

## X11 — multilingual decision distillation, deferred

**Hypothesis:** a balanced multilingual decision mixture improves selective prediction across languages, including transfer of calibration-selected thresholds. AURC gains across languages are an empirical question, not a guarantee.

**Borrow/source:** user-proposed [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), approximately 204K human-annotated instruction/response pairs in 65 languages (71 including dialects/scripts), Apache-2.0. Distinguish it from the much larger [Aya Collection](https://huggingface.co/datasets/CohereLabs/aya_collection), whose component provenance and terms need checking. These are instruction datasets, not automatically grounded evidence corpora. Pin a revision and verify counts at preparation time; the Aya card's prose and metadata have a small count discrepancy.

### Typed Decisions (LocalLLaMA)

**Status:** source to investigate; not approved for training or evaluation yet.

**Borrow/source:** user-proposed [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions). The Hub card currently describes a synthetic English text-classification dataset in Parquet with an Apache-2.0 license, five subsets, and 400 rows in `agent_trace_observability`. Its structured examples expose several candidate targets (`outcome`, `action`, `needs_review`, `risk`, and `urgency`) with label/confidence/probability fields. This looks relevant to structured decisions and uncertainty, but the small synthetic subset should be treated as a pilot or robustness source rather than evidence of broad real-world transfer.

**Follow-up:** inspect the complete card, all subset/split counts and schemas, generation and annotation process, label definitions, provenance, and any overlap with our task construction. Pin a dataset revision and local file hashes before preparing anything. Keep a subset/task-family holdout for evaluation; do not train and evaluate on the same generated examples. Confirm the license and any upstream component terms before redistribution or use.

**Setup:** use original language, inputs, targets and annotation provenance. Select objectively checkable tasks with sufficient evidence; preserve labels where possible. A multilingual teacher can propose same-language distractors and Boolean variants; add ordinals only with a defensible rubric. Existing target reuse is supervised conversion; teacher-generated labels or distributions are distillation. Do not treat a human-written open-ended response as automatically unique or factual, or place the target into the evidence as an answer shortcut. Validate semantic alternatives and label validity, with a small human-reviewed sample per language.

**Minimum:** after the English baseline, freeze a small three-language subset and a multilingual-capable sub-1B backbone. Reuse the same indexed output contract. Run matched CE versus CE+AURC with identical data and updates. Preserve official train/test membership; derive one fixed development/calibration split from train, grouping source items and translations together. The published Aya test covers only seven languages, so any additional language holdouts must be identified as our own fixed splits. Choose pilot languages after auditing usable task counts.

**Evaluation:** per-language accuracy, calibration and risk–coverage, language-macro summaries, and fixed-threshold transfer across languages; distinguish seen-language from held-out-language results. Check teacher errors, tokenizer lengths and English retention. Original ModernBERT-large should not be assumed to acquire strong multilingual representations from a small distillation run; backbone selection is a separate controlled decision.

**Our contribution:** test whether selective utility transfers across languages rather than merely improving pooled ECE. **If supported:** expand language/task coverage gradually. **If not:** inspect teacher/conversion quality and representation limits before increasing data; retain a narrower supported language scope. No download, generation or baseline-mixture change is authorized by this design note.

## X12 — option-aware head (Baseline V2)

**Hypothesis:** scoring each candidate from its own contextual representation reduces dependence on fixed output-slot identities and transfers better to new option sets, without an unacceptable quality or compute cost.

**Borrow/source:** Laya's published architecture uses an option-marker scorer at candidate-specific `[MASK]` positions and a two-layer decision head. The [decision-head comparison visual](../paper/decision-heads.svg) is a design sketch, not a claim of exact replication.

**Setup:** keep the V1 ModernBERT-large encoder, Kev decision-v7 rows/splits, seed, CE objective, optimizer budget and evaluation code fixed. Change only candidate rendering and decision head: add one candidate marker per option and score each marker with a shared candidate scorer. Record added parameters, sequence length and latency. Train the V2 CE baseline from the same pinned pretrained encoder, then compare matched AURC continuations only after the CE architecture comparison is sound. Measure accuracy, NLL, Brier, AURC, calibration, option-count slices and a controlled option-permutation diagnostic. V2 must retain the request-time option/probability interface and must not introduce fixed semantic output classes.

**Efficiency metric:** report **p50 end-to-end latency (↓)** for both V1 and V2 on the same fixed requests, hardware, batch size/concurrency and warm-up policy. The timer covers raw request receipt through preprocessing/tokenization, model inference and construction of the complete option-probability response. Include OCR/retrieval only if it is part of the declared serving path; report cache state, and never compare model-only timing to end-to-end timing. Include p95 and throughput where practical.

**Minimum:** one seed and matched compute on Kev, with low- and high-option-count slices. Preserve the V1 checkpoint, configuration, predictions and result tables unchanged.

**Broader evaluation:** after the architecture comparison, add datasets as separate evaluation suites. RVL-CDIP-N_MultiPage is a test-only, 16-class, multi-page PDF suite with 991 labeled records. Its current CC BY-NC 4.0 terms and PDF/image modality need to be respected. OCR permits a text-only evaluation with an explicit OCR pipeline; direct page/PDF input requires multimodal support. Never derive tuning or calibration data from its test split.

**If supported:** retain V2 as the next baseline and extend to frozen external suites. **If not:** inspect candidate marker rendering, option count and context cost before deciding whether to retain V1's simpler head.

## Run ledger requirements

Related research update: [LinkedIn Jev architecture and open baselines review](research-linkedin-jev.md). SemIf is a candidate external X6 control; jevlike is a possible later X1 architecture ablation. Neither changes the first CE/AURC pilot.

Additional deferred X6 comparison: [Decision 1.0 review and minimal comparison plan](decision-1-comparison.md). Eos-0.8B is the preferred sub-1B candidate; Kai/Lex are architecture references. Publisher aggregate accuracy and transfer results differ; evaluate selective utility separately. No extra run or baseline change is scheduled.

Every actual run should record experiment ID, parent checkpoint, resolved YAML config, model/code/data revisions, split/group manifest, seed, loss/ranking-batch settings, hardware, time/cost, saved prediction path, metric definitions, and a short conclusion: supported, unsupported, or inconclusive. Record the next action and why. Never replace planned criteria after inspecting final test results without marking the new analysis exploratory.
