# Experiment plan: AURC-trained small decision models

Updated: 2026-09-22. Proposed protocol, not experimental results.

## Hypotheses

H1: AURC-weighted continuation improves held-out selective prediction compared with a compute-matched CE continuation from the same supervised checkpoint.

H2: Some improvement transfers to unseen datasets, policy families, and document/OCR conditions.

H3 (later): visual evidence improves document decisions over OCR-only inputs on the same examples, and the selective-prediction benefits persist.

Calibration is evaluated independently. Improving confidence ordering does not establish that the probabilities are numerically calibrated. Lower AURC can also result from better accuracy, so report both and examine matched-coverage curves.

## E0 — establish data and measurement

Canonical example:

```json
{
  "id": "source/split/example/question",
  "group_id": "source/document-or-original-example",
  "source": "dataset_name",
  "split": "train",
  "state": "OCR text or other evidence",
  "question": "Which option answers the question?",
  "options": ["option text A", "option text B"],
  "target_index": 0,
  "type": "choice",
  "metadata": {"source_revision": "pinned revision", "ocr_engine": null}
}
```

Allow a set of acceptable indices or soft targets where annotations warrant them. Do not force ambiguous multi-answer records into a false single-label target. Store numeric rubric values separately for ordinal readout. Preserve provenance and original labels.

Follow official splits wherever available; otherwise create one seeded fixed split. Do not run cross-validation or repeated random splits. Reserve development and calibration subsets once from available official training/validation data; leave official tests untouched. Keep entire datasets or task families held out for transfer tests. Original document IDs govern splitting: all questions, OCR variants, corruptions, and synthesized options from one document stay together. Deduplicate across sources where feasible; record remaining contamination limitations. Preserve official test sets.

Pin code/model/dataset revisions, seeds, sampling rules, source licences, and resulting file hashes. Check legacy loader compatibility before relying on old `load_dataset` examples. No benchmark/test answer leakage into training or retrieval.

Audit actual token lengths using the ModernBERT tokenizer. Count state + question + all options + special markers. Report per-source percentiles and fractions within 512, 1,024, 2,048, 4,096, and 8,192 tokens. Record truncation/unsupported rates explicitly. Do not infer fit from page counts. For the first short-context pilot, label it as a restricted subset rather than full-document performance.

Validate AURC against tiny hand-computed examples: all correct, all incorrect, perfect and reversed confidence rankings, ties, variable batch sizes, and monotonic confidence transformations. Inspect the reference `get_AURC` denominator after removal; it appeared inconsistent during code review. Define tie handling explicitly and cross-check harmonic weighting against cumulative-risk calculations for distinct ranks. Keep full-dataset evaluation distinct from minibatch training estimates.

## E1 — single-head text model

- Backbone: official ModernBERT-large encoder, without the masked-language-model output head.
- Input: one state/question/options sequence per question. Batch questions independently.
- Readout: gather contextual option-marker embeddings; shared `Linear(hidden,256) -> GELU -> Linear(256,1)`; mask padded options before softmax.
- Output: zero-based winning index, logits, full probability vector; ordinal expected value computed in code.
- Full fine-tuning initially; separate encoder/head learning rates. Proposed starting rates: 1e-5 / 1e-4, adjusted on development data only.
- Start with 512–1,024-token examples; BF16 where supported; select microbatch size after a memory profile. Log throughput and peak memory. Test long-context quality separately.
- Shuffle choice options and remap labels. For ordinal tasks, preserve level semantics and numeric values; do not casually treat reordered positions as the rating values.

Candidate pilot mixture: Banking77, BoolQ, MNLI, Yelp or SST-5, plus verified policy examples. Consider CLINC/OOS after inspecting loaders, labels, and split membership. Rough training scale: 10k–20k examples, balanced by source. Use instruction/label paraphrases as grouped augmentations; keep held-out task families genuinely unseen.

Baseline checks: valid normalized outputs for varying option counts; no label leakage; strong performance over chance; optional small option-permutation diagnostic (no required sweep); source-level accuracy; declared length limits. Include a fixed-class classifier reference for a narrow task if useful, without substituting it for the dynamic-option model.

## Redo E2 before any new research training

The completed CE+AURC run used the 0.5 blend and therefore did not test AURC-weighted loss alone. The completed CE baseline is not a matched CE-only continuation. Treat both as preliminary references; they do not answer the objective comparison.

Before new runs, create the structured trainer/artifact workflow in [training-infrastructure-plan](training-infrastructure-plan.md). Use Hugging Face `Trainer`/`TrainingArguments` for the current ModernBERT custom index head and loss adapter; save Trainer state/checkpoints, local JSON logs, and raw development/calibration logits plus labels/IDs for best and final-budget checkpoints. Do not enable W&B. Axolotl was considered; defer it unless the experiment becomes causal-LM post-training or uses a supported reward-model objective, because the current option-masked index head and AURC surrogate still need custom integration.

Reconcile and retrieve the frozen CE parent checkpoint, pin the shared data/model/code and verify `ssh uab-gpu` versus Modal access, GPU allocation, durable artifacts and recovery. Before full arms, run a short synthetic-fixture smoke/resume and confirm the `.npz` logits agree row-for-row with JSONL labels and IDs. No new Kev training starts until those prerequisites are complete.

## E2 — controlled loss ablation

Train a shared CE warm-up checkpoint for each seed. Branch into:

| Arm | Continuation | Evaluation variants |
|---|---|---|
| A | Ordinary CE (`lambda=0`) | Raw; post-hoc temperature-scaled |
| B | Harmonic AURC-weighted CE surrogate only (`lambda=1`) | Raw; post-hoc temperature-scaled |
| C | CE blended with harmonic AURC-weighted CE (`lambda=0.5`) | Raw; post-hoc temperature-scaled |

Match examples, update counts, optimizer schedule, initialization, and seeds. Use one fixed seed for the initial comparison. Only repeat promising results across additional seeds as a later confirmation; do not budget this into the pilot.

For a ranking batch of B examples, let r_i be ascending confidence rank (higher is more confident). Define alpha_i = H_B - H_(B-r_i). Proposed loss:

`L = (1-lambda) * mean(CE_i) + lambda * mean(stop_gradient(alpha_i) * CE_i)`.

Use the three preregistered lambda values above; do not tune them on test. “AURC-only” means only the detached harmonic rank-weighted CE surrogate, not direct optimization of discrete AURC. Defer any further lambda sweep until the pilot warrants it. Maximum softmax probability is the first confidence function; margin is a later ablation. The reference implementation detaches confidence ranks: gradients flow through weighted CE, not through sorting. There is no separately trained confidence head.

Use the actual ranking-batch size to compute weights; handle incomplete batches correctly. Gradient accumulation does not increase the ranking population. Log ranking-batch size separately from effective optimizer batch size. First rank within comparable task/schema groups; later compare mixed-task ranking to reveal cross-task confidence effects. This grouping defines a different training mixture objective and must be disclosed.

Fit temperature on a disjoint calibration set for each arm. Start with a scalar; consider per-readout temperatures only with enough data and a prespecified protocol. Recompute AURC after temperature scaling: multiclass maximum-probability ordering across examples can change even when each example's winning class is unchanged.

## Metrics and selection

- Primary: per-task error-based AURC and coverage at prespecified target error rates, provisionally 1%, 5%, and 10% where sample sizes support them.
- Select deployment thresholds using development/calibration data; report achieved risk and coverage on untouched test data. Do not interpret a test-optimized coverage threshold as a deployable guarantee.
- Secondary: accuracy/macro-F1, NLL, Brier, ECE with stated binning, full risk–coverage curves, and frequency of confidently wrong predictions.
- Efficiency: **p50 end-to-end latency (↓)**, measured over the same fixed evaluation requests and hardware for each arm. Time from receiving the raw request through preprocessing/tokenization, model inference, and producing the complete option-probability response; include OCR or retrieval only when those are part of that arm's declared serving path. Report warm-up policy, batch size/concurrency, device, and preprocessing/cache state, alongside throughput and p95 where practical. Do not compare model-only forward-pass timing against end-to-end values.
- Ordinal: exact accuracy and MAE; optionally selective risk using normalized ordinal error, clearly separated from 0/1 AURC.
- Report per source, macro averages, and the declared deployment mixture; avoid comparing unnormalized raw losses across task types as if identical.
- Include paired bootstrap intervals, resampling original documents/examples rather than correlated derived questions, plus variability across seeds.
- Declare practical accuracy and calibration non-inferiority tolerances before the main run. AURC gains must not conceal a sacrificed source or accuracy collapse.

Go/no-go: advance when gains are reproducible, practically useful at chosen operating points, and compatible with those tolerances. If gains occur only on familiar tasks, investigate batch composition/data before increasing model size. Estimator consistency does not guarantee deployment risk under distribution shift.

## E3 — document classification and DUDE-derived choices

RVL-CDIP is excluded: the user rejects its scale and age. Do not use original RVL-CDIP. RVL-CDIP-N is explicitly approved as a separate candidate. Start with a small fixed DUDE-derived decision pilot, preserving official document splits and paired OCR/images.

DUDE synthesis:

1. Split by original document before deriving records.
2. Preserve question and annotated answer variants/types.
3. Generate distractors from same-document entities/values of a compatible type first, then supplement with controlled synthetic alternatives.
4. Reject duplicate, equivalent, or also-correct alternatives. Treat lists, multi-answer questions, and unanswerability explicitly; start with auditable single-answer types.
5. Shuffle options and store generation provenance. Inspect a stratified sample manually and run options-only/question-only baselines for shortcuts.
6. Ensure candidate construction does not reveal the answer through style, length, or formatting. Evaluate multiple distractor sets per source case without treating them as independent observations.

Call this a derived multiple-choice DUDE task, not the original generative benchmark. Correct-answer inclusion is part of the derived task definition. If later using an automatic candidate generator, report candidate recall and end-to-end performance separately.

For overlong inputs, compare full text where it fits against question-based retrieval/chunk selection under a fixed token budget. Retrieval must not see gold answers. Measure evidence retention where annotations permit, answer accuracy, and selective risk. Keep original split/group identities intact.

OCR-engine variants offer a natural sensitivity experiment. Synthetic OCR corruption is supplemental. Evidence removal can make the original target unanswerable: do not silently retain it as an ordinary answerable example. Track abstention/reliability separately from a semantic “not present” class.

## E4 — Decision Index integration

Use the public `apolinario/decision-index` engine interface and native scorers. Map ordered internal option indices back to external option keys; return probabilities for every supplied option. Verify adapter and scorer behavior on a small declared subset, keeping linked cases intact.

The documented suite comprises 132,422 requests across 37 benchmarks; the headline panel has 19 benchmarks across five areas. No truncation or option pruning; unsupported, errored, and abstained requests count as wrong in official index scoring. Keep official index scores separate from our selective-prediction curves. Never train on the frozen suite.

The default frozen dataset could not be accessed anonymously during planning. Verify access or exercise the provided pinned-source rebuild route; some upstream access may require acceptance of terms. Do not claim an exact reproduction until input hashes and scoring parity are checked. Preserve model revisions, raw probabilities, status records, latency scope, and hardware.

## E5 — later multimodal extension

Select an image-capable pretrained backbone or vision/text fusion architecture, counting all encoders and heads toward the sub-1B limit. ModernBERT alone does not consume pixels. Retain the same decision interface and one shared option-scoring head where feasible.

Compare OCR-only, image-capable, and combined evidence on matched document splits. Separate model-size/compute effects from modality effects as far as possible. Repeat the CE/AURC comparison only after establishing a useful visual baseline. Test blurry scans, layout-dependent answers, and missing evidence, with label validity audits.

## Deliverables

Versioned dataset manifests and converters; model and loss implementation; estimator tests; experiment configurations; checkpoints; per-example predictions; paired analysis report; Decision Index adapter; short model/data cards documenting limitations. No training, cloud spend, publishing, or external submissions have been performed as part of this planning step.

Source links and project scope: [roadmap](roadmap.md).

## Concrete multimodal candidate (proposed; not yet validated)

Keep the trained ModernBERT option encoder and single scalar output head. Add the vision tower from a small pretrained image encoder, with SigLIP 2 Base as an initial candidate. Obtain spatial patch features, project them into the fusion dimension, and allow contextual option embeddings to cross-attend to those features in one compact residual fusion block. The shared scalar head then scores the visually enriched option embeddings. This adds an internal fusion module, not a second output head. Text-only requests bypass visual fusion.

Start with one document page and its OCR, question, and options. Freeze the vision encoder and cache patch features; initially freeze the text backbone too, training projection/fusion and the shared head on paired document decision labels. Compare against the pre-extension text baseline on the exact same examples. If useful, selectively unfreeze text layers or adapters. Include text-only batches when updating shared components to monitor and limit regression.

The encoders are not automatically aligned: paired supervision must teach the fusion. Global pooled image vectors may suffice for coarse classification but are inadequate as the only representation for fine document QA; preserve patch positions. A low-resolution 224-pixel page loses small text, so initially retain OCR for reading and use vision for layout/appearance. Higher-resolution crops/tiles and their positions are a later experiment with explicit compute costs.

For multipage documents, encode pages independently, retain page identifiers, and limit visual tokens through fixed pooling or question-based page selection. Count image encoding, retrieval misses, and token compression in end-to-end evaluation. Never select pages from gold answers. A small DUDE-derived decision subset is the proposed visual feasibility test. Determine page selection using question-based retrieval, not gold answers; fine-grained multipage expansion follows only if useful.

Count ModernBERT, the vision tower, fusion modules, and head toward the sub-1B limit; verify exact parameter totals from instantiated modules before committing. The chosen vision tower is only part of a dual-encoder checkpoint, so do not confuse the full checkpoint count with the retained tower count. Document whether OCR is an external preprocessing service and account for its cost separately.

Source: https://huggingface.co/google/siglip2-base-patch16-224 (supports use as a vision encoder). The fusion architecture above is our experimental proposal, not a claim from that model card.


Latest scope correction: RVL-CDIP is not part of the planned experiments. The preferred document direction is a bounded DUDE-derived pilot; exact size and answer types remain to be selected after a metadata audit. Original RVL-CDIP links are inventory only. RVL-CDIP-N is explicitly in scope following the user clarification.


## User clarification: RVL-CDIP-N and all three primitives

Original RVL-CDIP remains excluded. RVL-CDIP-N is explicitly in scope. The public card inspected lists 1,002 examples in a single test split; verify whether the user has additional official splits. Preserve it as evaluation-only by default. If chosen for training and no other official splits exist, agree on one fixed document-grouped split and clearly call results a custom protocol, not original test performance.

Train the shared head on all three primitives from the start: Choice (categorical options), Boolean/Noul (false/true), and Score (ordered rubric levels with numeric values). Keep type metadata and explicit instructions. Boolean tasks need grounded negative examples; ordinal tasks need genuine ordinal labels such as review ratings or audited document-specific rubric annotations. Do not reinterpret nominal document class IDs as scores. DUDE can yield answer selection and supported/contradicted candidate verification, with care around missing evidence. Arbitrary candidate mismatch need not mean document-grounded contradiction. A document score task remains to be defined and labeled; it must not be invented from categorical ground truth. Return the same probability vector and index internally, with P(true) or expected rubric value as the corresponding public readout. Evaluate accuracy/selective risk per primitive and ordinal error for Score.

## Latest architecture decision: bounded index head (supersedes option-marker design)

User prefers plain numbered options and bounded integer outputs over special option tags. Default becomes ModernBERT pooled representation plus one fixed-capacity classification head with configurable max_options. Mask slots at or above the actual option count before softmax and loss; return zero-based index and probabilities for supplied options only. Reject requests exceeding configured capacity. Boolean uses two slots; ordinal levels use ordered indices with explicit numeric readout values. Input still includes the natural-language option descriptions, rendered with ordinary numbers. No special option-marker embeddings or per-marker gather required. A proposed initial capacity of 128 fits Banking77; CLINC150 plus out-of-scope requires at least 151. This fixed-capacity design trades arbitrary option counts for implementation simplicity. Do not implement both architectures for the pilot.

## Primary claim and graphical evidence

Hypothesis to test, not an established result: AURC-weighted training provides greater answered coverage at the same low selective risk than matched CE training, including a temperature-scaled CE baseline. Lower ECE alone does not establish useful error ranking. A nearly constant base-rate confidence can look calibrated while offering little rejection value.

Prespecified report panels: (1) risk versus coverage, highlighting low-risk operating regions and fixed error budgets; (2) reliability diagrams with sample counts and ECE definition, alongside NLL/Brier; (3) risk and coverage versus confidence threshold, with validation-selected thresholds evaluated unchanged on test; (4) optional automation utility versus error/defer costs using explicit cost assumptions. Display observed results and grouped bootstrap intervals, not illustrative curves as experimental evidence.

Numerical threshold spacing is not the primary outcome: strictly increasing transformations of a confidence score preserve rankings and risk–coverage curves while stretching/compressing the threshold axis. “More manageable thresholds” must therefore mean better attainable risk/coverage and/or empirical threshold transfer/stability, measured separately. Temperature scaling preserves within-example argmax but can change cross-example maximum-probability ranking for multiclass tasks; recompute all curves.

Compare all three arms on one fixed split/seed for the pilot; raw and separately calibrated readouts need no model retraining. Use per-task plots and macro/declared-mixture summaries, report full-coverage accuracy, and do not claim improvement at every coverage from improved average AURC. Avoid low-risk claims unsupported by sample size. Step-level results do not establish episode-level agent reliability.

## Follow-up: domain shift and RL-based abstention

User-requested potential follow-up, deferred until the text CE/AURC baseline is established. Evaluate fixed in-domain and held-out-domain/task-family splits; preserve official splits where available and avoid repeated resplitting. Measure both calibration and selective prediction out of domain. AURC optimization and its estimator guarantees do not guarantee either property under distribution shift. Jev's training mixture is undisclosed; broad mixture coverage as an explanation for its behavior is a hypothesis, not an established fact.

Compare supervised CE/AURC with a later reward-based fine-tuning approach for answer-versus-defer behavior. First define decision utility, error penalties, deferral cost and coverage constraints; include known-answer, unsupported, ambiguous and shifted examples with defensible labels/outcomes. RL is not inherently less prone to overfitting or more aware of novelty. It can also learn shortcuts or always abstain. Compare matched data and compute, held-out domains, and supervised abstention baselines before attributing benefits to RL. Keep probability calibration objectives separate from action-utility objectives; reward for a sampled correct action need not elicit honest probability distributions.

Future model-family comparisons may include a native multimodal Gemma-family model and a diffusion-based decision implementation. These are candidates to verify at that time, not selected dependencies, confirmed capabilities, or promised speed advantages. Count total parameters and measure matched workloads before revisiting the sub-1B design constraint.

## Benchmark to review

- [ ] Review [jev-benchmarks](https://github.com/thisisandreeeee/jev-benchmarks) as an interesting benchmark for follow-up.
