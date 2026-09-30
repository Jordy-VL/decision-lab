# Synthetic choice coverage and NeMo Data Designer

Recorded 2026-09-25 after Jordy's observation that the fixed 128-slot choice head mostly uses its earliest outputs. This is an investigation and proposed follow-up, not a training run or a change to the frozen baseline.

## Verified local findings

Audited only `data/kev-decision-v7/train.jsonl`, using the stored categorical `options` and `target_index` fields. No final test file was read. The file contains 15,576 rows: 6,904 categorical, 5,224 Boolean, and 3,448 ordinal. Audited SHA-256: `39fc904150b9fc5265ae17ac38723027d49067cb3a660183af6be97896b42b70`; the maximum option count across all three types is also 77.

| Number of categorical options | Training examples |
|---|---:|
| 2 | 1,680 |
| 3 | 1,224 |
| 4 | 1,000 |
| 6 | 1,000 |
| 14 | 1,000 |
| 77 | 1,000 |

- 5,752 / 6,904 categorical targets (83.3%) are at zero-based positions 0-9.
- Only 2,000 / 6,904 categorical rows have more than ten options. The first-six-position exposure is largely explained by short candidate lists; answer-position bias and candidate-count imbalance are different measurements.
- In `packages/modernbert-decisions/decisions/model.py`, `collate` activates a contiguous prefix of length equal to the option count. `forward` masks all other logits before the loss. The categorical training data therefore supplies no task-learning signal to slots 77-127 (51 slots). Weight decay is not useful supervision for those outputs.
- This audit is about the prepared training file and masking implementation. It does not establish what every historical run consumed, nor measure deployment or test-time degradation. Verify a run's data hash and any augmentation before attributing its results to this file.
- The existing generator already preserves source groups, records provenance, rejects invalid schema/duplicate choices and supports an independent answering check. An evidence span proves quote presence, not semantic label correctness.

## Proposed experiment

Preserve V1 and the existing baseline data/checkpoints. Treat this as the later X3 synthetic-data comparison in the [experiment register](experiment-register.md), with X12's option-aware scorer as a separate architecture experiment.

1. **Define the desired coverage before generation.** Use expected application requirements, not final test labels. Track primitive, source domain, choice count, target position conditional on choice count, semantic difficulty, ambiguity and actual tokenizer length. Suggested diagnostic choice counts are 8, 16, 32, 64, 96 and 128; these are stress-test strata, not a claim about deployment prevalence.
2. **Use a cheap augmentation control.** Shuffle categorical options with an exact target remap and preserve source/group/split IDs. Check instructions for positional references. Do not shuffle Boolean false/true or ordered numeric rubrics in violation of their schema. This can address position dependence within a candidate list but cannot activate slots beyond that list's size.
3. **Generate real large-candidate decisions.** Use independent, licensed source material and bounded task families such as routing among defined intents or selecting a record satisfying explicit constraints. Include plausible hard negatives. Do not append empty, duplicated, irrelevant or obviously impossible options merely to fill 128 slots. Require a uniquely supported answer; reject unresolved multiple-answer or unsupported cases.
4. **Control target position explicitly.** Sample the desired correct position uniformly conditional on each generated choice count, then place the validated answer and remap labels. Stratify output coverage; do not rely on an LLM prompt alone to produce balanced indices. Record active-slot and positive-target counts separately.
5. **Start with an audited pilot.** For example, at most 100 independent source groups with a small fixed number of questions each, subject to a separately configured generation budget. Report rejection/shortfall rates, near-duplicate checks, per-stratum counts and manual correctness checks. Independent model agreement is a useful filter, not ground truth.
6. **Audit actual context costs.** Tokenize full state, question and all options using the pinned model tokenizer; report rejected/unsupported lengths. Larger candidate sets must not be silently truncated or shortened using knowledge of the answer. Fewer high-quality long examples may be more informative than a large noisy batch.
7. **Compare bounded variants.** Frozen baseline; categorical permutation control; baseline plus audited high-choice synthetic data. Fix the parent initialization and specify both optimizer steps and training-token/compute budgets, since 128-choice examples cost more. Record any trade-off when both cannot be matched. Keep synthetic mixture weight modest and explicit; check small-choice performance for regression.
8. **Evaluate on separate development/calibration diagnostics first.** Report accuracy, NLL, Brier, AURC and coverage at fixed risk by choice-count band and target-position band, with sample sizes. A paired categorical permutation diagnostic checks whether predictions follow the answer content. Fit calibration separately and select thresholds without final-test tuning. High-count, independently generated diagnostics need their own held-out source groups and must not be mislabeled as the original benchmark.

No model call, generation run, training job, checkpoint change or benchmark update was performed for this note.

## Where NeMo may help

[NVIDIA NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) supports sampled columns, dependencies between fields, seed data, structured generation and custom validation. Those capabilities could encode coverage quotas and generation/validation stages. Its usefulness here is a hypothesis; the existing generator may be sufficient for the first pilot.

- Map generated records into the existing canonical/trainer schema, preserving provenance and the categorical target remap.
- Use deterministic validators for schema, counts, bounds, duplicate options and verifiable constraints; reserve model checks for semantic support.
- Keep the framework behind the existing data contract rather than changing the trainer interface for a tool experiment.
- Confirm a selected model endpoint, actual token rates, licence/usage terms, maximum requests and total budget before live generation. The request to investigate does not launch paid calls.

Official references: [repository](https://github.com/NVIDIA-NeMo/DataDesigner), [tutorials](https://docs.nvidia.com/nemo/datadesigner/latest/tutorials/overview), [basic generation tutorial](https://docs.nvidia.com/nemo/datadesigner/tutorials/the-basics). Reviewed 2026-09-25.

Additional user-selected reference: [Data Designer Got Skills](https://docs.nvidia.com/nemo/datadesigner/dev-notes/data-designer-got-skills). Its agent-focused CLI and skill could reduce configuration work for this pilot. Its benchmark results concern NVIDIA's own comparison, not decision-model accuracy or our measured generation efficiency. At the user's explicit request, NVIDIA's skill was installed in the Windows Codex skill directory on 2026-09-25; the Python runtime and provider setup were not changed.

## Architecture distinction

The planned X12 shared candidate scorer can reuse its parameters across positions, reducing dependence on separate fixed output neurons. It still needs varied choice counts, hard negatives and context-length testing; changing the head does not prove generalisation to larger candidate sets. Keep data and architecture effects separately identifiable.
