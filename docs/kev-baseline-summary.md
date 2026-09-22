# Kev baseline: what we are comparing against

Checked September 22, 2026. Kev is an independent open reproduction of the Jev decision interface. Its published data is the official source for **Kev's reproduction**, not TypeSafe's original Jev training mixture. Our implementation and research extensions are separate.

## Model and training

Kev's current family uses Qwen3.5 base models at 0.8B, 4B and 9B. It predicts distributions over categorical choices, Boolean decisions (`noul`) and ordered levels (`score`), without generating answer text. A dynamic pointer head scores options; LoRA adapts the backbone. This differs from our fully fine-tuned ModernBERT-large and bounded index head. [Source: repository](https://github.com/jaredpalmer/kev)

The base recipe uses cross-entropy, rank-16 LoRA, two epochs, and option/distractor/none-option augmentation. The 0.8B learning rate is 1e-4; larger models use 5e-5. These augmentations mean matching the raw records alone does not exactly reproduce the effective training stream. [Source: recipe](https://github.com/jaredpalmer/kev#training)

The current 0.8B checkpoint also received a later delta stage: 1,425 generated records mixed with 2,000 replayed training records, teaching date-related decisions and uncertainty when evidence is absent. Its `v7-base` revision predates that stage. Therefore our initial decision-v7-only experiment should use the pre-delta model as the closest data-matched external comparison, or clearly disclose the extra data in a current-checkpoint comparison. [Source: 0.8B model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-0.8b.md)

## Training data and frozen partitions

Decision-v7 contains 10,000 public training records: 1,000 each from Banking77, BoolQ, AG News, MNLI, SST-5, Yelp, TREC, DBpedia, Amazon reviews and IMDb. It adds 896 programmatic policy records and 1,680 compositional logic records. One record can contain several questions.

| Partition | Records | Questions |
|---|---:|---:|
| Train | 12,576 | 15,576 |
| Calibration | 968 | 1,148 |
| Development | 1,204 | 1,468 |
| Test | 1,176 | 1,440 |

The in-domain evaluation uses held-out examples from trained source families, not the training rows. Do not assume identical proportions or identical generated rule structures across partitions: evaluation is inherited from earlier frozen suites. The manifest explicitly does not certify overlap freedom between inherited legacy test and newer synthetic controls. [Source: decision-v7 manifest](https://github.com/jaredpalmer/kev/blob/main/evals/v7/decision-v7/manifest.json)

## Evaluation beyond familiar datasets

Transfer-v4 has 764 development and 764 test records, with no training or calibration partition. It includes MMLU, Emotion, TweetEval offensive, QNLI, PAWS and SciQ, plus held-out policy/composition tasks. These are unseen in Kev's supervised mixture; that is not evidence they were absent from Qwen pretraining. [Source: transfer-v4](https://github.com/jaredpalmer/kev/blob/main/evals/v4/transfer-v4/manifest.json)

Transfer-v9 extends that suite to 1,264 records per evaluation partition. Additions cover ten-option MMLU-Pro, buried evidence, and paired missing-evidence/intact controls. Evidence-free questions test uncertainty, not ordinary answer accuracy. [Source: transfer-v9](https://github.com/jaredpalmer/kev/blob/main/evals/v9/transfer-v9/manifest.json)

## What their scores mean

The benchmark reports accuracy, NLL, Brier, ECE, confident errors, selective risk/coverage and AURC, plus option-order and paired-rule diagnostics. Its primary objective is macro task NLL; our pooled question accuracy is not automatically the same aggregate. Coverage at a target error is an in-sample threshold frontier, not a guarantee for a threshold deployed on new data. AURC uses a right-step integral over whole confidence-tie groups; align this with our implementation before numerical comparisons. [Source: benchmark implementation](https://github.com/jaredpalmer/kev/blob/main/kev/benchmark.py)

Published served probabilities use temperature scaling fitted on in-domain **development** data; the dedicated suite calibration partition should not be confused with that published fitting procedure. Our planned separate calibration partition is a deliberate protocol difference. Always label raw versus calibrated results. [Source: 4B model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)

## Our first experiment

The adapter has verified and prepared the exact pinned train/calibration/development files, preserving labels, option order, groups and source metadata. Test remains unprepared. All prepared examples fit 2,048 ModernBERT tokens. See [preparation and commands](kev-baseline.md) for revision pins and validation evidence.

First run CE on these records; then compare matched CE and CE+AURC continuations. Keep development for model choices and calibration for temperature/threshold selection; apply frozen thresholds to final test. Report the retrospective test frontier separately. Extra synthetic data, vision and RL come later. Transfer loaders and exact upstream metric parity remain work to complete before claiming full Kev benchmark reproduction.

No research GPU training has run. The [Modal guide](modal-first-baseline.md) is ready; its launch wrapper and GPU profiling are pending. User actions are in the September 22 vault checklist. The user-supplied AURC evaluation implementation remains awaited.
