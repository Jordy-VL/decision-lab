# Selective prediction with compact indexed decision models

*Working manuscript — baseline methods and prospective experimental protocol. September 22, 2026. No research-scale training or evaluation results are available.*

## Abstract

Decision models map evidence and task-specific alternatives to a bounded probability distribution. Their usefulness in partially automated workflows depends both on predictive accuracy and on whether confidence identifies decisions that should be deferred. We propose a controlled study of a compact encoder trained with cross-entropy and a continuation objective motivated by area under the risk–coverage curve (AURC). Our baseline combines ModernBERT-large with a single masked index-classification head and uses the published Kev decision-v7 training records and predefined partitions. Matched continuation experiments will isolate the training objective while evaluating calibration and selective prediction separately. This draft specifies the baseline, data preparation and evaluation protocol; it does not report evidence of improved utility.

## 1. Motivation and research question

Probability calibration and selective prediction answer different questions. Calibration concerns agreement between predicted probabilities and observed outcomes, whereas selective prediction concerns the errors among examples retained by a confidence-based acceptance rule. Lower expected calibration error (ECE) alone need not produce a better ordering of correct and incorrect decisions.

Our primary question is whether an AURC-motivated training objective increases achievable coverage at low empirical risk relative to an otherwise matched cross-entropy continuation. A secondary question is whether thresholds selected on source-distribution calibration data remain useful on held-out domains. Neither improved calibration nor reliable transfer is assumed in advance.

## 2. Task and model

An example consists of a state s, question q, ordered candidate descriptions o_0 through o_(K−1), and a target index y. We serialize the state, question and numbered candidates into one text sequence. Boolean decisions use the ordered alternatives false and true; ordinal decisions use ordered rubric levels. All three primitives share the same output head. The returned index identifies a candidate in the current request, not a global semantic class.

The encoder is [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large). Its first-token representation h is passed through a two-layer MLP, z = W₂ GELU(W₁h + b₁) + b₂, with hidden width 128 and 128 output slots. Logits for slots outside the example's K candidates are masked before softmax. The prediction is argmax p, and initial selective confidence is max p. Full distributions and logits are retained for subsequent analysis.

All encoder and head parameters are fine-tuned. This architecture differs from candidate-pointer or independent option-scoring systems. In particular, the fixed slot head does not enforce option-permutation equivariance. We preserve upstream option order in the first experiment; option-order robustness is not an established property of this baseline.

Complete serialized inputs are limited to 2,048 tokens. Oversized sequences or candidate sets are rejected rather than silently truncated. The configured length is an experimental budget, not a claim about the encoder's maximum supported context.

## 3. Data and provenance

We use [Kev's decision-v7 reproduction suite](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/evals/v7/decision-v7/manifest.json), obtained through its [pinned Hugging Face mirror](https://huggingface.co/datasets/jaredpalmer/kev-suites/tree/a88f56db5341397299137cb68775c2ea6e3f68cb/v7/decision-v7). These are Kev's published reproduction data, not TypeSafe's original Jev training mixture.

The training partition combines 10,000 records from ten public sources—Banking77, BoolQ, AG News, MNLI, SST-5, Yelp, TREC, DBpedia, Amazon reviews and IMDb—with 896 policy records and 1,680 compositional records. Our adapter expands multi-question records into individual training examples while retaining their source/group identifiers, original labels, candidate order and partition membership. We introduce no additional synthetic data or rebalancing.

| Partition | Source records | Decision examples | Role in our protocol |
|---|---:|---:|---|
| Training | 12,576 | 15,576 | Parameter updates |
| Development | 1,204 | 1,468 | Model/protocol selection |
| Calibration | 968 | 1,148 | Temperature and acceptance-threshold fitting |
| Test | 1,176 | 1,440 | Final frozen evaluation |

Training, development and calibration downloads have passed manifest hash/count checks and adapter validation. Their maximum serialized ModernBERT lengths are 1,140, 1,124 and 1,078 tokens respectively. These are preparation measurements, not model results. Test counts are taken from the upstream manifest; test examples have not been downloaded for this experiment.

No duplicate question IDs, shared group IDs or identical canonical state text were found across the three prepared partitions. These checks do not establish semantic or pretraining decontamination. Upstream also does not certify absence of overlap between inherited legacy test examples and newer synthetic controls. The in-domain partitions should not be described as having identical mixture proportions.

## 4. Training protocol

The initial baseline minimizes mean cross-entropy, L_CE = −mean(log p_y). The committed pilot configuration specifies one epoch, seed 17, batch size 4, gradient accumulation 1, learning rate 2×10⁻⁵, weight decay 0.01, and gradient checkpointing. These are initial settings, not tuned or validated GPU settings. The backbone revision must be pinned before a research run; the checked-in placeholder currently follows the model's main revision.

After this shared CE warm-up, two continuations will start from the same checkpoint with matched data order, seed, optimizer initialization and update budget. One retains CE; the other uses the independently implemented harmonic rank-weighted CE surrogate motivated by [AsymptoticAURC](https://github.com/han678/AsymptoticAURC). The planned blend coefficient is 0.5. Confidence ranks are detached from differentiation and computed within actual microbatches; gradient accumulation does not enlarge the ranking set. With small microbatches, ranking quality is itself a limitation to investigate before scaling or making strong claims.

Our baseline is a controlled ModernBERT experiment on Kev records, not an exact reproduction of Kev's architecture, augmentation or two-epoch LoRA recipe. Current Kev releases can also include a later training stage; its pre-delta revision is the closer external data comparison. See the [Kev summary](../docs/kev-baseline-summary.md).

## 5. Evaluation and interpretation

For each continuation, report raw and temperature-scaled predictions. Temperature fitting uses calibration data only. Candidate acceptance thresholds are selected there and applied unchanged to test. Report full-coverage accuracy, NLL, Brier score, ECE with binning details, AURC, risk–coverage curves and achieved coverage/risk at the frozen thresholds. Ordinal error is reported by source and scale. Report per-type/source results alongside pooled results because candidate count and mixture composition affect confidence and aggregate scores.

For an acceptance rule A_t = {i : confidence_i ≥ t}, coverage is |A_t|/N and empirical risk is the fraction of incorrect predictions in A_t. Empty acceptance sets have undefined risk and must be identified explicitly. The retrospective best test-set frontier is a separate oracle diagnostic; thresholds selected using test labels are not deployment results or evidence of a population risk guarantee.

Lower AURC may reflect better ordinary accuracy as well as improved confidence ranking. We therefore interpret curves jointly with full-coverage error and accuracy. Source-calibrated evaluation on entirely held-out task/domain families is a later stage; Kev transfer suites are candidates, but their adapters and metric parity are not yet verified locally. This experiment is also distinct from a full Decision Index submission.

The initial pilot uses one seed and point estimates. Publication-level conclusions require appropriate group-aware uncertainty estimates and justified repeat runs. Confidence-tie handling and aggregation conventions must be reconciled before comparing against external AURC numbers. Further threshold implementation changes await review of the user's existing evaluation code.

## 6. Results and reproducibility status

**Results pending.** Tiny CPU smoke runs establish basic software operation only. No table of research accuracy, ECE or AURC is populated. The first GPU attempt has a user-authorized total budget of USD 10; actual hardware, precision, elapsed time and cost will be reported after execution. This budget is not evidence that the full experiment fits within it.

Each run must retain code/data/model revisions, resolved configuration, checkpoint lineage, split provenance, prediction artifacts and metric definitions. Failed and negative runs remain in the run ledger. The Modal launcher and GPU verification are pending. See the [experiment register](../docs/experiment-register.md), [data preparation](../docs/kev-baseline.md) and [Modal guide](../docs/modal-first-baseline.md) for operational details.

If selective utility improves without an unacceptable accuracy tradeoff, proceed to frozen-threshold transfer. If only calibration improves, report that narrower finding. If no useful gain is observed, inspect data semantics and ranking-batch limitations before increasing compute. Multilingual distillation, RL-based calibrated decisions, vision and external Decision 1.0 comparisons remain separate follow-up experiments.
