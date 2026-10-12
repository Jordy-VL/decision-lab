# kotoba-lang/typed-decisions comparison — October 2

Source reviewed at commit `e65215db7e554a31f557936f0bd1d8a0a78b6af5`.
This is a source audit, not an independent reproduction of their reported results.
The current training queue was not changed.

## Most relevant difference: span matching instead of marker-only scoring

Their default `pool=span` mean-pools contextual question text into q and option
text into o, then scores each option with a shared MLP over `[q, o, q * o]`.
Question and option markers, state tokens and padding are excluded from these
means; state still influences the contextual vectors through backbone attention.
The scalar scores are normalized separately within each question.

Our fixed head maps CLS to a fixed vector of slots. Our candidate head applies
a shared scalar MLP to each newly added marker's hidden state. It has no explicit
question-text pooling or matching interaction. Their alternatives `opt` and
`q-opt` score an option marker alone, or paired question/option markers; those
are not their default. The file's introductory docstring still describes marker
scoring, so the implementation and `pool` branch matter more than that summary.

[Encoder implementation](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/encoder.py).

## Input and state preparation

| Aspect | Decision Lab | typed-decisions encoder |
|---|---|---|
| Sequence | State, question, numbered options | CLS, STATE marker, state, then Q/instruction and OPT/text groups, SEP |
| Questions per sequence | One; state repeated for each question | Many questions share one state and one backbone pass |
| Option identities | Zero-based visible numbers; adapter also includes original criterion keys | Plain option text following OPT markers; no numeric prefix in encoder collator |
| State representation | Strings or sorted JSON serialization | Core schema and public wrapper expect a string |
| Length policy | Reject overlength, no truncation; current max 2,048 | Preserve complete questions/options; truncate state tail to remaining budget and configured state cap |
| Default training budgets | 2,048 total tokens | 8,192 total, at most 512 state tokens; actual bundle can override |
| Option capacity | Configured 128/256 | Dynamic per-batch option dimension; public choice API validates 2–255 |

Their collator first counts all question and option tokens, allocates the
remaining state budget, and raises if questions alone cannot fit. Truncation
means its coverage is not comparable to our no-truncation benchmark policy.
The encoder receives a regular padding attention mask: question groups are not
isolated by attention. Therefore adding/reordering other questions can affect
answers and should be tested. Multi-question packing is also a throughput change,
not merely a different classifier head.

[Collator](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/encoder.py),
[public wrapper](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/open_jev.py).

## Training and output differences

- Their optimizer defaults to backbone LR 3e-5 and head LR 1e-3. Our running
  configs use 1e-5 for both. Our candidate hidden layer is 128; their matching
  head uses backbone hidden width, so head capacity differs too.
- Their default loss is CE + Brier (weight 1), versus CE in our active runs;
  our earlier experiments also include AURC/AUGRC objectives.
- They implement optional augmentation and symmetric KL consistency between
  two forward passes, aligning distributions by option text after shuffling or
  distractor removal. Both controls default to zero: support does not mean every
  published run used them. Our new categorical permutation is enabled in the
  active configs; it has no consistency loss.
- Their Noul is probability of yes from no/yes options; ours uses false/true.
  Their score readout is the expected zero-based level. Our canonical ordinal
  rows carry explicit numeric anchors; our Decision Index adapter accepts choice
  and noul, not score requests.

[Training](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/train_encoder.py),
[augmentation](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/augment.py),
[schema/readout](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/schema.py).

Do not import their augmentations indiscriminately: score synonym templates are
task-dependent, and some negation wording is ambiguous. Their training code
already skips generic augmentation for examples marked with target families.
Option shuffling with label remapping is the cleaner first controlled comparison.

Their separate LLaDA path uses one pretrained mask position per question and
restricts vocabulary logits to single-token option labels. It is neither their
encoder span head nor our candidate-marker strategy.
[dLLM source](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/src/typed_decisions/dllm.py).

## Implication for our next ablation

They report marker-based small-set memorization alongside weak larger-run
learning, resembling our observations. They report improvements from span
pooling on ModernBERT-base, but ModernBERT-large remained weak in their smaller
span experiments. Their explanation about fresh-marker representations is a
hypothesis, not proof of our failure's cause. Datasets, objectives, budgets and
backbones differ; their accuracies are not directly comparable to ours.
[Reported ablations](https://github.com/kotoba-lang/typed-decisions/blob/e65215db7e554a31f557936f0bd1d8a0a78b6af5/README.md#the-ablation-that-mattered-what-the-head-reads).

The next useful architecture experiment is a separate question/option span
matching head on the same backbone, data, permutation protocol and length policy.
Test head learning rate separately. Keep one question per input initially, so
head pooling is not confounded with packing or state truncation. Mask padded
tokens out of means, validate nonempty spans, and test semantic permutation
consistency. The running marker-head and fixed-slot jobs remain valuable controls.
