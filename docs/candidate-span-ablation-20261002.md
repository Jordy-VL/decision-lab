# Question/option span head ablation

Implemented locally; no full training launched. Inspired by the [typed-decisions source audit](typed-decisions-comparison-20261002.md).

`candidate_spans` pools question text and each option text separately, then scores each option with a shared MLP over `[question, option, question * option]`. Padding, headings, numeric prefixes and candidate markers are excluded from pooling. They remain visible to encoder attention, preserving the candidate-marker baseline's input exactly. Contextual attention can still introduce order dependence: shared pooling does not guarantee encoder permutation invariance.

Matched CE configs for 128 and 256 options are in `packages/modernbert-decisions/configs/x2-candidate-spans-ce-10epoch-linear-1e5-permuted-{128,256}.yaml`. They retain v7 data, seed 17, epoch-wise categorical permutations, ModernBERT-large, head hidden width 128, 2048-token rejection limit and the same optimizer/schedule. Start fresh, not from a marker checkpoint. The matching feature vector increases first-layer head parameters; record parameter counts when comparing. This is a head ablation, not a complete typed-decisions reproduction.

Local validation: 20 package tests pass, including explicit pooled-logit agreement, gradients, padded-option probability zero, span reassignment consistency, Unicode/multiline input, invalid/mixed spans, and 256-option save/load. The reassignment test isolates the scorer with a token-local test encoder; it is not evidence that a contextual trained model is invariant.

Next: full corpus encoding preflight, tiny pretrained fit, then matched training and held-out option-permutation diagnostics. Keep remote queued training sources unchanged: the running fixed/candidate-marker queue checks their hashes before each arm. Do not sync this new head into that frozen source tree until the queue completes.
