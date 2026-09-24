# Roadmap

The first milestone is a small text decision model and a fair test of selective prediction. This is an independent Jev-like system, not a reconstruction of its undisclosed training recipe.

1. **V1 baseline (frozen reference):** ModernBERT-large, CLS pooling and a bounded index head, trained on the pinned Kev decision-v7 suite. Preserve the checkpoint, predictions and protocol as the reference point.
2. **V2 architecture baseline:** add an option-aware head closer to Laya: candidate-specific marker representations and shared scoring, rather than fixed output slots. First compare V1 and V2 with the same Kev data, split, seed, CE objective and compute budget; only then compare matched AURC continuations.
3. **Primary research result:** compare raw and temperature-calibrated variants on untouched examples. Seek greater coverage at fixed low risk, with accuracy, probability quality, counts and uncertainty alongside it. Reserve whole sources/rule families for transfer evaluation.
4. **Broader evaluation:** Kev remains the reproducible anchor. Add external datasets as separate, named evaluation suites after the Kev comparison is stable; preserve their official splits and report per-suite results rather than pooling unlike tasks. RVL-CDIP-N_MultiPage is a later document-domain candidate.
5. **Synthetic-data contribution:** add a controlled dose of teacher-generated examples from financial, legal, everyday and general text. Audit semantic correctness before training. Live generation requires a configured endpoint, model, rates and pilot budget.
6. **Documents and later hypotheses:** audit DUDE OCR and RVL-CDIP-N_MultiPage. The latter is 16-class, multi-page PDF data with a test-only split; text-only evaluation needs OCR, while direct PDF/image evaluation needs multimodal support. Original RVL-CDIP remains excluded. Vision fusion, multi-turn rollouts, RL abstention and numeric prediction remain separate hypotheses.

Architecture: V1 full fine-tunes ModernBERT-large with CLS pooling and one bounded index head. V2 is planned as an option-aware candidate scorer. Both should return request-level choices and probabilities, with Boolean false/true and explicit numeric anchors for ordinal tasks.

Use one fixed seed/split for the initial V1/V2 comparison. No cross-validation or mandatory permutation sweep. LoRA remains an optional memory experiment. Direct visual input is deferred until text/OCR results identify a need.

The [experiment register](experiment-register.md) is the authoritative inventory: each experiment records sources, hypothesis, setup, minimum comparison, differentiation and the next action for positive or negative results. The [protocol](experiments.md) explains measurement. [Numeric outputs](numeric-outputs.md) distinguishes selection from regression.
