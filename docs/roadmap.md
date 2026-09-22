# Roadmap

The first milestone is a small text decision model and a fair test of selective prediction. This is an independent Jev-like system, not a reconstruction of its undisclosed training recipe.

1. **Ready to inspect:** two uv workspace packages, grounded text generator, minimal ModernBERT trainer, bounded index interface, comparison plots, and offline smoke checks. No trained research model yet.
2. **First real pilot:** audit a small fixed mixture, preserve official splits, fit a CE warm-up, then matched CE and CE+AURC continuations. Evaluate categorical, Boolean and ordinal tasks separately and together. SST-5 and a modest Yelp sample supply ordinal examples.
3. **Primary research result:** compare raw and temperature-calibrated variants on untouched examples. Seek greater coverage at fixed low risk, with accuracy, probability quality, counts and uncertainty alongside it. Reserve whole sources/rule families for transfer evaluation.
4. **Synthetic-data contribution:** add an equal-sized controlled dose of teacher-generated examples from financial, legal, everyday and general text. Audit semantic correctness before training. Live generation requires a configured endpoint, model, rates and pilot budget.
5. **Documents:** inspect DUDE OCR token lengths and convert gold answers into checked alternatives. RVL-CDIP-N is a possible evaluation source; original RVL-CDIP is excluded. Preserve document groups and official partitions.
6. **Later hypotheses:** vision with a small SigLIP encoder/fusion, multi-turn rollouts, RL abstention under domain shift, and numeric prediction only if a use case warrants a new primitive.

Architecture: full fine-tuning of ModernBERT-large, CLS pooling and one bounded index head. Number options normally in text; mask unused outputs. Return zero-based index and probabilities. Boolean uses false/true; ordinal uses explicit ordered numeric anchors. No option-marker tokens or per-option scalar head in this implementation.

Use one fixed seed/split for the initial pilot. No cross-validation or mandatory permutation sweep. LoRA remains an optional memory experiment. Vision is deferred until text results identify a need.

The [experiment register](experiment-register.md) is the authoritative inventory: each experiment records sources, hypothesis, setup, minimum comparison, differentiation and the next action for positive or negative results. The [protocol](experiments.md) explains measurement. [Numeric outputs](numeric-outputs.md) distinguishes selection from regression.
