# Baseline training-data audit

2026-09-22. Different projects do not share one universal fine-tuning mixture. Matching dataset names is insufficient: match records, transformations, partitions and checkpoint stage.

| Project | Data situation | Comparison role |
|---|---|---|
| Kev | Published decision-v7 recipe with public sources and generated rules; newer checkpoints also have a delta/replay stage | Preferred matched-data anchor, conditional on verified training artifact access |
| Dev-0.4B | Card credits Banking77, BoolQ, CodeSearchNet and Yelp; run metadata names local train_refine/validation_refine JSONL files | Same-backbone reference; exact training-file parity not established |
| SemIf | Main direct-logit baseline uses frozen pretrained models; separate labeled calibration data | External inference/calibration baseline, not matched fine-tuning |
| jevlike | Default synthetic task and optional task-specific/pretrained-encoder paths | Architectural control; not evidence of a common mixture |
| Jev | Full training corpus and reproducible RLCD specification unavailable in inspected public materials | External service comparator, not a controlled training-data reproduction |

Sources: [Kev recipe](https://github.com/jaredpalmer/kev#training), [Kev model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-0.8b.md), [Dev card](https://huggingface.co/mpnikhil/dev-0.4b) and [run metadata](https://huggingface.co/mpnikhil/dev-0.4b/blob/main/run.json) (locally retained inspection from earlier today; web fetch unavailable during this follow-up), [SemIf method](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/METHOD.md), [jevlike](https://github.com/vinnylarouge/jevlike), [TypeSafe launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Kev decision-v7 public sources: Banking77, BoolQ, AG News, MNLI, SST-5, Yelp Review Full, TREC, DBpedia-14, Amazon reviews and IMDb. The manifest distinguishes records from questions: multiple questions can share a record. Preserve this grouping when flattening to our per-question trainer.

Decision: keep new synthetic/domain mixtures out of the initial experiment. Freeze one accessible suite, retain its splits, and run our matched CE/AURC arms. Document the backbone, head, optimizer and augmentation differences when comparing against released Kev. Use its pre-delta checkpoint for the base-suite comparison rather than treating its later refined checkpoint as identically trained.

Attribution wording: **Kev's official reproduction source and published suite**. This is not official TypeSafe/Jev training data, and we did not originate the records. Our contribution here is the format adapter, our model/training comparison and its evaluation.

Access status and exact adapter behavior belong in the dedicated Kev loader documentation. An advertised manifest is not proof that all training bytes are downloadable. Do not silently regenerate approximate substitutes and label them the same baseline.
