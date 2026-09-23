# Configuration and data-loader direction

User decision, 2026-09-22. The trainer now accepts either sectioned YAML or legacy flat YAML, with the existing flat dataclass/argparse fields as its single definition. Saved resolved config remains flat. See [integration review](config-integration-review.md). Mixture sampling and Hugging Face wrapper changes remain deferred.

## One experiment config

Organize the eventual single YAML into three sections:

- **data:** pinned baseline suite, explicit predefined partition files and hashes, format adapter, and optional additional training sources.
- **model:** pretrained backbone/revision and decision-head capacity/settings.
- **training:** seed, optimizer, batch/accumulation, epochs, CE/AURC settings and output paths.

Preserve one CLI entry point, overrides and saved resolved YAML. Avoid a general configuration framework or duplicated field definitions.

## Baseline loader first

Create a dedicated adapter for the pinned competitor suite. Preserve supplied train/development/calibration/test roles rather than re-splitting. If no calibration partition exists, explicitly specify the reserved partition/protocol; never silently repurpose test. Validate source hashes, group separation, option order, target mapping and ordinal semantics. The same adapter serves both CE and AURC. Record upstream training augmentations separately from source files.

## Optional mixtures later

Permit additional labeled sources, including synthetic generator exports, only in the training mixture by default. Each source declares its path/revision, format, provenance and optional sampling weight or quota. Pick one simple balancing mechanism initially; do not expose overlapping weight/quota controls without defined precedence.

With no additions, the loader must produce the original baseline unchanged. For mixture experiments, record realized source counts and whether sampling repeats examples; keep epoch/update budgets explicit. Freeze evaluation partitions and prevent document/derived-example leakage across sources. Hold the mixture fixed across CE/AURC arms. Data addition and balancing are separate experimental variables, not baseline prerequisites.

## Hugging Face integration

For current ModernBERT experiments, use `Trainer` and `TrainingArguments` for optimizer/scheduler management, structured logs, resumable checkpoints, periodic development evaluation and prediction. The decision model returns a Hugging Face `SequenceClassifierOutput`; a small `DecisionTrainer` adapter supplies the option-masked CE/AURC objective. Keep data/model/training settings in the experiment YAML and write each resolved config into its run folder. Use `report_to: none`; local Trainer logs and explicit logits/prediction bundles are the source of truth.

The project still owns its small inference checkpoint wrapper because the custom option-index head is not a stock Transformers architecture. Trainer checkpoints use the standard state files for interruption recovery; final/best model folders and evaluation artifacts are also saved in project format. This is not yet an `AutoModel.from_pretrained()` packaging claim.

Axolotl remains a deferred option for a generative fine-tuning/RL or supported reward-model track. It is not selected for this option-masked ModernBERT experiment because we would still need a custom model/objective integration. Review [the trainer decision and redo plan](training-infrastructure-plan.md) before broader experiments.

Order: exact baseline loader, Trainer artifact/recovery smoke, matched CE/AURC objective matrix, then optional synthetic mixtures and balancing. Any configuration migration must preserve the working CLI or provide a clear migration path.
