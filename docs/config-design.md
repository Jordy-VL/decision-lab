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

Preference to investigate: subclass a standard Hugging Face model configuration when packaging the custom encoder/head for compatible save/load. Keep dataset mixing and training settings in the top-level experiment config rather than putting them inside the model's configuration object. Retain the backbone configuration and explicitly serialize decision-head settings. Verify the installed Transformers API before implementing this; no subclass or new compatibility claim is introduced by this note.

Order: exact baseline loader, matched CE/AURC experiment, then optional synthetic mixtures and balancing. Any configuration migration must preserve the working flat CLI or provide a clear migration path.
