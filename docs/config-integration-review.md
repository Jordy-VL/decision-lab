# Configuration integration review

Reviewed 2026-09-22 against the working trainer and locally installed Transformers **4.57.6**. Nested YAML support is implemented; the proposed Hugging Face wrapper is not.

## Decision

Keep one experiment YAML, one argparse entry point and the saved resolved configuration. Separate **data**, **model** and **training** concerns. Establish the frozen baseline adapter before adding source mixing. Keep the current model checkpoint format for the first baseline run; a custom Hugging Face model wrapper is a separate compatibility change, not required to train ModernBERT.

The current [config module](../packages/modernbert-decisions/decisions/config.py) defines fields once in a dataclass, derives argparse flags from those fields, applies command-line values after YAML, rejects unknown fields/types and saves resolved YAML. Retain these properties. The [model module](../packages/modernbert-decisions/decisions/model.py) already uses `AutoModel` and `AutoTokenizer` for the backbone; the enclosing `DecisionModel` is an ordinary `torch.nn.Module` with its own `save`/`load` methods.

## Smallest useful migration

Implemented the smaller migration: a section-to-field name map converts nested YAML into the existing `Config` dataclass. That dataclass remains the single definition of types/defaults and still generates all CLI flags. No Hydra, Fire or separate framework is added. Legacy flat YAML and CLI flags remain valid; mixed flat/nested YAML is rejected rather than resolved silently. Three nested dataclasses can be introduced later if callers need section objects, without changing the user-facing YAML.

The runnable example is [ce-baseline.yaml](../packages/modernbert-decisions/configs/ce-baseline.yaml). `data.path`, `data.revision`, `data.split` map to the existing flat `data`, `data_revision`, `data_split`; `model.name` maps to flat `model`. Other keys keep their original names within the appropriate section. Command-line overrides stay flat, for example `--data PATH --batch-size 2`. `save_config` deliberately still writes **resolved flat YAML**, which remains reloadable and compatible with existing checkpoints and run consumers. Paths remain relative to the working directory.

Parser smoke verification passed in the existing CPU Python environment: old flat YAML, new nested YAML, CLI and Boolean override precedence, saved flat round-trip, and rejection of five invalid/misplaced/mixed configurations. No model weights were loaded and no training ran for this change.

| Section | Existing settings to place there |
| --- | --- |
| `data` | Input path, dataset revision, declared split/format, fallback split fractions; baseline manifest and its source hashes when the adapter exists |
| `model` | Backbone identifier/revision, `head_hidden`, `max_options`, token budget (`max_length`) |
| `training` | Seed, optimizer, batch/accumulation, epochs, CE/AURC options, gradient checkpointing, device, output, checkpoint and evaluation split |

`training.split` is a command selection, not permission to generate a new split. `data.split` labels a file via the existing flat `data_split` setting; it does not override declared roles on rows. Resolve paths consistently with the existing working-directory convention and document it. Do not silently switch relative paths to the YAML directory.

On resume, the loaded checkpoint continues to own head dimensions; save these actual values in the resolved YAML, as the current initializer does. Store baseline revision, conversion version and source file hashes alongside that YAML. A weights-only continuation is still a fresh optimizer, not an exact interrupted-run resume.

The baseline must be usable with a single source and no mixing fields. Later, add a `data.sources` list and one well-defined training-only sampling mechanism. Do not implement weights, quotas and oversampling modes simultaneously. Preserve evaluation partition membership and report actual sampled counts. Synthetic additions must remain a separate experiment variable.

## `AutoConfig` versus a custom configuration

`AutoConfig` is a factory and registry; it is **not** the class to subclass. A future `DecisionConfig(PretrainedConfig)` can contain the backbone configuration and explicit index-head dimensions, with a distinct `model_type`, such as `modernbert_decision`. Its constructor must accept `**kwargs` and pass them to the superclass. If using composition, preserve the encoder's own `model_type` and reconstruct its nested configuration explicitly, rather than letting a plain nested dictionary masquerade as a config object.

Keep experiment paths, data mixture weights, optimizer and AURC training settings outside this model configuration. The index head does not have globally named labels: option meaning changes on every request. A serialized `num_labels` or `max_options` describes capacity, not a fixed semantic class vocabulary.

Verified API distinctions in Transformers 4.57.6:

- `PretrainedConfig.save_pretrained(save_directory, ...)` serializes configuration; it does not save model weights or make the enclosing module compatible.
- `AutoConfig.register(model_type, config, exist_ok=False)` registers a config class in the running Python process and checks the model-type name.
- `AutoModel.register(config_class, model_class, exist_ok=False)` registers the corresponding model implementation. For a standard portable checkpoint, implement `DecisionModel(PreTrainedModel)` with a matching `config_class`, constructor, forward contract and weight serialization.
- `register_for_auto_class(...)` concerns exporting custom implementation code for Auto-class loading; it is different from runtime registration. Defer Hub packaging/remote-code behavior until distribution is required.

The current checkpoint (`encoder/`, `tokenizer/`, `head.pt`, `model.json`) is already sufficient for training and evaluation in this package. Merely writing a `DecisionConfig` would introduce another representation without replacing that format. Change configuration and checkpoint contracts in separate commits.

## Acceptance checks and ordering

1. Frozen loader: record counts and hashes; preserve official roles, options and targets; fail on unsupported semantics or accidental repartitioning.
2. Configuration migration: exercise an existing flat YAML/CLI invocation and its nested equivalent, compare resolved values, check a CLI override and rejection of unknown/ambiguous keys. A small smoke invocation is sufficient; no new unit-test framework.
3. Baseline execution: load real frozen data, complete the existing tiny CPU smoke path, then perform a bounded GPU pilot. No synthetic additions are needed.
4. Optional model packaging: only when implementing it, compare logits and masks before/after standard save/load using a tiny random encoder and verify Auto-class loading in a fresh process. Retain an explicit loader for old checkpoints.
5. Source mixtures and other extensions follow the matched-data CE/AURC baseline.

## Verified implementation sources

Local API inspection used the existing trainer smoke environment at `jev-modernbert-training/work/venv/Lib/site-packages/transformers/`; no downloads, model inference or training were needed. Relevant upstream files at the matching release:

- [`configuration_utils.py`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/configuration_utils.py): `PretrainedConfig`, `save_pretrained`, `to_dict`, `register_for_auto_class`.
- [`configuration_auto.py`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/configuration_auto.py): `AutoConfig.for_model`, `from_pretrained`, `register`.
- [`auto_factory.py`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/auto_factory.py): model registration and loading dispatch.
- [`modeling_utils.py`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py): `PreTrainedModel`, configuration ownership and model serialization.

The signatures and current trainer behavior were inspected locally; custom-wrapper portability has **not** been implemented or verified.
