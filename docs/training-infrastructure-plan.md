# Training experiments and infrastructure plan

Updated 2026-09-23. This plan adapts the explicit run recipes and controlled ablations in the [DistilDoc classifier README](https://github.com/Jordy-VL/DistilDoc_ICDAR24/blob/main/src/clf/README.md) to decision-lab. It is not a proposal to reproduce DistilDoc's model or copy its dependency stack.

## Experiment structure

Treat each comparison as a named experiment group with a machine-readable matrix of arms. For X2, the group is three matched continuations from one frozen CE checkpoint:

| Arm | Objective | AURC coefficient |
|---|---|---:|
| `ce` | Cross-entropy | 0 |
| `aurc_only` | Detached harmonic rank-weighted CE surrogate only | 1 |
| `ce_aurc_mix` | CE blended with rank-weighted CE | 0.5 |

All arms share data and split manifests, model/checkpoint lineage, seed, sample order, microbatch composition, optimizer, schedule and update budget. Use a fresh optimizer for each arm. The surrogate is not direct optimization of discrete AURC. Add model, data, schedule or modality variants as separate experiment groups, changing one factor at a time where the research question requires attribution. Start with one seed; repeat only after the pilot is interpretable.

Each run gets a unique ID and immutable bundle containing the resolved config, code commit, model/data revisions and hashes, split manifest, parent checkpoint, objective settings, environment/package versions, hardware, update-level training log, checkpoints, and a concise conclusion/next action. Keep launch recipes in version control and avoid hand-edited differences between arms.

## Evaluation artifacts

For each arm and selected checkpoint, retain predictions for development (validation) and calibration partitions: raw logits, labels, probabilities, example/group IDs, source/type, and metric definitions. Save the exact checkpoint and config that produced each dump. Fit temperatures and operating thresholds on calibration only. Do not use final-test labels for checkpoint selection or tuning. The current CLI can produce evaluation prediction dumps, but does not run validation inside training; explicit post-checkpoint evaluation is the immediate workflow, with periodic evaluation as a later trainer improvement.

## Trainer requirements before scaling

1. Preserve exact recovery: model, optimizer, scheduler, RNG, epoch, batch position and data order.
2. Make the learning-rate schedule and evaluation/checkpoint cadence explicit in config; retain the best development checkpoint by a preregistered metric as well as the final-budget checkpoint.
3. Log structured per-step/epoch loss, LR, examples/updates, throughput, elapsed time and peak GPU memory. Record run completion/failure/cancellation accurately.
4. Save immutable development/calibration prediction bundles for every selected checkpoint. Keep final test access as a separate frozen evaluation action.
5. Provide one experiment-group launcher that renders and validates arm configs, checks shared fields, and submits each run without overwriting artifacts. A tracking service such as W&B is optional; the artifact bundle must remain complete without it.

## Training framework choice

For the current ModernBERT index-head experiment, use Hugging Face `Trainer` and `TrainingArguments`, with a small project-specific loss adapter and the regular Hugging Face `ModelOutput` interface. This keeps the AURC surrogate and option mask visible while relying on the standard optimizer/scheduler, logging, checkpoint, resume and prediction loops. `report_to="none"` keeps run tracking file-based. Trainer checkpoints contain the model, optimizer, scheduler, RNG and `TrainerState`; the project also writes deployable model folders and standalone `.npz` logits plus JSONL predictions.

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is a reasonable candidate for a future causal-LM fine-tuning or RL track, but is not the simpler fit for this experiment. Its documented focus is LLM post-training, and its sequence-classification path is presented as outcome/process reward modeling; that does not provide our option-masked index head or harmonic AURC objective as a ready YAML setting. Using it here would still require custom model/trainer integration and would add a separate dependency/runtime stack. Reconsider Axolotl if a later experiment changes the model objective to generative SFT/RL or a supported reward-model formulation. Sources: [Axolotl overview and requirements](https://github.com/axolotl-ai-cloud/axolotl), [reward-modeling guide](https://docs.axolotl.ai/docs/agents/reward_modelling.html), [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

## Backend decision: Modal or `ssh uab-gpu`

The backend is not yet selected. Modal has completed GPU runs but has function time limits and volume-commit behavior that the launcher must accommodate. The repository also has an SSH quickstart for the existing `uab-gpu` host, but authentication, scheduler/allocation process, current GPU capacity, storage durability and applicable usage constraints remain unverified. Preserve a backend-neutral run bundle and launcher boundary so the experiments do not depend on either provider.

Before choosing, run a bounded feasibility check on both paths: verify access and GPU type/memory, allocation and wall-time rules, dependency installation, dataset/checkpoint transfer, durable artifact retrieval, interruption/resume, and expected queue plus runtime cost. Do not launch the full three-arm study during this check. Choose the path that can run all matched arms with reliable recovery and accessible artifacts at acceptable operational cost; a hybrid is reasonable if one backend is better for profiling and another for longer runs.

## Next steps

1. Reconcile the existing AURC and CE run artifacts with their actual configs and parent checkpoint; label them preliminary because they are not the requested matched three-arm group.
2. Confirm the data/model/code pins and materialize the same frozen CE parent checkpoint locally/on the selected backend.
3. Generate development and calibration logits/probabilities for that parent checkpoint, establishing the paired evaluation baseline.
4. Compare Modal and `ssh uab-gpu` feasibility without launching a full run; record the decision and constraints.
5. Before any new research training, install the pinned stack and run a short end-to-end Trainer smoke run, verify Trainer checkpoint recovery, best/final model folders, and development/calibration `.npz` logits against JSONL IDs/labels.
6. Run the three matched arms, evaluate each on development/calibration, compare, and only then decide whether repetitions or external suites are warranted.
