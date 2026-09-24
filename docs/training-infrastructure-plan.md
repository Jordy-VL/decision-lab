# Training experiments and infrastructure plan

Updated 2026-09-23. This plan adapts the explicit run recipes and controlled ablations in the [DistilDoc classifier README](https://github.com/Jordy-VL/DistilDoc_ICDAR24/blob/main/src/clf/README.md) to decision-lab. It is not a proposal to reproduce DistilDoc's model or copy its dependency stack.

## Experiment structure

Treat each comparison as a named experiment group with a machine-readable matrix of arms. For X2, the group is three matched fresh-start runs from the same pinned pretrained ModernBERT checkpoint:

| Arm | Objective | AURC coefficient |
|---|---|---:|
| `ce` | Cross-entropy | 0 |
| `aurc_only` | Detached harmonic rank-weighted CE surrogate only | 1 |
| `ce_aurc_mix` | CE blended with rank-weighted CE | 0.5 |

All arms share data and split manifests, pinned pretrained model revision, seed, sample order, microbatch composition, optimizer, schedule and update budget. Each arm initializes the encoder and head from the same pretrained model and uses a fresh optimizer. The surrogate is not direct optimization of discrete AURC. Add model, data, schedule or modality variants as separate experiment groups, changing one factor at a time where the research question requires attribution. Start with one seed; repeat only after the pilot is interpretable.

Each run gets a unique ID and immutable bundle containing the resolved config, code commit, model/data revisions and hashes, split manifest, parent checkpoint, objective settings, environment/package versions, hardware, update-level training log, checkpoints, and a concise conclusion/next action. Keep launch recipes in version control and avoid hand-edited differences between arms.

## Evaluation artifacts

For each arm and selected checkpoint, retain predictions for development (validation) and calibration partitions: raw logits, labels, probabilities, example/group IDs, source/type, and metric definitions. Save the exact checkpoint and config that produced each dump. Fit temperatures and operating thresholds on calibration only. Do not use final-test labels for checkpoint selection or tuning. The HF Trainer path now evaluates at the configured save/evaluation cadence, selects the best development checkpoint by NLL, retains the final-budget checkpoint, and exports development/calibration logits and JSONL predictions for both.

## Trainer requirements before scaling

1. Preserve exact recovery: model, optimizer, scheduler, RNG, epoch, batch position and data order.
2. Make the learning-rate schedule and evaluation/checkpoint cadence explicit in config; retain the best development checkpoint by a preregistered metric as well as the final-budget checkpoint.
3. Log structured per-step/epoch loss, LR, examples/updates, throughput, elapsed time and peak GPU memory. Record run completion/failure/cancellation accurately.
4. Save immutable development/calibration prediction bundles for every selected checkpoint. Keep final test access as a separate frozen evaluation action.
5. Provide one experiment-group launcher that renders and validates arm configs, checks shared fields, and submits each run without overwriting artifacts. A tracking service such as W&B is optional; the artifact bundle must remain complete without it.

## Training framework choice

For the current ModernBERT index-head experiment, use Hugging Face `Trainer` and `TrainingArguments`, with a small project-specific loss adapter and the regular Hugging Face `ModelOutput` interface. This keeps the AURC surrogate and option mask visible while relying on the standard optimizer/scheduler, logging, checkpoint, resume and prediction loops. `report_to="none"` keeps run tracking file-based. Trainer checkpoints contain the model, optimizer, scheduler, RNG and `TrainerState`; the project also writes deployable model folders and standalone `.npz` logits plus JSONL predictions.

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is a reasonable candidate for a future causal-LM fine-tuning or RL track, but is not the simpler fit for this experiment. Its documented focus is LLM post-training, and its sequence-classification path is presented as outcome/process reward modeling; that does not provide our option-masked index head or harmonic AURC objective as a ready YAML setting. Using it here would still require custom model/trainer integration and would add a separate dependency/runtime stack. Reconsider Axolotl if a later experiment changes the model objective to generative SFT/RL or a supported reward-model formulation. Sources: [Axolotl overview and requirements](https://github.com/axolotl-ai-cloud/axolotl), [reward-modeling guide](https://docs.axolotl.ai/docs/agents/reward_modelling.html), [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

## Backend decision: `ssh uab-gpu`

`uab-gpu` is the selected backend going forward. Use the scheduler/allocation process on that host, request the required GPUs explicitly, and preserve durable run bundles on its approved storage. Modal remains historical context for the preliminary baseline only; do not launch the primary X2 arms there.

Before choosing, run a bounded feasibility check on both paths: verify access and GPU type/memory, allocation and wall-time rules, dependency installation, dataset/checkpoint transfer, durable artifact retrieval, interruption/resume, and expected queue plus runtime cost. Do not launch the full three-arm study during this check. Choose the path that can run all matched arms with reliable recovery and accessible artifacts at acceptable operational cost; a hybrid is reasonable if one backend is better for profiling and another for longer runs.

## Durable artifacts: Hugging Face Hub

Use the Hub as a candidate account-owned artifact store, separate from the choice of compute backend. Recommended layout is a private model repo for deployable best/final checkpoints and a private dataset repo for immutable run bundles: resolved configs, code/data/model revisions and hashes, Trainer state/log history, evaluation reports, JSONL predictions, and raw development/calibration `.npz` logits. Upload only after a run completes (or at explicitly chosen recovery milestones), and record the Hub repo IDs plus commit revisions in the run ledger. Keep large public-source datasets out unless their terms permit redistribution; keep the artifact repos private by default, especially while predictions and benchmark labels are present.

The Hub supports private model/dataset repos, and its Xet storage backend is designed for large binary model/data files. Check account storage limits before transferring full optimizer/recovery checkpoints; the Hub's current storage page lists 100 GB private storage for free accounts. On CVC or another host, authenticate with a write-scoped HF token using the Hub CLI or `huggingface_hub`; store the token in that host's secret store, never in configs or Git. A later uploader can call `HfApi.upload_folder()` and pin each resulting commit SHA. Sources: [repo creation/private visibility](https://huggingface.co/docs/huggingface_hub/guides/repository), [Xet large-file storage](https://huggingface.co/docs/hub/en/xet/index), [storage limits](https://huggingface.co/docs/hub/storage-limits), [folder uploads](https://huggingface.co/docs/huggingface_hub/main/guides/upload).

No Hugging Face artifact repo has been created and no files have been uploaded. This is a proposed storage destination pending the user's preferred namespace and access policy.

## Next steps

1. **Complete (2026-09-23):** run the synthetic CPU Trainer smoke before further research training. It verified earlier-best selection, final-budget checkpoint parity with the last Trainer checkpoint, development/calibration `.npz` logits against JSONL IDs/labels and saved-best-model inference, and bitwise-equal final weights after a step-2 resume versus uninterrupted four-step training. See [SMOKE.md](../packages/modernbert-decisions/SMOKE.md). This does not verify GPU or CVC behavior.
2. Reconcile the existing AURC and CE run artifacts with their actual configs; label them preliminary because they are not the matched fresh-start three-arm group.
3. Confirm the data/model/code pins on `uab-gpu`, prepare the frozen train/development/calibration partitions, and verify the pinned pretrained ModernBERT revision can be loaded.
4. Run the three matched X2 arms from the fresh pretrained checkpoint, retaining development and calibration logits/probabilities for each arm.
5. Record the `uab-gpu` allocation, hardware, wall-time and durable-storage constraints in each run bundle.
6. Only after reconciliation and feasibility, run the three matched arms, evaluate each on development/calibration, compare, and then decide whether repetitions or external suites are warranted.
