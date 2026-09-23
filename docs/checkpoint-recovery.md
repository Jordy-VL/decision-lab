# Periodic recovery — September 23, 2026

Training now saves a recovery bundle every 250 optimizer updates by default (`training.save_every`). Each completed bundle contains encoder/tokenizer/head, AdamW state, CPU/CUDA random state, shuffle state/order, next batch position and cumulative epoch loss. Publication uses a temporary directory, a completed-bundle rename and an atomic latest pointer. Interrupted incomplete bundles are never selected automatically.

The Modal launcher commits each completed bundle to its persistent Volume before pruning older completed bundles; the newest two are retained. The final weights-only checkpoint remains at `train/checkpoint`. A process kill can still lose work since the last durable checkpoint, but no longer the entire epoch. Cancellation status is caught where Python cleanup can run; platform state is authoritative after an uncatchable termination.

Resume trusted recovery artifacts with `decisions train --config ... --resume /path/to/recovery/step-000250 --output NEW_EMPTY_DIRECTORY`. This differs from `--checkpoint`, which intentionally starts a fresh optimizer for matched continuation experiments. Resume rejects changed data bytes or training settings. Only use trusted local recovery files: optimizer/RNG restoration requires Python pickle deserialization. Exact reproducibility across different hardware/software is not promised.

A one-off tiny CPU verification exercised dropout, two epochs, accumulation and partial final windows. Interruption after update two followed by recovery produced bit-identical final weights and epoch losses to uninterrupted training; altered learning rate was rejected. This is a functional verification, not research evidence. No permanent unit-test suite was added.

The new launcher uses an explicit A100-40GB and submits a detached function call so the client can exit without waiting for training. It retains the 75-minute function ceiling, 55-minute training ceiling and 7.5-minute limit for each evaluation. No automatic retries and no test-data access. At launch, Modal's monthly billing summary reported $0.94 metered, offset by credits; the total authorized initial budget remains $10.
