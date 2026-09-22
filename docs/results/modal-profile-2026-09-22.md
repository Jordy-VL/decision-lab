# First Modal GPU profile

Status: passed. [Raw measurements](modal-profile-2026-09-22.json). [Modal run](https://modal.com/apps/jordy-vlan/main/ap-xqLBFwGmHPi6oq9PAnzbgQ).

ModernBERT-large at a pinned revision, full float32 CE updates, batch size 4, gradient checkpointing, AdamW. All 15,576 prepared training rows were tokenized without truncation; 128 seeded shuffled examples supplied 32 updates. A separate backward pass on the four longest examples checked peak padding memory. No test data was used; no trained checkpoint was retained.

| Measurement | Observed |
|---|---:|
| GPU | A100 SXM4 40GB |
| Mean update time, excluding first four updates | 0.374 s |
| Training throughput | 10.70 examples/s |
| Peak allocated / reserved memory | 7.41 / 8.28 GiB |
| GPU function body, including model loading/tokenization | 43.08 s |
| Projected epoch over 15,576 examples | 24.26 min |

The first update took 15.69 s, including compilation/startup effects. Other sampled updates vary with sequence length. This small sample is a planning estimate, not a confidence interval. The profiler mirrors the baseline's forward/loss/backward/clip/AdamW sequence, but does not benchmark checkpoint serialization, validation inference or training-loop host overhead. Gradient values and losses were finite for the 32 optimizer steps; no quality improvement is claimed.

At the previously checked approximate A100 + two CPU + 16 GiB RAM rate of $2.32/hour, projected epoch training compute is about $0.94. The measured function body alone corresponds to about $0.03; actual charged usage also includes startup, CPU preparation/build and storage, so this is not an invoice. Allow a conservative $2–3 envelope for an initial epoch plus setup/checkpoint/development checks, subject to runtime monitoring. The user's $10 cap remains the total first-attempt authorization, not an instruction to spend it all.

Next: implement a bounded full-training launcher with persistent checkpoint output and runtime accounting. Keep the agreed baseline hyperparameters fixed for the first run; available VRAM alone is not a reason to change batch size before the matched-loss experiment. The profiling app completed and no full training run has been launched.
