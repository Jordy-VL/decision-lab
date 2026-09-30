# Five-epoch matched replication — 2026-09-30

Prepared on main; this document does not record a launch or a result.

September 30 follow-up: the user authorized GPUs 5, 6 and 7. GPU 6 was busy
and left untouched; the bounded [candidate diagnostic](candidate-diagnosis-20260930.md)
completed on GPU 5. The current runs filesystem still has about 100 GB free,
below this six-run queue's 116 GiB preflight. This replication batch remains
unlaunched pending sufficient durable output storage; GPU authorization is
no longer the blocker.

Freeze six fresh fixed-slot runs: CE, AURC-only and CE+AURC (lambda 0, 1,
0.5), each at seeds **17 and 29**, for **five epochs**. These were the
previously declared duration and replication seed; no test-driven schedule
search is introduced. Five epochs follows the existing pilot comparison and
is not a new independently selected optimum. Seed 17 is rerun to provide a
matched current-code reference, since the historical sweep selected on NLL.

Use the original five-epoch configs unchanged in scientific settings except
for seed and the current Trainer's development-AUGRC selector: pinned
ModernBERT-large revision, Kev decision-v7 partitions, fixed-slot head,
2048 tokens, 128 outputs, batch 4, accumulation 1, learning rate 2e-5,
weight decay 0.01, warmup 5%, cosine_with_min_lr scheduler, and save/evaluate
every 250 steps, retaining two Trainer checkpoints per run. Configs explicitly state the existing default scheduler,
head and AURC-family loss; CE is its lambda-zero case, preserving the original
resolved configuration. The code defines the scheduler's minimum learning
rate. The source hashes in each launch bundle freeze that implementation.
There is no configurable selector key: hf_trainer.py uses
metric_for_best_model="augrc" with lower being better.

New configs and outputs are named
`x2-{ce,aurc-only,ce-aurc-mix}-5epoch-seed{17,29}-augrc-selection`.
They do not reuse historical outputs. No checkpoint or resume path is set;
every arm starts from the same pinned pretrained model with a fresh optimizer.
Record the current code revision and dirty-source hashes, rather than claiming
byte-identical reproduction of the historical code. All six runs should use
one frozen source/data snapshot.

## Launch after allocation

From the repository root, inspect placement without starting training:

```sh
bash jobs/replicate-five-epoch.sh --gpus 0 --dry-run
```

Supply only GPU indices assigned to this job. One GPU queues all six runs
sequentially; multiple explicit GPU indices assign runs round-robin, with one
training process per GPU. At most six runs and six GPUs are accepted. The
launcher checks fresh outputs and required splits, then uses the already
prepared environment with `uv run --no-sync`; set `UV` if needed. It loads the
project `.env` if present, without copying its contents into provenance.

For a durable allocated session, use a scheduler job or run the coordinator
under nohup, choosing a fresh coordinator log filename:

```sh
nohup bash jobs/replicate-five-epoch.sh --gpus 0 > runs/replication-coordinator-20260930.log 2>&1 < /dev/null &
```

The launcher requires 16 GiB per planned run plus a 20 GiB reserve (116 GiB
for six runs) on the runs filesystem. This is a conservative preflight based
on prior five-epoch output sizes, not a guaranteed peak bound; other jobs can
consume space after the check. It refuses insufficient capacity without deleting
anything. Use approved durable storage with sufficient capacity before launching.

The coordinator holds an exclusive lock and records its PID, worker/training
PIDs, Git revision/status, source/config/data SHA-256 hashes, logs, timestamps
and exit codes under a fresh `runs/replication-five-epoch-TIMESTAMP-PID/`
directory. A failing run stops its GPU queue; other queues finish independently.
Every queued start rechecks source/data hashes and output absence. It never
archives, deletes or resumes outputs. Investigate failure and prepare an
explicit continuation before retrying: the fresh-output guard intentionally
rejects an existing partial run. A dry run does not validate GPU allocation,
CUDA availability, model cache or the training environment.

## Analysis fixed before these runs

Select checkpoints on development AUGRC, report AURC secondarily, and retain
NLL, Brier and accuracy diagnostics. Fit temperature/isotonic/spline calibration
only on calibration data and apply frozen mappings to test. Compare matched
arms within each seed, with raw and calibrated risk-coverage and coverage at
prespecified selective-risk levels (including 5%). Keep seed-specific results
visible; two seeds are limited replication, not publication-strength certainty.
Do not tune epochs, loss weights, schedules, thresholds or calibration methods
on these test outcomes. Candidate-head architecture comparisons are a separate
experiment, not part of these six runs.
