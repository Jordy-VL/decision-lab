# Run inventory and next experiment plan — 2026-09-30

## Resolved access and retrieved snapshot

Password-authenticated SSH succeeded later on September 30. Remote checkout
is also `1e7a5a4`. Retrieved metadata, configs, Trainer histories, reports and
prediction JSONL into the ignored `runs/remote-review-20260930/` directory;
existing local runs were not overwritten. No checkpoint weights were copied.
The 52,825,515-byte archive has SHA256
`253daca59a936056d90a8c6f5de8234516b67a2b267fb197b0edfedd48337c2c`.
Its retrieval manifest records 21 completed training runs, 2 failed training
runs and 11 evaluation metadata records. These are saved states, not a live
GPU allocation check.

All five fixed-slot/candidate-head ten-epoch linear runs are marked completed.
Their metadata and saved predictions now support the follow-up analysis.
The completed fixed-slot plot now contains 465 saved evaluations (155 per arm),
replacing the earlier partial 428-evaluation snapshot. See the
[remote findings](remote-run-findings-20260930.md) for matching and metrics.

## Earlier local-only inventory

Initially inspected local artifacts on main at `1e7a5a4`. Both Windows and WSL
`ssh -o BatchMode=yes -o ConnectTimeout=12 uab-gpu` attempts failed during
SSH banner exchange (the Windows attempt used 15 seconds). At that point remote
completion was unverified. The successful authenticated retrieval above
supersedes that access blocker; current allocation/GPU capacity remains unchecked.

## Local evidence

| Run | Available evidence | Limitation |
| --- | --- | --- |
| x2-aurc-only-5epoch | metadata says completed, seed 17, code 1087e277; exported checkpoint; 5 epochs, 6716.9 seconds | No local prediction JSONL or Trainer state |
| x2-ce-5epoch | Exported checkpoint | No local metadata, prediction JSONL or Trainer state |
| x2-ce-aurc-mix-5epoch | Directory exists | No exported checkpoint, metadata, prediction JSONL or Trainer state |
| Linear 10-epoch arms | Tracked CSV/PNG development curves | No corresponding local run directories; curves alone do not verify completion |

The historical run ledger is dated September 23 and does not inventory the
newer sweep. Its old `running` entry is not a current cluster observation.
Do not overwrite it with inferred completion. Historical corrected AURC
continuation predictions and tiny-model smoke predictions also exist, but
are not substitutes for the matched five-epoch suite.

## Experiment sequence

1. Inventory remote `metadata.json`, resolved configs, Trainer state,
   best/final checkpoint directories and development/calibration predictions.
   Record code/model/data hashes and actual selection metric for every arm.
   Confirm completion before using the archiving/rerun job; do not infer it
   from the last point in a curve. Synchronize missing small artifacts first.
2. Keep historical NLL-selected results under their original labels. New
   matched runs use development AUGRC, lower is better, with AURC, NLL and
   Brier as diagnostics. Never silently compare different checkpoint rules
   as if only the training loss changed.
3. Fit one positive scalar temperature per arm using calibration NLL only.
   Freeze temperatures and whole-tie thresholds at 1%, 5% and 10% empirical
   selective risk (minimum 30 accepted calibration examples), then apply
   unchanged to test. Save raw/calibrated probabilities, parameters, counts,
   NLL, Brier, ECE, AURC, AUGRC and risk-coverage. Report zero-acceptance cases
   explicitly. `scripts/compare_runs.py` is the existing temperature baseline;
   it still needs AUGRC and richer counts before it meets the full protocol.
4. Prespecify isotonic/spline estimands and regularization before fitting.
   A scalar correctness calibrator cannot provide multiclass NLL/Brier;
   report its confidence metrics separately from full-distribution methods.
   Fit/tune only within calibration and never select a method on test.
   The repository does not yet specify a reproducible spline method.
5. Repeat fixed-slot CE/AURC-only/mix at five epochs with seed 29, identical
   frozen data/model revision, batch/update budget, optimizer and schedule.
   Use separate outputs, for example `runs/x2-ce-5epoch-seed29-augrc-selection`.
   A matched seed-17 AUGRC-selected comparison is also needed if only the
   historical NLL-selected seed-17 results are available; changing both seed
   and selector cannot isolate seed variability.
6. For V1/V2, first compare CE under the same seed, optimizer, split and
   update budget using the audited marker implementation. Record candidate
   marker/tokenizer revision, parameter count, sequence lengths, option-count
   slices, permutation diagnostics and end-to-end latency on fixed requests.
   Existing ten-epoch linear candidate-mask configs are a separate schedule
   comparison, not a matched substitute for a five-epoch cosine arm.
7. Only after the CE architecture comparison is valid, compare matched loss
   arms across heads. Retain all V1 artifacts and original result labels.

No new training or model inference was launched during this review. Access to
remote artifacts is restored. Historical saved test predictions can be used
for the prespecified temperature analysis without selecting on test outcomes;
isotonic/spline specification and fresh matched replications remain pending.
