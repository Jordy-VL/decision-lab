# Candidate-head diagnosis — 2026-09-30

This is an observational breakdown of existing development predictions, plus a
bounded train-only diagnostic completed on user-authorized GPU 5. It does not establish
why the candidate head learns poorly. No architecture was changed.

## Saved-development evidence

Input: `runs/remote-review-20260930/runs/`, current
`x2-{ce,candidate-masks-ce,candidate-masks-aurc-only}-10epoch-linear-1e5/`
`evaluations/best/development/predictions.jsonl`. All are development-AUGRC
selected checkpoints. Accuracy is exact `index == target_index`.

| Option count | Rows | Fixed CE | Candidate CE | Candidate AURC |
|---|---:|---:|---:|---:|
| 2 | 600 | .8250 | .6667 | .6400 |
| 3 | 152 | .7763 | .3947 | .3750 |
| 4 | 116 | .6897 | .4655 | .3362 |
| 5 | 252 | .5437 | .3214 | .2817 |
| 6 | 104 | .7692 | .3846 | .2596 |
| 7 | 12 | .0833 | .0833 | .0833 |
| 14 | 104 | .7500 | .3942 | .0865 |
| 15 | 12 | .0833 | .1667 | .0833 |
| 77 | 104 | .6346 | .0192 | .0192 |
| 78 | 12 | .0000 | .0000 | .0000 |

High-option banking77 is especially weak: 2/116 correct for both candidate
arms versus 66/116 for fixed CE. This does not explain the entire gap: binary
and low-option tasks also decline. Option counts are confounded with sources
and task semantics, and the 12-row groups are too small for strong conclusions.

| Source | Rows | Fixed CE | Candidate CE | Candidate AURC |
|---|---:|---:|---:|---:|
| agnews | 116 | .6724 | .4741 | .3448 |
| agnews_yn | 184 | .9457 | .7880 | .7826 |
| banking77 | 116 | .5690 | .0172 | .0172 |
| boolq | 80 | .8000 | .6250 | .6125 |
| dbpedia14 | 116 | .6810 | .3707 | .0862 |
| imdb | 80 | .9125 | .6375 | .4875 |
| mnli | 116 | .6379 | .3276 | .3276 |
| trec | 116 | .6983 | .3534 | .2414 |
| amazon | 80 | .5500 | .3750 | .3125 |
| sst5 | 80 | .5375 | .3000 | .2125 |
| yelp | 80 | .6125 | .3000 | .3125 |
| yelp_yn | 80 | .8875 | .8000 | .7125 |

Each of the four contrastive development sources has 24 rows. Candidate CE
gets exactly .5000 on all four, compared with fixed CE's 1.0000 age eligibility,
.9583 quantity limit, .7500 return window and 1.0000 spend threshold. Composition
sources have only 16 rows each and mixed outcomes; they do not support a general
claim that every source declines.

Using normalized position bucket `min(3, floor(4 * index / option_count))`:

| Bucket | Gold rows | Fixed CE predictions | Candidate CE predictions | Candidate AURC predictions |
|---|---:|---:|---:|---:|
| 0 | 630 | 606 | 719 | 733 |
| 1 | 212 | 229 | 194 | 220 |
| 2 | 489 | 512 | 405 | 318 |
| 3 | 137 | 121 | 150 | 197 |

Candidate CE accuracy conditional on gold bucket is .6127/.3255/.3620/.3577,
versus fixed CE .7492/.6934/.7301/.5839. Candidate models overpredict bucket zero,
but this is not a causal position-bias measurement: binary tasks occupy only
buckets zero and two and source mixes differ. A source-controlled training or
development permutation experiment would be needed to isolate order sensitivity.

Training history does not show that the candidate selector simply stopped
before learning: CE selects step 28,250 and AURC step 25,250 out of 38,940.
Candidate minimum NLL occurs earlier (11,250 and 15,500), and final-budget NLL
and AUGRC worsen. See [remote-run findings](remote-run-findings-20260930.md).

## Bounded pretrained diagnostic

`scripts/diagnose_candidate_head.py` defaults to the pinned pretrained encoder,
new candidate head/marker, eight training examples, seed 17, AdamW LR `1e-5`,
weight decay .01 and at most 100 updates. Each update accumulates microbatch-one
gradients over the same tiny set. Sources are traversed round-robin after seeded
shuffling. Input must explicitly contain only training rows; overlength skips
are recorded. No development, calibration or test rows enter this script.

Run from repository root within an allocated GPU environment:

```sh
CUDA_VISIBLE_DEVICES=5 jobs/quickstart.sh diagnose-candidate \
  --out runs/candidate-diagnostic-UNIQUE --examples 8 --steps 100 --seed 17
```

The JSON output records IDs, source, targets, option counts, marker positions,
data hash, code revision, GPU, initial/final fit, per-update marker/head gradient
norms and parameter-change norms, total gradient norm before clipping, and
evaluation-mode tiny-set fit every ten steps. It writes no model checkpoint and
refuses an occupied output directory. `--checkpoint` can inspect a saved
candidate model with the same train-only protocol; this is a distinct probe.

Interpretation must stay narrow: finite nonzero marker gradients and changed
marker weights rule out a completely disconnected marker embedding on these
examples. Falling loss/overfitting establishes optimization capacity on this
tiny set. Failure to overfit within 100 updates at the original small LR does
not prove an architecture bug. The earlier fast-tokenizer/tiny-encoder tests
are separate evidence.

## Completed GPU check and encoding preflight

The user authorized GPUs 5, 6 and 7 on September 30. GPU 6 was occupied and
left untouched; the diagnostic used an idle NVIDIA A40 on GPU 5. From remote
`/data/133-1/users/sbiswas/experiments/decision-lab` on `main`, first ran
`jobs/quickstart.sh check` successfully: 15,576 training rows. Then synchronized
the audited model guards, eight tests, diagnostic and quickstart entry point
after verifying the remote tracked files matched their expected base hashes.
All eight tests passed both locally and remotely. No changes were committed.

Diagnostic started at 18:43:22 UTC (20:43:22 Europe/Paris), completed 100 steps,
and wrote no checkpoint. Source hashes, training-data hash, pinned pretrained
revision, arguments and per-step gradients are recorded in the
[diagnostic report](../reports/candidate-diagnostic-20260930.json); the
[launch record](../reports/candidate-diagnostic-20260930-launch.json) records
the exact quickstart command and GPU. Retrieved evidence bundle SHA256:
`64fd140794b51c919578c18c083ce14504ea52a6b37e641ad55e89dcbd36faad`.

| Tiny training-set measurement | Initial | After 100 updates |
| --- | ---: | ---: |
| Accuracy | 0/8 | 8/8 |
| Mean cross-entropy | 1.6199097 | 0.00040149 |

No rows were excluded. The eight examples span eight sources and include one
77-option Banking77 example (1,133 tokens). Marker/head gradient norms were
nonzero; final parameter-change norms were 0.00182671 and 0.13172863 respectively.
The small sample is memorized, establishing gradient connectivity and tiny-set
optimization capacity. It does not establish generalization or explain the
full-run candidate gap; one Banking77 example does not resolve that source's
poor development performance. Follow-up should compare matched optimization
behavior and source-controlled option-order sensitivity on train/development,
without tuning on test.

The separate CPU-only [encoding preflight](../reports/encoding-preflight-20260930.json)
checked every train/development/calibration row with the saved fixed-slot and
candidate tokenizers. All 18,192 examples encode under both heads, with zero
marker mismatches, zero overlength failures and zero pairwise ID/group overlaps
between splits. Maximum tokens are 1,140/1,124/1,078 for fixed-slot and
1,217/1,202/1,155 for candidate train/development/calibration respectively.
The preflight JSON labels fixed-slot as `cls`, the alias passed to `encode`;
that function takes the unmarked fixed-slot path for this value. No test
predictions or test tuning were performed by these checks.
