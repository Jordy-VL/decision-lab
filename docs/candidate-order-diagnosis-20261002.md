# Candidate order diagnosis and permutation training — October 2

The frozen candidate head's poor performance is not explained by a padding or
unused-slot masking failure in these probes. It responds to candidate text and
tracks reordered answers more often than fixed-slot CE, but remains inaccurate
and order-sensitive. No evidence supports declaring the architecture repaired.

## Frozen development probe

Selected 12 categorical development rows per source by smallest SHA256(id),
180 total, independent of correctness. Compared the completed candidate CE and
fixed-slot CE ten-epoch linear-1e-5 checkpoints. No test data or parameter updates.

| Measurement | Candidate CE | Fixed-slot CE |
|---|---:|---:|
| Original correct | 73/180 | 118/180 |
| Reversed-option correct, labels remapped | 59/180 | 59/180 |
| Same answer after reversal | 67/180 | 17/180 |
| Same slot after reversal | 77/180 | 179/180 |
| Maximum single/batched probability difference | 0.00000978 | 0.00000447 |
| Probability on unused slots | 0 | 0 |

For the candidate model, changing the gather positions of unused slots left
probabilities exactly unchanged. Replacing every candidate text with identical
text changed predictions on all 12 Banking77 rows and all 12 DBpedia14 rows;
the candidate path is not simply disconnected from option text. This is an
input-ablation sensitivity check, not proof that each marker represents the
correct semantics. Banking77 remains 0/12 correct and has zero answer consistency
under reversal. Small source slices are diagnostic, not population estimates.

Raw per-example evidence and the training option/target distribution are in
[candidate-invariance-20261002.json](../reports/candidate-invariance-20261002.json).
Run with `jobs/quickstart.sh probe-candidate-invariance` on an authorized idle GPU;
the script refuses to overwrite its output. GPU 5 was used; GPUs 6 and 7 were
occupied and left untouched.

## Corrective training experiment implemented

`training.permute_training_options: true` enables deterministic categorical
option shuffling, keyed by seed, epoch and example ID. The target index follows
the same answer text. The dataset re-encodes the permuted row so candidate marker
positions are rebuilt, rather than shuffling already encoded tensors.

Only training rows are accepted by the permuted dataset. Boolean false/true and
ordinal numeric ordering stay unchanged. Development/calibration datasets remain
unmodified. The epoch callback reconstructs the epoch's permutations on Trainer
resume; permutation choices do not depend on call order or global RNG consumption.
Existing configs default to disabled, preserving historical experiment behavior.

Eighteen package tests pass, including label preservation, input immutability,
epoch determinism, nontraining rejection, candidate markers and remapped labels,
and an actual two-epoch Trainer run with unchanged development examples.
A 256-option fixed-slot forward and checkpoint round trip also passes.

Prepared fresh-output CE configs for both heads at capacities 128 and 256:
`packages/modernbert-decisions/configs/x2-{ce,candidate-masks-ce}-10epoch-linear-1e5-permuted-{128,256}.yaml`.
All four configs parse and validate. Full training has not been launched.
Compare 128-slot permutation against its matched baseline first; changing both
capacity and permutation at once would confound the result.

The training audit confirms maximum option count 77 (1,000 rows); there are no
training examples supervising fixed slots 77–255. A 256-slot fixed head therefore
needs high-cardinality training data before claiming useful 256-option capacity.
The candidate head shares a scalar scorer across options, so increasing its
capacity does not introduce separate output weights, but length and transfer
limits remain. The existing 128-slot checkpoints and official scores are intact.

## Bounded learning smoke

The train-only candidate diagnostic now accepts `--permute-options`. It changes
categorical order each optimizer update and evaluates five further orders of the
same eight training examples after 100 updates. This checks learning feasibility,
not held-out generalization; the full experiment permutes once per epoch.

Both matched runs completed on GPU 5 from the same pinned pretrained encoder,
seed 17, eight training examples and 100 updates. Identical initial loss was
1.61991. The original-order control fit all eight, but its performance degraded
when options moved. The permuted run had not fully fit the original eight, yet
performed better on new orders:

| Measurement | No permutation | Permuted training |
|---|---:|---:|
| Final original-order accuracy | 8/8 | 6/8 |
| Final original-order CE | 0.000444 | 0.815391 |
| Correct across five unseen orders | 24/40 | 29/40 |
| Accuracy per unseen order | .625/.625/.500/.625/.625 | .750/.750/.625/.750/.750 |

The forty evaluations are correlated views of eight training examples, not
forty independent held-out cases. Boolean/ordinal examples retain their original
orders in both runs. This supports testing permutation training at a larger
controlled budget, but does not establish generalization or a repaired head.
No model checkpoint was exported by these smoke runs.
[Full smoke evidence, hashes and update traces](../reports/candidate-permutation-smoke-20261002.json).

Next: compare original versus permuted candidate CE with the same training
budget and development checkpoint selection, and report a separately fixed
development permutation-consistency metric. Keep the 256-capacity variant
separate; it does not itself address position bias. Candidate pooling/marker
architecture changes should be separate ablations rather than bundled into this
training change. The current tests did not identify a masking implementation bug
that justifies replacing the marker strategy outright.

## Authorized full training launch — October 2, 15:41 CEST

The user requested training the 256-option configs. Launched fixed-slot CE first,
then candidate CE sequentially on GPU 5; GPUs 6/7 were occupied. Both runs use
seed 17, ten epochs, LR 1e-5, linear decay, batch four, development-AUGRC checkpoint
selection and categorical option permutation each epoch. Trainer also shuffles
example order. Original checkpoints are not initialization sources: both start
from the pinned pretrained encoder with a fresh head.

Remote queue: `runs/permuted-256-queue-20261002/`, coordinator PID 124224.
`status.json`, `launch.json`, `source-hashes.json`, `fixed.log`, `candidate.log`
record progress and provenance. The coordinator survives SSH disconnects, stops
on failure, and checks unchanged source/config hashes, idle GPU 5 and at least
80 GiB free before each arm. Initial free storage was 237 GB as reported by df.
Verified fixed-slot training advancing through step 22/38,940. Candidate training
is queued, not yet started. The run commands are
`jobs/quickstart.sh train-permuted-256-fixed` and
`jobs/quickstart.sh train-permuted-256-candidate`.

This supersedes the earlier 'not launched' status. The limitation on untrained
fixed slots 77–255 remains; no higher-cardinality data was added to this experiment.
