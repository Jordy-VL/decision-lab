# Decision Index transfer diagnosis — October 1

The strongest observed failure is option-position dependence in both fixed-slot
AURC and CE checkpoints. We did not find an output-index mapping error in the
tested paths. This diagnosis does not establish the training cause or rule out
all adapter issues; it is an exploratory analysis after seeing benchmark results.

## Evidence

GPU 5 ran both completed 10-epoch linear-1e-5 checkpoints, without training or
changing the official run. Selected the 64 smallest SHA256(run_id) values for
each of BANKING77, Humicroedit, MMLU, WinoGrande, HellaSwag, When2Call and HoVer,
restricted to single-question requests and official non-excluded rows. Selection
did not use correctness or confidence. One When2Call request exceeded capacity.

The 128-example development sanity check gives 90/128 correct for AURC (70.3%)
and 94/128 for CE (73.4%). All 320 original AURC predictions in the first five
benchmark samples exactly reproduce the saved official evaluation. Four adapter
mapping/batching/capacity tests pass. These checks support successful checkpoint
loading and stable inference; they are not proof of general semantic correctness.

Reversing criterion insertion order preserves each key, description and gold
answer, while changing the slot assigned to that answer. A model following the
answer's meaning should retain its answer key. Observed counts out of 64:

| Benchmark | AURC same slot | AURC same answer | CE same slot | CE same answer |
|---|---:|---:|---:|---:|
| BANKING77 | 57 | 2 | 59 | 3 |
| Humicroedit | 64 | 0 | 64 | 0 |
| MMLU | 52 | 5 | 54 | 2 |
| WinoGrande | 51 | 13 | 60 | 4 |
| HellaSwag | 54 | 3 | 58 | 1 |
| HoVer | 63 | 1 | 64 | 0 |

Both models always selected slot 0 on the Humicroedit sample. AURC's mean
confidence there was 99.91%, despite only 37/64 correct before reversal.
The full saved run likewise selected slot 0 for all 2,628 Humicroedit requests.
Slot bias need not mean one constant slot across every task: MMLU and HellaSwag
use multiple slots but largely retain them when answer meanings move.

## Formatting check

Training uses `decisions.data.text`, which retains standard JSON spaces; the
Decision Index adapter uses compact JSON. The first five sample benchmarks have
no observed prediction or confidence change under training-style serialization.
For the structured-input samples:

| Model | Benchmark | Original correct | Training spaces correct | Changed answers |
|---|---|---:|---:|---:|
| AURC | When2Call | 26/63 | 25/63 | 4 |
| AURC | HoVer | 31/64 | 31/64 | 2 |
| CE | When2Call | 25/63 | 23/63 | 5 |
| CE | HoVer | 34/64 | 34/64 | 0 |

Spacing affects some predictions but does not explain or repair the collapse
on these samples. Original slot-to-key mapping follows criterion insertion order
and uses the matching probability index; reversing order shows that mapping
changes as intended. No adapter or checkpoint was changed in the official run.
The reported Decision Index remains 1.48/100.

## Interpretation and next experiment

The evidence supports poor semantic option tracking under this external suite,
with high confidence on weak predictions. CE shares the failure, so attributing
it specifically to AURC loss is unsupported. Development performance plus weak
external transfer is consistent with learned task/slot shortcuts, but proving
that requires training-data and controlled-training checks. This is a small CE
diagnostic, not a complete CE Decision Index comparison.

Next, use training/development data only to measure option-permutation consistency,
including remapped labels. Run a small controlled CE comparison with per-epoch
option permutations and a candidate-scoring head, then evaluate on a newly fixed
development diagnostic. Check training's source/slot/label distributions first.
Do not optimize these external benchmark samples and relabel them as untouched
test evidence. Calibration can reduce overconfidence but cannot repair wrong
answer ranking. Resolve semantic tracking before another full leaderboard run.

## Artifacts and reproduction

[Summary, sample IDs, checkpoint hashes and raw-artifact hashes](../reports/decision-index-transfer-diagnostic-20261001.json).
Detailed per-example records remain on the server under
`runs/decision-index-transfer-diagnostic-20261001{,-structured}.json`.
The reproducible script is `scripts/diagnose_decision_index_transfer.py`.
From the remote project root, with an authorized idle GPU:

```sh
HF_HOME=/home-local/sbiswas/.cache/huggingface-decision-lab CUDA_VISIBLE_DEVICES=5 \
  jobs/quickstart.sh diagnose-decision-index-transfer
HF_HOME=/home-local/sbiswas/.cache/huggingface-decision-lab CUDA_VISIBLE_DEVICES=5 \
  jobs/quickstart.sh diagnose-decision-index-transfer --structured-only
```

The script refuses to overwrite an existing diagnostic artifact. These commands
were completed on October 1; do not rerun over the existing artifacts.
