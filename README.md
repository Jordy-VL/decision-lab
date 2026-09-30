# Decision Lab

A text-first research workspace for a small Jev-like decision model. The central experiment asks whether AURC-weighted fine-tuning improves answered coverage at low error risk beyond cross-entropy and post-hoc calibration.

**Pilot training results exist; conclusions remain preliminary.** The matched duration sweep is recorded below, but calibration and independent-seed replication remain pending. Start the review with the [experiment register](docs/experiment-register.md): original sources, hypotheses, setup, smallest comparisons, and actions for both positive and negative outcomes.

## Experiment log — 2026-09-25

- Completed the matched v7 duration sweep: CE-only, AURC-only and CE+AURC
  mix at 2, 5 and 10 epochs, with raw frozen-test evaluation.
- The historical matched 5-epoch comparison is a single-seed pilot.
  September 30 reanalysis of the retrieved exports corrects the earlier raw-NLL
  claim: AURC-only raw test NLL is 1.27215 versus CE 1.24148 (lower is better).
  See the [raw and temperature-scaled analysis](docs/five-epoch-calibration-20260930.md);
  independent-seed replication remains pending.
- Checkpoint selection is now based on development AUGRC; NLL remains a
  diagnostic because it directly favors the CE objective. The historical
  NLL-selected results remain archived and labeled as such.
- Added configurable linear and cosine-with-minimum-learning-rate schedulers.
  The next run set is two prepared 10-epoch linear-decay probes—CE-only and
  AURC-only—with a `1e-5` learning rate and 10% warmup. Run them only when
  GPU capacity is available, and keep their outputs separate from the
  original sweep.

### Next todos

See the [September 30 run inventory and execution plan](docs/run-review-20260930.md)
for verified artifacts retrieved from the GPU server. Historical
logs below do not establish current GPU or run status.

1. Run and evaluate the two linear-decay probes, selecting checkpoints by
   development AUGRC (lower is better); retain AURC as a secondary diagnostic.
2. Fit calibration on calibration data only—temperature scaling first, then
   the declared isotonic and spline comparisons—and apply frozen mappings to
   test.
3. Report raw and calibrated NLL, Brier, ECE, AURC, risk-coverage and
   coverage at fixed selective-risk levels.
4. Repeat the matched 5-epoch CE/AURC-only/mix comparison with seed 29 before
   making a causal or publication-strength claim.
5. Finish the candidate-marker/shared-scorer X12 head, then compare its CE
   architecture against V1 before testing AURC effects across heads.

## Workspace

```text
packages/jev-synthetic/          HF text -> grounded questions -> trainer/Kev JSONL
packages/modernbert-decisions/   JSONL -> ModernBERT -> index and probabilities
scripts/compare_runs.py          paired predictions -> selective/calibration plots
docs/                           roadmap, protocol, experiment register, verification
uv.lock                         resolved workspace dependencies
```

One Git repository, two independently packaged modules; no Git submodules. Use Python 3.11+ and uv. Commands below run from the repository root. Use a fresh output path when repeating a run.

## Offline example

```sh
uv sync --extra generate
uv run --extra generate jev-data generate --config packages/jev-synthetic/configs/pilot.json --out runs/offline
uv run --extra generate jev-data validate runs/offline/canonical.jsonl
uv run --extra train decisions check --data runs/offline/trainer.jsonl
```

The default generator emits six explicit mock fixtures: two categorical, two Boolean, two ordinal. These are for interface inspection, not training evidence. `check` validates schema and splits without downloading a model. Selecting the train extra installs PyTorch; for both packages use `uv sync --extra all`. Linux's resolved PyTorch wheel includes CUDA dependencies; choose an appropriate PyTorch build for your server before a real run.

To sample real raw text without any model API calls:

```sh
uv run --extra generate jev-data sample-mixture --registry packages/jev-synthetic/configs/mixture.json --out runs/sources.jsonl
```

Pinned sources cover financial filings, legal clauses, everyday conversations and general articles. See [source revisions and terms](packages/jev-synthetic/SOURCES.md); the default mixture includes noncommercial material. Sampling is bounded and domain-stratified, not uniform over all Hugging Face datasets. The [generator README](packages/jev-synthetic/README.md) explains how to use the sampled sources for live generation.

## First training run

**Baseline data is now prepared and verified.** Use [the pinned Kev adapter and commands](docs/kev-baseline.md) for the first research run. It preserves Kev's official reproduction suite; our work is the format adapter. Custom synthetic mixtures remain a later experiment. The examples below are still only illustrative.

For server cloning, environment setup and the X2 prerequisites, follow the [GPU quickstart](docs/gpu-quickstart.md). Research runs use `uab-gpu`; obtain a scheduler allocation and record the selected GPU IDs in the run bundle.

For the cloud alternative, see the [Modal first-baseline setup and budget guide](docs/modal-first-baseline.md). Its current launcher predates the Trainer migration and must be adapted before reuse; choose the backend only after the feasibility check.

The old illustrative-data training examples have been retired; training now requires explicit train, development and calibration files. Follow the X2 redo plan and do not use the example JSONL as a benchmark or launch a new arm before the Trainer artifact/recovery smoke passes.

```sh
uv run --extra train decisions train --config packages/modernbert-decisions/configs/x2-ce.yaml
uv run --extra train decisions train --config packages/modernbert-decisions/configs/x2-aurc-only.yaml
uv run --extra train decisions train --config packages/modernbert-decisions/configs/x2-ce-aurc-mix.yaml
```

Before running, verify data paths, allocation and the Trainer artifact/recovery prerequisites. The current X2 configs start from the same pinned pretrained ModernBERT revision with fresh optimizers, as specified in the [experimental protocol](docs/experiments.md); historical CE-continuation runs are separate. Each records a resolved config, hashes, Trainer state/checkpoints and best/final development and calibration logits. Use fresh output directories and review the [run inventory](docs/run-review-20260930.md) before scheduling another arm.

The model fully fine-tunes ModernBERT-large with CLS pooling and one bounded index head. Options are ordinary numbered text, unused slots are masked, and predictions return zero-based indices and full probabilities. Boolean uses false/true. Ordinal examples include explicit ordered numeric anchors. Default capacity is 128 options; CLINC150 plus OOS requires at least 151. No separate confidence head or text generation.

## Compare predictions

Evaluate each arm separately on calibration and untouched test splits, saving `predictions.jsonl`. Then:

```sh
uv run --extra report python scripts/compare_runs.py --ce runs/ce-test/predictions.jsonl --aurc runs/aurc-test/predictions.jsonl --ce-calibration runs/ce-calibration/predictions.jsonl --aurc-calibration runs/aurc-calibration/predictions.jsonl --out runs/comparison
```

The report saves PNG/SVG risk-coverage, reliability and threshold plots, JSON metrics and CSV curve points. It compares raw and temperature-scaled variants, selecting thresholds on calibration only. If calibration files are omitted, only raw comparisons are made. Use `--illustrative` for fixture/random-model outputs. Counts and point estimates are provided; publication-quality uncertainty analysis is a later step.

## Live generation and scope

The configurable teacher uses OpenAI-compatible Chat Completions, served through Nebius or local vLLM. Put endpoint, model and API key in local environment variables. Explicit live enablement, configured rates and a small budget are required; no key is needed for the offline workflow. Evidence-span validation proves quote presence, not full semantic correctness. Audit generated labels before training. See [motivating examples](packages/jev-synthetic/examples/motivating-examples.md).

Text and CE versus AURC come first. Follow official splits or one fixed document-grouped split, with one pilot seed. SST-5/Yelp ordinals, DUDE-derived decisions and RVL-CDIP-N are in the plan; original RVL-CDIP is excluded. Vision, RL abstention, multi-turn rollouts and a possible numeric primitive are separate follow-ups.

See [roadmap](docs/roadmap.md), [experimental protocol](docs/experiments.md), [numeric assessment](docs/numeric-outputs.md), and [verification record](docs/verification.md). Repository is local; nothing has been published.
