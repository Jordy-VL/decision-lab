# Verification record

2026-09-22. These checks establish implementation readiness, not model quality.

- uv lock resolved 88 packages. Generator and report extras installed in the integrated Linux workspace (Python 3.14.4). Train-extra dependency resolution passed a dry run; the Linux GPU runtime was not installed.
- Integrated offline generation emitted six validated records, two of each primitive. The copied trainer accepted the resulting trainer.jsonl directly.
- The generator's implementation task passed 22 tests, checked six exports against Kev commit 90990a5fac2995b9faa3190f7d437e84f2067768, and sampled seven real HF texts across four domains. Details are in its examples/verification.json and SOURCES.md.
- The integrated trainer passed a CPU smoke run with a randomly initialized two-layer ModernBERT (hidden size 32), tiny local tokenizer, PyTorch 2.14.0+cpu and Transformers 4.57.6. Checks covered forward/backward, unused-slot masks, harmonic loss arithmetic, exact ties, checkpoint/tokenizer roundtrip, partial microbatch accumulation, both matched continuation arms, evaluation/prediction, overflow rejection and config overrides. No trainer test suite was added.
- Both smoke arms were evaluated on distinct fixture validation/test groups. The reporting script generated raw and temperature-scaled plots plus JSON/CSV summaries. The PNG was visually inspected. All such plots are prominently labeled illustrative. Three test rows and one calibration row provide no statistical evidence; min-accepted was lowered to 1 only for this smoke.

Local untracked artifacts live in runs/offline, runs/decision-smoke-wu63c9sy and runs/integrated-comparison. The report's calibration/test overlap guard and paired metadata requirements prevent accidental comparison of mismatched files; they do not replace a full provenance audit.

No paid teacher call, pretrained ModernBERT fine-tuning, full dataset benchmark, GPU measurement, or hypothesis confirmation has occurred. Real dataset converters, document token audits, robust uncertainty intervals and later multimodal/RL experiments remain in the experiment register. The code is local and unpublished.

- Direct integration also trained the six newly generated fixtures for two CPU updates with the AURC blend, using the tiny random checkpoint. This verifies the generator-to-training path only.
