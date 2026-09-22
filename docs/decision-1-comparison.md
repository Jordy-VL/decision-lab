# Decision 1.0 — deferred external comparison

Reviewed September 22, 2026. User requested retaining these findings for a later comparison. This does not change the first ModernBERT CE versus CE+AURC experiment or authorize an additional GPU run.

## Sources and candidates

[Release announcement](https://vllm-sr.ai/blog/decision-models/) describes six models with the same state/questions/criteria interface. Kai and Lex use Vela-based bidirectional encoder paths with separate typed readouts. Eos, Sol, Nox and Lux use Qwen3.5 hybrid text backbones and a shared candidate head. All candidates for a question are scored together; independent questions are batched.

| Candidate | Published size tier | Complete input budget | Potential role |
|---|---:|---:|---|
| Kai / Lex | 0.6B | 1,024 tokens | Typed encoder architecture comparison; Lex specializes in operational workflows |
| Eos | 0.8B | 16,384 tokens | Preferred first external candidate within our sub-1B budget |
| Lux | 9B | 16,384 tokens | Methodology reference, outside our initial size/compute budget |

Size tiers are release names, not independently counted deployed parameters. The announcement says native vLLM Semantic Router integration is planned; released weights/local inference do not imply an already deployed hosted API.

## Evidence and interpretation

The [Eos card](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) reports 3,766 scored questions across a selected 54-task regression suite. Its weighted overall score is 61.89 versus Kev-0.8B at 58.28. Transfer reverses that ordering: 52.01 versus 61.19. These are publisher measurements, not our reproduced results, and not scores on the full Decision Index. Fixed category weights are 30/25/15/15/15 percent for decisions/composition/reading/inference/transfer. Do not interpret the aggregate lead as universal superiority or proof of unseen-domain calibration.

[Lux methods](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/main/METHODS.md) document full-parameter adaptation on 24,000 examples, composition replay and KL retention against an initial model, separate checkpoint selection, and a positive temperature fitted on 1,600 independent calibration examples. This is useful protocol guidance, not evidence that RL is necessary or that probabilities transfer without calibration.

[Kai fine-tuning](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/main/FINETUNING.md) supplies a CLI with hard/soft targets, split-overlap checks, checkpoint selection and optimizer/RNG resume. The example objective is CE plus 0.1 ranked probability score for ordinal questions. Some embedding parameters remain frozen. RPS is an interesting ordinal control; it is not an AURC objective.

## Minimal later comparison

1. Pin Eos code, weights and benchmark revisions; inspect its inference path and data attribution before loading it.
2. Adapt our fixed evaluation records to its request format, preserving option order, labels and complete evidence. Reject or explicitly report overflow rather than silently truncating. Keep selection/calibration/test roles distinct.
3. First run a bounded development-only compatibility and timing sample. Measure actual memory, latency and total inference cost before expanding. No current budget allocation to this comparison.
4. Report accuracy, raw and source-calibrated probability quality, risk–coverage, coverage at validation-selected risk thresholds and their transfer to untouched domains. Keep retrospective test frontiers separately labeled. Align confidence and tie conventions before comparing AURC.
5. Use released Eos as an external-system comparison, not a causal loss ablation: its backbone, training mixture and adaptation differ. Any later matched-data training comparison needs its own declared protocol.

**Hypothesis:** our AURC continuation improves selective utility and frozen-threshold transfer relative to CE; the external baseline establishes how competitive that utility is. If the external system wins, inspect task/domain/capacity differences before changing the objective. If our method wins only after target-domain calibration, narrow the transfer claim. If inconclusive, retain the simpler baseline and report uncertainty.

Reuse candidates: typed data adapters, split/admission checks, ordinal RPS and calibration separation. Avoid importing the multi-path architecture into the initial single-head experiment.
