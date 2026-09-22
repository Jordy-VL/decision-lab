# Jev synthetic decision data

A small, relocatable Python package for grounded **text** examples: choice, boolean/noul and ordinal score. This is additional data for the ModernBERT experiment, not a replacement for SST-5 and a small Yelp sample. It does not train, deploy, provision cloud resources, generate images, or call a model by default.

## Install and run offline

Python 3.11+ and uv:

```sh
uv sync --extra test
uv run pytest
uv run jev-data generate --config configs/pilot.json --out runs/offline
uv run jev-data validate runs/offline/canonical.jsonl
uv run jev-data schemas schemas
```

Alternatively `python -m pip install -e '.[test]'`. The package contains no absolute-path assumptions. `examples/mock/` contains a clearly labeled deterministic fixture run, **not teacher-generated corpus data**. Its domain/function fields exercise scheduling only; they are not evidence of domain coverage. `examples/motivating-examples.md` offers human-authored task designs for review.

## Small corpus mixture

```sh
uv run jev-data sample-mixture --registry configs/mixture.json --out runs/sources.jsonl
# Or sample the configurable single corpus in pilot.json:
uv run jev-data sample --config configs/pilot.json --out runs/wiki-sources.jsonl
uv run jev-data estimate --config configs/pilot.json --sources runs/sources.jsonl
```

The registry pins finance (EDGAR), legal (LEDGAR), everyday (DialogSum), and general (WikiText-2 raw) sources. Its per-source `max_documents` values are quotas; shortfalls are reported, not filled by huge downloads. Only raw input text is used, even where a source also has annotations. Read `SOURCES.md` for verified schemas, licensing and limitations. `examples/hf-sources.report.json` records the actual small connectivity check; `examples/hf-sources.jsonl` preserves the selected text and source metadata. No model processed those samples.

The single-corpus `sample` command uses the corpus embedded in `pilot.json`; for that route, change domain quotas to general-compatible tasks before live generation. The default live domain quotas are intended for the mixture command.

Streaming uses a bounded row prefix and deterministic hash-ranking within that prefix. It is **not** a uniform sample of the whole corpus. Parquet readers may fetch a whole row group, and CSV readers may buffer data; `scan_rows` limits processed rows, not exact network bytes. No whole-corpus download or legacy remote loading script is requested. Length bounds reject entire long texts rather than silently truncating them. Language detection is deterministic but heuristic; basic alphabetic/repetition filters and normalized exact dedup are not comprehensive quality checks.

WikiText rows are lines; complete articles are assembled before sampling. The last incomplete article in the bounded prefix is discarded. Other sources use original IDs when available, otherwise pinned row IDs. All questions from a source keep its group and split. Official splits are preserved; test generation is disabled. For a genuinely unsplit dataset set `official_splits=false`: one seeded group hash assigns 80/10/10 train/validation/test; no cross-validation. Generation refuses test records. Keep calibration/development allocation fixed downstream. Exact dedup runs across selected sources; near duplicates and undisclosed original parent documents remain audit limitations. Never supply Decision Index frozen evaluation data; known Kev-suite/Decision-Index dataset IDs are blocked.

## Live pilot, only when configured locally

1. Configure `JEV_BASE_URL`, `JEV_MODEL`, and `JEV_API_KEY` in the process environment. An ignored `.env` can be used with `uv run --env-file .env ...`; this package does not auto-load secrets. Never paste keys into a task or commit them.
2. Copy `configs/pilot.json`; set `live_enabled=true`, actual input/output USD-per-million rates, a small budget and request limits. Unknown rates remain unknown and block live calls. Confirm the provider's price and credit applicability independently.
3. Review the sampled sources and quotas. Run explicitly:

```sh
uv run --env-file .env jev-data generate --config configs/my-pilot.json \
  --sources runs/sources.jsonl --out runs/live-pilot --live
```

`base_url` includes the API prefix, typically `/v1`; the client appends `/chat/completions`. It sends role-based `messages`, **not** legacy text Completions (`/completions` with `prompt`). Nebius-hosted and self-hosted vLLM endpoints are interchangeable when they implement this interface. A VLM served by vLLM can be the teacher while receiving only text. No model name is assumed or verified from a suggested name; use the exact endpoint-supported ID. A Nebius Cloud credit balance does not establish eligibility for a particular inference service, and $100 is not a spending authorization.

The defaults are a six-question pilot, two concurrent calls, two transport attempts, one generation round, 1,800 output tokens per request, 12 total attempts, 21,600 total reserved output tokens and a $1 configured budget. These limits include checker calls and failed retries. `max_source_documents` and `max_options` bound inputs and answer capacity. Quotas are configurable for primitive, domain and question function; totals must agree. Domain-assigned sources are preferred, and finance/legal shortfalls cannot silently use unrelated general text. A rejection leaves a reported shortfall; no unbounded refill loop. The current quota pairing is a small pilot, not a balanced factorial task suite.

`response_mode=json_schema` sends a strict JSON schema; `json_object` and `plain` are alternatives. On a provider rejection that explicitly mentions `response_format`, an enabled fallback uses plain JSON and the same local validator. Every fallback consumes another attempt. Local validation always applies. Transport is async, bounded, timeout-controlled and uses bounded exponential backoff. Configure the endpoint's context limit and output-token parameter compatibility before paid use; the request presently uses `max_tokens`.

## Grounding and output contracts

Canonical JSONL is validated against `schemas/canonical.schema.json` (generated from Pydantic). Each record has `id`, nested `source`, `question`, `label_provenance`, `verification`, and `checker_agrees`. Source includes original text, dataset/config/revision/split/source/group IDs, text SHA-256, license and provenance URL. Question includes exact evidence offsets, fact, instructions, ordered options, zero-based `target_index`, rationale, type, domain/function, rubric and numeric ordinal values.

Every evidence span is checked against the exact source. This mechanically validates quotation integrity, **not** semantic support. Duplicate options are rejected using Unicode/case/punctuation normalization and a few obvious aliases; semantic equivalence still needs review. Negative booleans require a contradiction relation and evidence, not a missing mention. Ordinal values must strictly increase, every level needs a rubric, and the full rubric must appear in the instructions. Operational correctness of the rubric remains a semantic review task.

Live labels are `teacher-proposed`. The independent check option withholds the proposed target, rationale, fact and evidence and asks for an independent answer; disagreement or an unsupported answer rejects the item. Agreement is not proof and does not upgrade provenance. No teacher confidence is generated or used as a calibrated target. `annotation-backed` is available for separately validated annotated imports, not assigned by this generator.

`trainer.jsonl` is the sibling trainer contract:

```json
{"id":"example-1","group_id":"fixture/archive","source":"offline-fixture","split":"train","state":"The archive opened in 1998.","question":"When did the archive open?","options":["1998","2008"],"target_index":0,"type":"categorical"}
```

Types are `categorical`, `boolean`, `ordinal`. Boolean options are exactly `["false","true"]`. Ordinal rows additionally have `values`, aligned with ordered options. There are no special model tokens or option tags. The training architecture is independent of this format. A future predictor must return a zero-based index and a full normalized probability vector; `models.prediction` validates that interface and calculates ordinal expectation using explicit values.

`kev.jsonl` exports labeled System One requests: choice criteria map `o0`, `o1`, ... to options and the label is a key; noul label is a JSON boolean; score label is the ordered level index. `_meta` retains record/group/split IDs. Numeric values and rubrics appear in score descriptions, while canonical records preserve exact original structures. Export revalidates the canonical record and the Kev label mapping. `scripts/verify_kev.py --kev-root ... --data ...` checks against the real upstream `split_data.check_record` and `kev.api.SystemOneRequest/to_record`. No training model dependencies are needed for that check.

## Resuming and accounting

Each output directory is a run identity (configuration, source hashes, mode, endpoint/model and package version). Changing it requires a new directory. Work IDs are deterministic; accepted IDs are skipped on resume. JSONL checkpoints are rewritten atomically with fsync under an exclusive run lock, safe for bounded pilots. A stale `.lock` after a hard crash must be removed only after confirming no process still uses the run. Filesystem corruption fails closed.

Before each network attempt a reservation is persisted. Input reservation uses UTF-8 request bytes plus overhead, and output reservation uses the configured maximum. Reported usage reconciles successful attempts. Failed, interrupted or usage-less calls retain their pessimistic reservations. The ledger survives restart and counts attempts globally. `requests.jsonl` records endpoint/model, request settings and hashes, usage, finish reasons and sanitized error types/status codes; it never logs the API key, HTTP error body or raw exception text. Full prompts can be reconstructed from source plus package prompt code; raw provider responses are represented by hashes and accepted validated records.

This is not an exact billing cap: provider tokenization, hidden billing or inaccurate usage can exceed estimates. A crash between provider completion and local persistence can lead to a later retry; endpoint-independent exactly-once billing is impossible. Retries and failures may cost money. Limits are local to one run directory; starting a new run starts a new ledger. Do not run multiple independent paid pilots assuming a shared cap.

## Review and research boundaries

Audit a small pilot before scaling: label/evidence correctness, numeric rubric applicability, negative support, domain/function and label balance, source-length distribution using the actual ModernBERT tokenizer, duplicate/near-duplicate leakage, and source licenses/attribution. Test an options-only baseline, shuffled-context and option-permutation controls; high options-only accuracy can expose length/wording shortcuts. Count filtering and rejection shortfalls rather than claiming coverage of every possible task. No tokenizer fit or quality success is claimed from character bounds alone.

The [AsymptoticAURC repository](https://github.com/han678/AsymptoticAURC) informed selective-prediction context only. No training or AURC experiment is implemented here; confidence ranking and probability calibration must be evaluated separately. Documents/images and DUDE-derived choices remain later work. Original RVL-CDIP remains excluded; RVL-CDIP-N is a later allowed candidate. See the parent roadmap for the accepted experiment protocol.
