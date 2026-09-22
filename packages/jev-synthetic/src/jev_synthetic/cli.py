import argparse
import asyncio
import json
from pathlib import Path
from .config import Config, Corpus, load
from .corpus import sample
from .export import exports
from .models import Source, Record, Proposal, digest
from .pipeline import run
from .storage import atomic, read_rows, write_rows


def mock_sources():
    text = "The archive opened in 1998. It contains three collections of maps, letters, and photographs. The reading room opens at nine in the morning and closes at five in the afternoon. Visitors may read materials in the room but cannot borrow original documents."
    return [Source(dataset="offline-fixture", config=None, revision="1", source_split="train", split="train",
                   source_id="archive", group_id="offline-fixture/archive", text=text, text_sha256=digest(text),
                   license="CC0-1.0 (project-authored fixture)", provenance_url="project:mock_sources", mock=True)]


def main():
    parser = argparse.ArgumentParser(description="Grounded decision data generation; paid mode disabled by default")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config")
    init.add_argument("path", type=Path)
    schemas = sub.add_parser("schemas")
    schemas.add_argument("out", type=Path)
    estimate = sub.add_parser("estimate")
    estimate.add_argument("--config", required=True)
    estimate.add_argument("--sources", required=True, type=Path)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--config", required=True)
    sampling.add_argument("--out", required=True, type=Path)
    mix = sub.add_parser("sample-mixture")
    mix.add_argument("--registry", required=True)
    mix.add_argument("--out", required=True, type=Path)
    generate = sub.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--sources", type=Path)
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--live", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    export = sub.add_parser("export")
    export.add_argument("path", type=Path)
    export.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "init-config":
        atomic(args.path, Config().model_dump_json(indent=2))
    elif args.command == "schemas":
        for name, model in (("canonical", Record), ("proposal", Proposal), ("source", Source), ("config", Config)):
            atomic(args.out / f"{name}.schema.json", json.dumps(model.model_json_schema(), indent=2))
    elif args.command == "estimate":
        from .prompts import generation
        cfg = load(args.config)
        sources = [Source.model_validate(r) for r in read_rows(args.sources)]
        schema = Proposal.model_json_schema()
        largest = max((len(json.dumps(generation(s, "score", s.domain, "quantity_or_degree"), ensure_ascii=False).encode()) for s in sources), default=0)
        # Allow schema in both response_format and appended instruction, plus transport overhead.
        input_bound = largest + 2 * len(json.dumps(schema).encode()) + 4096
        attempts = min(cfg.max_requests, sum(cfg.primitive_quotas.values()) * cfg.rounds * cfg.max_attempts * (2 if cfg.independent_check else 1))
        known = cfg.input_usd_per_million is not None and cfg.output_usd_per_million is not None
        cost = attempts * (input_bound * cfg.input_usd_per_million + cfg.max_output_tokens * cfg.output_usd_per_million) / 1e6 if known else None
        print(json.dumps({"rates_known": known, "conservative_input_tokens_per_attempt": input_bound,
                          "max_attempts": attempts, "estimated_upper_cost_usd": cost, "configured_budget_usd": cfg.budget_usd,
                          "note": "Estimate only; failures may be billed and provider accounting may differ."}, indent=2))
    elif args.command == "sample":
        sources = sample(load(args.config).corpus)
        write_rows(args.out, [s.model_dump() for s in sources])
        print(json.dumps({"sampled": len(sources)}))
    elif args.command == "sample-mixture":
        registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
        sources, status = [], []
        for entry in registry["sources"]:
            if not entry["enabled"]:
                continue
            try:
                rows = sample(Corpus.model_validate(entry["corpus"]))
                sources.extend(rows)
                status.append({"name": entry["name"], "sampled": len(rows), "status": "ok"})
            except Exception as exc:
                status.append({"name": entry["name"], "status": "failed", "error_type": type(exc).__name__})
        # Exact normalized dedup across selected corpora happens again before generation.
        write_rows(args.out, [s.model_dump() for s in sources])
        atomic(args.out.with_suffix(".report.json"), json.dumps(status, indent=2))
        print(json.dumps(status))
    elif args.command == "generate":
        if args.live and args.sources is None:
            parser.error("--live requires --sources; configure credentials in the local environment")
        cfg = load(args.config)
        sources = [Source.model_validate(r) for r in read_rows(args.sources)] if args.sources else mock_sources()
        if not args.live and args.sources:
            parser.error("offline fixtures use no --sources; real sources require explicit --live")
        print(json.dumps(asyncio.run(run(cfg, sources, args.out, args.live)), indent=2))
    else:
        rows = [Record.model_validate(r).model_dump() for r in read_rows(args.path)]
        if args.command == "export":
            exports(args.out, rows)
        print(json.dumps({"valid_records": len(rows)}))


if __name__ == "__main__":
    main()
