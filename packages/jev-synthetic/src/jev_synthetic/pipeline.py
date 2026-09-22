import asyncio
import json
from collections import Counter
from pathlib import Path
from .client import Client, endpoint, LimitError
from .export import exports
from .models import Proposal, Record, Check, Source, digest, normalized
from .prompts import generation, checking
from .storage import Journal, atomic, lock, read_rows, write_rows


def expanded(quotas):
    return [key for key, n in quotas.items() for _ in range(n)]


def mock_proposal(source, primitive, domain, function):
    quote = source.text.split(".")[0] + "."
    common = dict(type=primitive, domain=domain, function=function, fact=quote,
                  evidence=[dict(start=0, end=len(quote), quote=quote)], rationale="Offline fixture only.",
                  relation="supported", numeric_values=[], rubric=[])
    if primitive == "choice":
        common.update(instructions="Which statement is explicitly in the source?", options=[quote, "The archive is closed permanently."], target_index=0)
    elif primitive == "noul":
        common.update(instructions=f"Does the source state: {quote}", options=["false", "true"], target_index=1)
    else:
        rubric = ["Zero occurrences of the exact word archive, ignoring case.", "One occurrence of the exact word archive, ignoring case.", "Two or more occurrences of the exact word archive, ignoring case."]
        count = sum(w.strip(".,").casefold() == "archive" for w in source.text.split())
        common.update(instructions="Count occurrences of the word archive, ignoring case. " + " ".join(rubric),
                      options=["none", "one", "two or more"], numeric_values=[0.0, 1.0, 2.0], rubric=rubric, target_index=min(count, 2))
    return Proposal.model_validate(common).grounded(source.text)


async def run(cfg, sources, directory, live=False, transport=None):
    directory = Path(directory)
    if not sources:
        raise ValueError("no sources")
    if len(sources) > cfg.max_source_documents:
        raise ValueError("source document limit exceeded")
    if live and (not cfg.live_enabled or any(s.mock for s in sources)):
        raise ValueError("live execution requires live_enabled and real sources")
    if any(s.split == "test" for s in sources):
        raise ValueError("official/fixed test sources cannot generate training data")
    group_splits = {}
    seen = set()
    unique = []
    for source in sources:
        if source.group_id in group_splits and group_splits[source.group_id] != source.split:
            raise ValueError("group crosses splits")
        group_splits[source.group_id] = source.split
        key = digest(normalized(source.text))
        if key not in seen:
            unique.append(source)
            seen.add(key)
    sources = unique
    with lock(directory):
        identity = dict(config=cfg.model_dump(), source_hashes=[digest(s.model_dump()) for s in sources], live=live,
                        endpoint=list(endpoint()[:2]) if live else ["mock", "offline-fixture"], version="0.1.0")
        manifest = directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("run identity changed; use a new output directory")
        atomic(manifest, json.dumps(identity, indent=2))
        journal = Journal(directory / "requests.jsonl")
        completed = Journal(directory / "checkpoint.jsonl")
        records = {r["id"]: Record.model_validate(r).model_dump() for r in read_rows(directory / "canonical.jsonl")} if (directory / "canonical.jsonl").exists() else {}
        client = Client(cfg, journal, *endpoint(), transport=transport) if live else None
        primitives, domains, functions = map(expanded, (cfg.primitive_quotas, cfg.domain_quotas, cfg.function_quotas))
        done = {r["id"] for r in completed.rows if r["status"] == "accepted"} | set(records)
        async def work(i):
            p, d, f = primitives[i], domains[i], functions[i]
            candidates = [s for s in sources if s.domain == d]
            if not candidates and (not live or d in ("science", "history", "arts_and_culture", "general")):
                candidates = [s for s in sources if s.domain == "general"]
            if not candidates:
                completed.add(dict(id=digest([identity, i]), status="rejected", error_type="NoSourceForDomain"))
                return
            s = candidates[i % len(candidates)]
            wid = digest([identity, i, s.group_id])
            if wid in done:
                return
            for round_index in range(cfg.rounds):
                call_id = f"{wid}:round{round_index}"
                try:
                    if live:
                        raw = await client.ask(call_id, generation(s, p, d, f), Proposal.model_json_schema())
                        question = Proposal.model_validate(raw).grounded(s.text)
                        if (question.type, question.domain, question.function) != (p, d, f):
                            raise ValueError("quota assignment mismatch")
                    else:
                        question = mock_proposal(s, p, d, f)
                    if len(question.options) > cfg.max_options:
                        raise ValueError("option capacity exceeded")
                    agrees = None
                    if live and cfg.independent_check:
                        check = Check.model_validate(await client.ask(call_id + ":check", checking(s, question), Check.model_json_schema()))
                        agrees = check.supported and check.target_index == question.target_index
                        if not agrees:
                            raise ValueError("independent check disagrees or unsupported")
                    record = Record(id=wid, source=s, question=question, label_provenance="teacher-proposed" if live else "programmatically-verified",
                                    verification="exact spans only; semantic claim remains teacher-proposed" if live else "offline deterministic fixture, not corpus synthesis",
                                    checker_agrees=agrees)
                    records[wid] = record.model_dump()
                    write_rows(directory / "canonical.jsonl", [records[k] for k in sorted(records)])
                    completed.add(dict(id=wid, status="accepted", round=round_index))
                    return
                except (ValueError, LimitError) as exc:
                    completed.add(dict(id=wid, status="rejected", round=round_index, error_type=type(exc).__name__))
        try:
            # Bounded task batches as well as bounded network concurrency.
            for start in range(0, len(primitives), cfg.concurrency):
                await asyncio.gather(*(work(i) for i in range(start, min(len(primitives), start + cfg.concurrency))))
        finally:
            if client:
                await client.http.aclose()
        rows = [records[k] for k in sorted(records)]
        exports(directory, rows)
        report = {"mock": not live, "accepted": len(rows), "requested": len(primitives),
                  "primitive_counts": dict(Counter(r["question"]["type"] for r in rows)),
                  "domain_counts": dict(Counter(r["question"]["domain"] for r in rows)),
                  "function_counts": dict(Counter(r["question"]["function"] for r in rows)),
                  "rates_known": cfg.input_usd_per_million is not None and cfg.output_usd_per_million is not None,
                  "billing": client.totals() if client else [0, 0, 0],
                  "file_hashes": {p.name: digest(p.read_text(encoding="utf-8")) for p in directory.glob("*.jsonl")}}
        atomic(directory / "report.json", json.dumps(report, indent=2))
        return report
