import asyncio
import copy
import json
import httpx
import pytest
from pydantic import ValidationError
from jev_synthetic.cli import mock_sources
from jev_synthetic.config import Config, Corpus
from jev_synthetic.corpus import documents, fixed_split, sample_rows
from jev_synthetic.export import kev, trainer
from jev_synthetic.models import Proposal, Source, prediction
from jev_synthetic.pipeline import mock_proposal, run
from jev_synthetic.prompts import checking
from jev_synthetic.client import Client, LimitError, endpoint
from jev_synthetic.storage import Journal, read_rows


@pytest.fixture
def source():
    return mock_sources()[0]


@pytest.mark.parametrize("change", [
    {"target_index": True}, {"target_index": 2}, {"options": ["Yes", "true"]},
    {"unknown": "x"}, {"evidence": [{"start": 1, "end": 5, "quote": "wrong"}]},
])
def test_reject_invalid(source, change):
    raw = mock_proposal(source, "choice", "general", "fact").model_dump()
    raw.update(change)
    with pytest.raises(ValueError):
        Proposal.model_validate(raw).grounded(source.text)


def test_boolean_false_requires_contradiction(source):
    raw = mock_proposal(source, "noul", "general", "fact").model_dump()
    raw["target_index"] = 0
    with pytest.raises(ValueError):
        Proposal.model_validate(raw)


def test_ordinal_rejects_unordered_or_missing_rubric(source):
    raw = mock_proposal(source, "score", "general", "count").model_dump()
    for change in ({"numeric_values": [0.0, 2.0, 1.0]}, {"rubric": []}, {"instructions": "Choose a score"}):
        with pytest.raises(ValueError):
            Proposal.model_validate(raw | change)


def test_blind_check(source):
    q = mock_proposal(source, "choice", "general", "fact")
    payload = json.loads(checking(source, q)[1]["content"])
    assert not {"target_index", "evidence", "rationale", "fact", "relation"} & payload.keys()


def test_mock_resume_exports(tmp_path, source):
    cfg = Config()
    first = asyncio.run(run(cfg, [source], tmp_path))
    original = (tmp_path / "canonical.jsonl").read_bytes()
    second = asyncio.run(run(cfg, [source], tmp_path))
    assert first["accepted"] == second["accepted"] == 6
    assert original == (tmp_path / "canonical.jsonl").read_bytes()
    rows = read_rows(tmp_path / "canonical.jsonl")
    for r in rows:
        k, t = kev(r), trainer(r)
        q = k["questions"]["q"]
        assert t["target_index"] == r["question"]["target_index"]
        if q["type"] == "noul":
            assert type(q["label"]) is bool and t["options"] == ["false", "true"]
    with pytest.raises(ValueError):
        asyncio.run(run(cfg.model_copy(update={"rounds": 2}), [source], tmp_path))


def test_headers_drop_partial_article():
    rows = [{"text": t} for t in [" = First = \n", "paragraph\n", " = Second = \n", "cut off"]]
    assert list(documents(rows, Corpus())) == [("= First =", " = First = \nparagraph\n")]
    assert fixed_split("document", 42) == fixed_split("document", 42)


def test_hash_integrity_and_eval_block(source):
    with pytest.raises(ValueError):
        Source.model_validate(source.model_dump() | {"text": "changed"})
    with pytest.raises(ValueError):
        Source.model_validate(source.model_dump() | {"dataset": "jaredpalmer/kev-suites"})


def test_probability_contract():
    assert prediction([0.2, 0.8], [2, 5]) == {"index": 1, "probabilities": [0.2, 0.8], "expected_value": 4.4}
    with pytest.raises(ValueError):
        prediction([0.1, float("nan")])


def client_config(**kwargs):
    return Config(input_usd_per_million=1.0, output_usd_per_million=2.0, **kwargs)


def test_budget_reservation_restart_and_no_secret(tmp_path):
    async def scenario():
        journal = Journal(tmp_path / "requests.jsonl")
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401, text="SECRET-ECHO")
        c = Client(client_config(), journal, "https://endpoint.test/v1", "model", "SECRET-ECHO", httpx.MockTransport(handler))
        with pytest.raises(LimitError):
            await c.ask("w", [], {})
        await c.http.aclose()
        assert len(calls) == 1 and c.totals()[1] > 0
        assert "SECRET-ECHO" not in (tmp_path / "requests.jsonl").read_text()
        reopened = Client(client_config(max_requests=1), Journal(journal.path), c.base, c.model, "secret", httpx.MockTransport(handler))
        with pytest.raises(LimitError):
            await reopened.ask("new", [], {})
        assert len(calls) == 1
        await reopened.http.aclose()
    asyncio.run(scenario())


def test_json_fallback_usage_reconciliation(tmp_path):
    async def scenario():
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            if len(calls) == 1:
                return httpx.Response(400, text="response_format unsupported")
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}})
        c = Client(client_config(), Journal(tmp_path / "requests.jsonl"), "https://endpoint.test/v1", "model", "secret", httpx.MockTransport(handler))
        assert await c.ask("w", [], {}) == {"ok": True}
        assert "response_format" in calls[0] and "response_format" not in calls[1]
        assert c.totals()[0] == 2 and c.totals()[2] == 1810  # failed attempt conservatively retained
        await c.http.aclose()
    asyncio.run(scenario())


def test_unknown_rates_make_zero_calls(tmp_path):
    async def scenario():
        c = Client(Config(), Journal(tmp_path / "requests.jsonl"), "https://endpoint.test/v1", "model", "secret")
        with pytest.raises(LimitError):
            await c.ask("w", [], {})
        assert not c.journal.rows
        await c.http.aclose()
    asyncio.run(scenario())


def test_live_guard(tmp_path, source):
    with pytest.raises(ValueError):
        asyncio.run(run(Config(), [source], tmp_path, live=True))


def test_no_credential_urls(monkeypatch):
    monkeypatch.setenv("JEV_BASE_URL", "https://user:secret@host/v1")
    monkeypatch.setenv("JEV_MODEL", "model")
    monkeypatch.setenv("JEV_API_KEY", "secret")
    with pytest.raises(ValueError):
        endpoint()


def test_concurrency_and_budget_race(tmp_path):
    async def scenario():
        active = 0
        peak = 0
        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
        c = Client(client_config(concurrency=2, max_requests=3), Journal(tmp_path / "requests.jsonl"),
                   "https://endpoint.test/v1", "model", "secret", httpx.MockTransport(handler))
        results = await asyncio.gather(*(c.ask(str(i), [], {}) for i in range(6)), return_exceptions=True)
        assert peak <= 2 and c.totals()[0] == 3
        assert sum(isinstance(r, LimitError) for r in results) == 3
        assert c.totals()[2] == 5400  # no usage: preserve all reservations
        await c.http.aclose()
    asyncio.run(scenario())


def test_budget_blocks_before_transport(tmp_path):
    async def scenario():
        def handler(request):
            raise AssertionError("network must not run")
        c = Client(client_config(budget_usd=0.000001), Journal(tmp_path / "requests.jsonl"),
                   "https://endpoint.test/v1", "model", "secret", httpx.MockTransport(handler))
        with pytest.raises(LimitError):
            await c.ask("w", [], {})
        assert not c.journal.rows
        await c.http.aclose()
    asyncio.run(scenario())


def test_timeout_attempts_stay_bounded(tmp_path):
    async def scenario():
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("secret response details")
        c = Client(client_config(max_attempts=1), Journal(tmp_path / "requests.jsonl"),
                   "https://endpoint.test/v1", "model", "secret", httpx.MockTransport(handler))
        for _ in range(2):
            with pytest.raises(LimitError):
                await c.ask("same-work", [], {})
        assert calls == 1 and "secret response details" not in c.journal.path.read_text()
        await c.http.aclose()
    asyncio.run(scenario())


def test_simulated_live_pipeline_and_checker(tmp_path, source, monkeypatch):
    for name, value in {"JEV_BASE_URL": "https://endpoint.test/v1", "JEV_MODEL": "fake", "JEV_API_KEY": "secret"}.items():
        monkeypatch.setenv(name, value)
    source = source.model_copy(update={"mock": False})
    cfg = client_config(live_enabled=True, independent_check=True,
                        primitive_quotas={"choice": 1, "noul": 1, "score": 1},
                        domain_quotas={"general": 3}, function_quotas={"fact": 3})
    def handler(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][1]["content"])
        if "task" in user:
            task = user["task"]
            result = mock_proposal(source, task["type"], task["domain"], task["function"]).model_dump()
        else:
            assert "target_index" not in user and "rationale" not in user
            target = 0 if len(user["options"]) == 2 and user["options"][0] != "false" else 1
            result = dict(target_index=target, supported=True, rationale="Simulated check")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 100}})
    report = asyncio.run(run(cfg, [source], tmp_path, live=True, transport=httpx.MockTransport(handler)))
    assert report["accepted"] == 3 and report["billing"][0] == 6
    records = read_rows(tmp_path / "canonical.jsonl")
    assert all(r["label_provenance"] == "teacher-proposed" and r["checker_agrees"] for r in records)
    again = asyncio.run(run(cfg, [source], tmp_path, live=True, transport=httpx.MockTransport(handler)))
    assert again["billing"][0] == 6


def test_fixed_split_groups_and_dedup(source):
    cfg = Corpus(article_headers=False, group_field="id", min_chars=20, max_chars=1000,
                 official_splits=False, scan_rows=5, max_documents=5)
    rows = [{"id": "one", "text": source.text}, {"id": "one", "text": source.text},
            {"id": "two", "text": source.text.upper()}]
    sampled = sample_rows(rows, cfg)
    assert len(sampled) == 1
    assert sampled[0].split == fixed_split(sampled[0].group_id, cfg.seed)
