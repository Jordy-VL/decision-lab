"""Async Chat Completions transport with durable pessimistic reservations."""
import asyncio
import json
import os
from urllib.parse import urlsplit
import httpx
from .models import digest


class LimitError(RuntimeError):
    pass


def endpoint():
    base, model, key = (os.environ.get(k, "") for k in ("JEV_BASE_URL", "JEV_MODEL", "JEV_API_KEY"))
    parsed = urlsplit(base)
    if not base or not model or not key or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("configure JEV_BASE_URL, JEV_MODEL and JEV_API_KEY locally; URL must not contain secrets")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("use HTTPS or a loopback HTTP endpoint")
    return base.rstrip("/"), model, key


class Client:
    def __init__(self, cfg, journal, base, model, key, transport=None):
        self.cfg, self.journal, self.base, self.model = cfg, journal, base, model
        self.http = httpx.AsyncClient(timeout=cfg.timeout_seconds, transport=transport,
                                     headers={"Authorization": f"Bearer {key}"})
        self.semaphore = asyncio.Semaphore(cfg.concurrency)
        self.mutex = asyncio.Lock()

    def totals(self):
        reservations = {r["attempt_id"]: r for r in self.journal.rows if r["event"] == "reserve"}
        settled = {r["attempt_id"]: r for r in self.journal.rows if r["event"] == "settle"}
        cost = sum(settled.get(k, {}).get("cost_usd", r["cost_usd"]) for k, r in reservations.items())
        output = sum(settled.get(k, {}).get("output_tokens", r["output_tokens"]) for k, r in reservations.items())
        return len(reservations), cost, output

    async def ask(self, work_id, messages, schema):
        cfg = self.cfg
        mode = cfg.response_mode
        if cfg.input_usd_per_million is None or cfg.output_usd_per_million is None:
            raise LimitError("rates unknown: live calls require configured input/output prices")
        prior = sum(r["event"] == "reserve" and r["work_id"] == work_id for r in self.journal.rows)
        for attempt in range(prior, cfg.max_attempts):
            body = dict(model=self.model, messages=messages, temperature=cfg.temperature, max_tokens=cfg.max_output_tokens)
            if mode == "json_schema":
                body["response_format"] = {"type": mode, "json_schema": {"name": "decision", "strict": True, "schema": schema}}
            elif mode == "json_object":
                body["response_format"] = {"type": mode}
            # Schema remains available even when provider structured output is unavailable.
            body["messages"] = messages + [{"role": "user", "content": "Output JSON schema: " + json.dumps(schema)}]
            request_hash = digest(body)
            # UTF-8 byte count + overhead is intentionally conservative for normal tokenizers;
            # provider accounting can differ, hence not a guaranteed billing cap.
            input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
            reserved = (input_bound * cfg.input_usd_per_million + cfg.max_output_tokens * cfg.output_usd_per_million) / 1e6
            attempt_id = f"{work_id}:{attempt}"
            async with self.semaphore:
                async with self.mutex:
                    requests, cost, output = self.totals()
                    if requests >= cfg.max_requests or cost + reserved > cfg.budget_usd or output + cfg.max_output_tokens > cfg.max_total_output_tokens:
                        raise LimitError("request, output-token, or budget reservation limit reached")
                    self.journal.add(dict(event="reserve", attempt_id=attempt_id, work_id=work_id,
                                          cost_usd=reserved, output_tokens=cfg.max_output_tokens,
                                          request_hash=request_hash, model=self.model, base_url=self.base,
                                          request_config={k: v for k, v in body.items() if k != "messages"}))
                try:
                    response = await self.http.post(self.base + "/chat/completions", json=body)
                    if response.status_code != 200:
                        self.journal.add(dict(event="error", attempt_id=attempt_id, status=response.status_code))
                        if response.status_code in (400, 422) and mode != "plain" and cfg.allow_json_fallback and "response_format" in response.text:
                            mode = "plain"
                        elif response.status_code not in (408, 409, 429) and response.status_code < 500:
                            raise LimitError(f"non-retryable HTTP {response.status_code}")
                    else:
                        data = response.json()
                        choice = data["choices"][0]
                        usage = data.get("usage") or {}
                        input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
                        settlement = dict(event="settle", attempt_id=attempt_id, finish_reason=choice.get("finish_reason"),
                                          response_hash=digest(data), usage=usage)
                        if type(input_tokens) is int and type(output_tokens) is int and min(input_tokens, output_tokens) >= 0:
                            settlement.update(cost_usd=(input_tokens * cfg.input_usd_per_million + output_tokens * cfg.output_usd_per_million) / 1e6, output_tokens=output_tokens)
                        self.journal.add(settlement)
                        if choice.get("finish_reason") != "stop":
                            raise ValueError("incomplete response")
                        return json.loads(choice["message"]["content"])
                except (httpx.TimeoutException, httpx.TransportError, ValueError, KeyError, IndexError) as exc:
                    # Never persist exception strings or provider error bodies: they may echo keys.
                    self.journal.add(dict(event="error", attempt_id=attempt_id, error_type=type(exc).__name__))
            if attempt + 1 < cfg.max_attempts:
                await asyncio.sleep(min(2 ** attempt, 8))
        raise LimitError("attempt limit reached")
