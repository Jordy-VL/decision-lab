import json
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from .models import Strict


class Corpus(Strict):
    domain: str = "general"
    dataset: str = "Salesforce/wikitext"
    config: str | None = "wikitext-2-raw-v1"
    revision: str = "b08601e04326c79dfdd32d625aee71d232d685c3"
    split: str = "train"
    official_splits: bool = True
    text_field: str = "text"
    group_field: str | None = None
    data_files: str | None = None
    article_headers: bool = True
    license: str = "CC-BY-SA-3.0 / GFDL (dataset card)"
    provenance_url: str = "https://huggingface.co/datasets/Salesforce/wikitext"
    language: str = "en"
    seed: int = 42
    scan_rows: int = Field(default=300, ge=1, le=100000)
    max_documents: int = Field(default=6, ge=1, le=10000)
    min_chars: int = Field(default=200, ge=20)
    max_chars: int = Field(default=12000, ge=100, le=100000)


class Config(Strict):
    corpus: Corpus = Field(default_factory=Corpus)
    live_enabled: bool = False
    max_options: int = Field(default=20, ge=2, le=20)
    max_source_documents: int = Field(default=8, ge=1, le=10000)
    concurrency: int = Field(default=2, ge=1, le=16)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    max_attempts: int = Field(default=2, ge=1, le=5)
    max_requests: int = Field(default=12, ge=1, le=10000)
    rounds: int = Field(default=1, ge=1, le=5)
    max_output_tokens: int = Field(default=1800, ge=100, le=16000)
    max_total_output_tokens: int = Field(default=21600, ge=100)
    temperature: float = Field(default=0.3, ge=0, le=2)
    response_mode: Literal["json_schema", "json_object", "plain"] = "json_schema"
    allow_json_fallback: bool = True
    independent_check: bool = False
    budget_usd: float = Field(default=1.0, gt=0)
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
    primitive_quotas: dict[str, int] = Field(default_factory=lambda: {"choice": 2, "noul": 2, "score": 2})
    domain_quotas: dict[str, int] = Field(default_factory=lambda: {"science": 2, "history": 2, "arts_and_culture": 2})
    function_quotas: dict[str, int] = Field(default_factory=lambda: {"entity_relation": 2, "temporal_reasoning": 2, "quantity_or_degree": 2})

    @model_validator(mode="after")
    def quotas(self):
        for quotas in (self.primitive_quotas, self.domain_quotas, self.function_quotas):
            if not quotas or any(type(n) is not int or n < 1 for n in quotas.values()):
                raise ValueError("quotas must be positive integers")
        if set(self.primitive_quotas) != {"choice", "noul", "score"}:
            raise ValueError("all three primitives required")
        totals = [sum(q.values()) for q in (self.primitive_quotas, self.domain_quotas, self.function_quotas)]
        if len(set(totals)) != 1:
            raise ValueError("quota totals must match")
        if self.corpus.min_chars > self.corpus.max_chars:
            raise ValueError("invalid length bounds")
        return self


def load(path):
    return Config.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
