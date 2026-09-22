import hashlib
import json
import math
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(value):
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Evidence(Strict):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class Proposal(Strict):
    type: Literal["choice", "noul", "score"]
    domain: str = Field(min_length=1)
    function: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=20)
    target_index: int = Field(ge=0)
    evidence: list[Evidence] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    relation: Literal["supported", "contradicted"]
    numeric_values: list[float]
    rubric: list[str]

    @model_validator(mode="after")
    def validate_question(self):
        keys = [re.sub(r"\W+", "", normalized(x)) for x in self.options]
        aliases = {"yes": "true", "no": "false", "noneoftheabove": "noneofthese"}
        keys = [aliases.get(k, k) for k in keys]
        if not all(keys) or len(set(keys)) != len(keys):
            raise ValueError("empty or obviously equivalent options")
        if self.target_index >= len(self.options):
            raise ValueError("label out of range")
        if self.type == "noul":
            if self.options != ["false", "true"]:
                raise ValueError("noul order must be false, true")
            if (self.target_index == 0) != (self.relation == "contradicted"):
                raise ValueError("negative boolean needs explicit contradiction")
        if self.type == "score":
            if len(self.rubric) != len(self.options) or not all(x.strip() for x in self.rubric):
                raise ValueError("every ordinal level requires an explicit rubric")
            if any(r not in self.instructions for r in self.rubric):
                raise ValueError("ordinal instructions must contain the full rubric")
            if len(self.numeric_values) != len(self.options) or any(a >= b for a, b in zip(self.numeric_values, self.numeric_values[1:])):
                raise ValueError("numeric ordinal values must strictly increase")
        elif self.rubric or self.numeric_values:
            raise ValueError("only score has rubric/numeric values")
        return self

    def grounded(self, text):
        for span in self.evidence:
            if span.end > len(text) or span.start >= span.end or text[span.start:span.end] != span.quote:
                raise ValueError("evidence must match exact character offsets")
        # Exact quotation proves span integrity, not semantic entailment.
        return self


class Source(Strict):
    domain: str = "general"
    dataset: str
    config: str | None
    revision: str
    source_split: str
    split: str
    source_id: str
    group_id: str
    text: str
    text_sha256: str
    license: str
    provenance_url: str
    mock: bool = False

    @model_validator(mode="after")
    def integrity(self):
        if digest(self.text) != self.text_sha256:
            raise ValueError("source hash mismatch")
        if any(x in self.dataset.casefold() for x in ("kev-suites", "decision-index", "decision_index")):
            raise ValueError("frozen Decision Index data is prohibited")
        return self


class Record(Strict):
    id: str
    source: Source
    question: Proposal
    label_provenance: Literal["teacher-proposed", "annotation-backed", "programmatically-verified"]
    verification: str
    checker_agrees: bool | None

    @model_validator(mode="after")
    def integrity(self):
        self.question.grounded(self.source.text)
        return self


class Check(Strict):
    target_index: int = Field(ge=0)
    supported: bool
    rationale: str


def prediction(probabilities: list[float], numeric_values=None):
    if len(probabilities) < 2 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError("invalid probability distribution")
    result = {"index": max(range(len(probabilities)), key=probabilities.__getitem__), "probabilities": probabilities}
    if numeric_values is not None:
        if len(numeric_values) != len(probabilities):
            raise ValueError("ordinal dimensions differ")
        result["expected_value"] = sum(p * v for p, v in zip(probabilities, numeric_values))
    return result
