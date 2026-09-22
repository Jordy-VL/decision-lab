import re
from itertools import islice
from .models import Source, digest, normalized


def fixed_split(group, seed):
    bucket = int(digest(f"{seed}:{group}")[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def documents(rows, cfg):
    """Assemble complete WikiText articles. Drop final potentially truncated article."""
    if not cfg.article_headers:
        for i, row in enumerate(islice(rows, cfg.scan_rows)):
            text = row[cfg.text_field]
            if not isinstance(text, str):
                raise ValueError("text field must be a string")
            yield str(row[cfg.group_field]) if cfg.group_field else f"{cfg.split}/{i}", text
        return
    parts, group = [], None
    for row in islice(rows, cfg.scan_rows):
        text = row[cfg.text_field]
        if re.fullmatch(r"\s*= [^=]+ =\s*", text):
            if group is not None:
                yield group, "".join(parts)
            group, parts = text.strip(), [text]
        elif group is not None:
            parts.append(text)


def sample_rows(rows, cfg):
    from langdetect import DetectorFactory, detect, LangDetectException
    DetectorFactory.seed = cfg.seed
    result, seen = [], set()
    for group, text in documents(rows, cfg):
        key = digest(normalized(text))
        if key in seen or not cfg.min_chars <= len(text) <= cfg.max_chars:
            continue
        if sum(c.isalpha() for c in text) / len(text) < 0.45:
            continue
        if len(set(text.split())) / max(1, len(text.split())) < 0.12:
            continue
        try:
            if cfg.language and detect(text) != cfg.language:
                continue
        except LangDetectException:
            continue
        seen.add(key)
        gid = f"{cfg.dataset}/{cfg.config}/{group}"
        split = cfg.split if cfg.official_splits else fixed_split(gid, cfg.seed)
        result.append(Source(domain=cfg.domain, dataset=cfg.dataset, config=cfg.config, revision=cfg.revision,
                             source_split=cfg.split, split=split, source_id=f"{cfg.split}/{group}",
                             group_id=gid, text=text, text_sha256=digest(text), license=cfg.license,
                             provenance_url=cfg.provenance_url))
    # Hash-ranking a bounded prefix is deterministic, not a uniform full-corpus sample.
    return sorted(result, key=lambda x: digest(f"{cfg.seed}:{x.group_id}"))[:cfg.max_documents]


def sample(cfg):
    from datasets import load_dataset
    if any(x in cfg.dataset.casefold() for x in ("kev-suites", "decision-index", "decision_index")):
        raise ValueError("frozen evaluation corpus prohibited")
    if cfg.split == "test":
        raise ValueError("test corpus generation disabled; leave official tests untouched")
    if cfg.data_files:
        url = f"https://huggingface.co/datasets/{cfg.dataset}/resolve/{cfg.revision}/{cfg.data_files}"
        rows = load_dataset("parquet", data_files={cfg.split: url}, split=cfg.split, streaming=True)
    else:
        rows = load_dataset(cfg.dataset, cfg.config, split=cfg.split, revision=cfg.revision, streaming=True)
    return sample_rows(rows, cfg)
