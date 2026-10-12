import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

CANDIDATE_MARKER = "<decision_candidate>"

class DecisionModel(nn.Module):
    """Encoder with either a fixed-slot or shared candidate scorer head."""
    def __init__(self, encoder, head_hidden, max_options, decision_head="fixed_slot"):
        super().__init__()
        if decision_head not in ("fixed_slot", "candidate_masks", "candidate_spans"):
            raise ValueError(f"unsupported decision head: {decision_head}")
        self.encoder = encoder
        self.decision_head = decision_head
        if decision_head == "fixed_slot":
            self.head = nn.Sequential(nn.Linear(encoder.config.hidden_size, head_hidden), nn.GELU(), nn.Linear(head_hidden, max_options))
        elif decision_head == "candidate_spans":
            self.head = nn.Sequential(nn.Linear(3 * encoder.config.hidden_size, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        else:
            self.head = nn.Sequential(nn.Linear(encoder.config.hidden_size, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        self.head_hidden, self.max_options = head_hidden, max_options

    def forward(self, input_ids, attention_mask, option_mask, candidate_positions=None, span_ids=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.decision_head == "fixed_slot":
            logits = self.head(hidden[:, 0]).masked_fill(~option_mask, float("-inf"))
        elif self.decision_head == "candidate_spans":
            if span_ids is None or span_ids.shape != input_ids.shape:
                raise ValueError("candidate_spans requires aligned span_ids")
            if (span_ids < 0).any() or (span_ids > self.max_options + 1).any():
                raise ValueError("invalid span ID")
            # 0 = ignored, 1 = question text, 2.. = option text.
            valid = (attention_mask != 0) & (span_ids != 0)
            slots = self.max_options + 2
            sums = hidden.new_zeros(hidden.shape[0], slots, hidden.shape[-1])
            sums.scatter_add_(1, span_ids.unsqueeze(-1).expand_as(hidden), hidden * valid.unsqueeze(-1))
            counts = hidden.new_zeros(hidden.shape[0], slots)
            counts.scatter_add_(1, span_ids, valid.to(hidden.dtype))
            if (counts[:, 1] == 0).any() or ((counts[:, 2:] == 0) & option_mask).any():
                raise ValueError("question and valid option spans must be nonempty")
            means = sums / counts.clamp_min(1).unsqueeze(-1)
            q = means[:, 1:2].expand(-1, self.max_options, -1)
            o = means[:, 2:]
            logits = self.head(torch.cat([q, o, q * o], dim=-1)).squeeze(-1).masked_fill(~option_mask, float("-inf"))
        else:
            if candidate_positions is None:
                raise ValueError("candidate_positions are required for candidate_masks")
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1)
            candidate_hidden = hidden[batch_indices, candidate_positions]
            logits = self.head(candidate_hidden).squeeze(-1).masked_fill(~option_mask, float("-inf"))
        return SequenceClassifierOutput(logits=logits)

    @property
    def is_gradient_checkpointing(self):
        return self.encoder.is_gradient_checkpointing

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def save(self, path, tokenizer):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "tokenizer")
        torch.save(self.head.state_dict(), path / "head.pt")
        (path / "model.json").write_text(json.dumps({"head_hidden": self.head_hidden, "max_options": self.max_options,
                                                    "decision_head": self.decision_head,
                                                    "format": "cls-index-v1"}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if meta["format"] != "cls-index-v1":
            raise ValueError("unsupported checkpoint format")
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        model = cls(AutoModel.from_pretrained(path / "encoder", attn_implementation="sdpa"), meta["head_hidden"], meta["max_options"],
                    meta.get("decision_head", "fixed_slot"))
        model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
        return model, tokenizer


def initialize(config):
    if config.checkpoint:
        model, tokenizer = DecisionModel.load(config.checkpoint)
        # The checkpoint owns architecture; persist its actual settings in resolved config.
        config.head_hidden, config.max_options, config.decision_head = model.head_hidden, model.max_options, model.decision_head
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    encoder = AutoModel.from_pretrained(config.model, revision=config.revision, attn_implementation="sdpa")
    if config.decision_head in ("candidate_masks", "candidate_spans"):
        tokenizer.add_special_tokens({"additional_special_tokens": [CANDIDATE_MARKER]})
        encoder.resize_token_embeddings(len(tokenizer))
    return DecisionModel(encoder, config.head_hidden, config.max_options, config.decision_head), tokenizer


def encode(row, tokenizer, max_length, max_options, decision_head="fixed_slot"):
    """No truncation. Count content plus headings, option indices and BOS/EOS."""
    if len(row["options"]) > max_options:
        raise ValueError(f'{row["id"]}: {len(row["options"])} options exceeds max_options={max_options}')
    option_prefix = (CANDIDATE_MARKER + " " if decision_head in ("candidate_masks", "candidate_spans") else "")
    sections = ["State:\n" + row["state"], "\nQuestion:\n" + row["question"],
                "\nOptions:\n" + "\n".join(f"{option_prefix}{i}: {value}" for i, value in enumerate(row["options"]))]
    encoded = tokenizer("".join(sections), add_special_tokens=True, truncation=False,
                        return_offsets_mapping=True, return_special_tokens_mask=True)
    ids = encoded["input_ids"]
    if ids[0] != tokenizer.cls_token_id:
        raise ValueError("CLS pooling requires tokenizer to prepend CLS")
    candidate_positions = []
    if decision_head in ("candidate_masks", "candidate_spans"):
        marker_id = tokenizer.convert_tokens_to_ids(CANDIDATE_MARKER)
        if marker_id is None or marker_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError("candidate tokenizer must register the candidate marker")
        candidate_positions = [i for i, token_id in enumerate(ids) if token_id == marker_id]
        if len(candidate_positions) != len(row["options"]):
            raise ValueError(f'{row["id"]}: candidate marker count does not match option count')
    names = ("state_with_heading", "question_with_heading", "options_with_indices")
    counts = dict.fromkeys((*names, "special_tokens"), 0)
    boundaries = [len(sections[0]), len(sections[0]) + len(sections[1])]
    for (start, _), special in zip(encoded["offset_mapping"], encoded["special_tokens_mask"]):
        key = "special_tokens" if special else names[sum(start >= boundary for boundary in boundaries)]
        counts[key] += 1
    counts["total"] = len(ids)
    if len(ids) > max_length:
        raise ValueError(f'{row["id"]}: {len(ids)} tokens exceeds max_length={max_length}; counts={counts}')
    result = {"input_ids": ids, "row": row, "token_counts": counts, "candidate_positions": candidate_positions}
    if decision_head == "candidate_spans":
        q_start = len(sections[0]) + len("\nQuestion:\n")
        spans = [(q_start, q_start + len(row["question"]))]
        cursor = len(sections[0]) + len(sections[1]) + len("\nOptions:\n")
        for i, value in enumerate(row["options"]):
            start = cursor + len(f"{option_prefix}{i}: ")
            spans.append((start, start + len(value)))
            cursor = start + len(value) + 1
        span_ids = []
        for (start, end), special in zip(encoded["offset_mapping"], encoded["special_tokens_mask"]):
            matches = [i + 1 for i, (a, b) in enumerate(spans) if not special and end > start and start < b and end > a]
            if len(matches) > 1:
                raise ValueError("token crosses multiple text spans")
            span_ids.append(matches[0] if matches else 0)
        if set(range(1, len(spans) + 1)) - set(span_ids):
            raise ValueError(f'{row["id"]}: question and option text spans must be nonempty')
        result["span_ids"] = span_ids
    return result


def collate(examples, pad_token_id, max_options):
    b, length = len(examples), max(len(e["input_ids"]) for e in examples)
    ids = torch.full((b, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((b, length), dtype=torch.long)
    mask = torch.zeros((b, max_options), dtype=torch.bool)
    positions = torch.zeros((b, max_options), dtype=torch.long)
    candidate_batch = any(e.get("candidate_positions") for e in examples)
    span_batch = any("span_ids" in e for e in examples)
    segments = torch.zeros((b, length), dtype=torch.long)
    for i, e in enumerate(examples):
        n, k = len(e["input_ids"]), len(e["row"]["options"])
        if span_batch:
            seg = e.get("span_ids", [])
            if len(seg) != n or set(seg) - set(range(k + 2)) or not set(range(1, k + 2)).issubset(seg):
                raise ValueError("invalid or missing text spans in batch")
            segments[i, :n] = torch.tensor(seg, dtype=torch.long)
        if candidate_batch:
            candidate_positions = e.get("candidate_positions", [])
            if len(candidate_positions) != k or any(p <= 0 or p >= n for p in candidate_positions):
                raise ValueError(f'{e["row"]["id"]}: expected one valid candidate position per option')
        ids[i, :n], attention[i, :n] = torch.tensor(e["input_ids"]), 1
        mask[i, :k] = True
        if e.get("candidate_positions"):
            positions[i, :k] = torch.tensor(e["candidate_positions"], dtype=torch.long)
    batch = dict(input_ids=ids, attention_mask=attention, option_mask=mask, candidate_positions=positions)
    if span_batch:
        batch['span_ids'] = segments
    return batch
