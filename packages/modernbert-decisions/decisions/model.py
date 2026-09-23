import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class DecisionModel(nn.Module):
    """One CLS-pooled encoder and fixed-capacity index head."""
    def __init__(self, encoder, head_hidden, max_options):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.config.hidden_size, head_hidden), nn.GELU(), nn.Linear(head_hidden, max_options))
        self.head_hidden, self.max_options = head_hidden, max_options

    def forward(self, input_ids, attention_mask, option_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(hidden).masked_fill(~option_mask, float("-inf"))

    def save(self, path, tokenizer):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "tokenizer")
        torch.save(self.head.state_dict(), path / "head.pt")
        (path / "model.json").write_text(json.dumps({"head_hidden": self.head_hidden, "max_options": self.max_options,
                                                    "format": "cls-index-v1"}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if meta["format"] != "cls-index-v1":
            raise ValueError("unsupported checkpoint format")
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        model = cls(AutoModel.from_pretrained(path / "encoder", attn_implementation="sdpa"), meta["head_hidden"], meta["max_options"])
        model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
        return model, tokenizer


def initialize(config):
    if config.checkpoint or config.resume:
        model, tokenizer = DecisionModel.load(config.resume or config.checkpoint)
        # The checkpoint owns architecture; persist its actual settings in resolved config.
        config.head_hidden, config.max_options = model.head_hidden, model.max_options
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    encoder = AutoModel.from_pretrained(config.model, revision=config.revision, attn_implementation="sdpa")
    return DecisionModel(encoder, config.head_hidden, config.max_options), tokenizer


def encode(row, tokenizer, max_length, max_options):
    """No truncation. Count content plus headings, option indices and BOS/EOS."""
    if len(row["options"]) > max_options:
        raise ValueError(f'{row["id"]}: {len(row["options"])} options exceeds max_options={max_options}')
    sections = ["State:\n" + row["state"], "\nQuestion:\n" + row["question"],
                "\nOptions:\n" + "\n".join(f"{i}: {value}" for i, value in enumerate(row["options"]))]
    encoded = tokenizer("".join(sections), add_special_tokens=True, truncation=False,
                        return_offsets_mapping=True, return_special_tokens_mask=True)
    ids = encoded["input_ids"]
    if ids[0] != tokenizer.cls_token_id:
        raise ValueError("CLS pooling requires tokenizer to prepend CLS")
    names = ("state_with_heading", "question_with_heading", "options_with_indices")
    counts = dict.fromkeys((*names, "special_tokens"), 0)
    boundaries = [len(sections[0]), len(sections[0]) + len(sections[1])]
    for (start, _), special in zip(encoded["offset_mapping"], encoded["special_tokens_mask"]):
        key = "special_tokens" if special else names[sum(start >= boundary for boundary in boundaries)]
        counts[key] += 1
    counts["total"] = len(ids)
    if len(ids) > max_length:
        raise ValueError(f'{row["id"]}: {len(ids)} tokens exceeds max_length={max_length}; counts={counts}')
    return {"input_ids": ids, "row": row, "token_counts": counts}


def collate(examples, pad_token_id, max_options):
    b, length = len(examples), max(len(e["input_ids"]) for e in examples)
    ids = torch.full((b, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((b, length), dtype=torch.long)
    mask = torch.zeros((b, max_options), dtype=torch.bool)
    for i, e in enumerate(examples):
        n, k = len(e["input_ids"]), len(e["row"]["options"])
        ids[i, :n], attention[i, :n] = torch.tensor(e["input_ids"]), 1
        mask[i, :k] = True
    return dict(input_ids=ids, attention_mask=attention, option_mask=mask)
