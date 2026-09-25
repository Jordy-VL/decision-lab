import types
import unittest

import torch
from torch import nn

from decisions.config import Config
from decisions.model import CANDIDATE_MARKER, DecisionModel, collate, encode


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size=4, vocab_size=32):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.is_gradient_checkpointing = False

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class MarkerTokenizer:
    cls_token_id = 101
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 99 if token == CANDIDATE_MARKER else 1

    def __call__(self, text, **kwargs):
        marker_count = text.count(CANDIDATE_MARKER)
        ids = [self.cls_token_id] + [99 if i < marker_count else i + 1 for i in range(marker_count + 3)]
        return {
            "input_ids": ids,
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(ids) - 1)],
            "special_tokens_mask": [1] + [0] * (len(ids) - 1),
        }


def example(option_count=2):
    return {
        "input_ids": list(range(1, option_count + 4)),
        "candidate_positions": list(range(1, option_count + 1)),
        "row": {"id": "row", "options": [f"option-{i}" for i in range(option_count)]},
        "token_counts": {"total": option_count + 3},
    }


class DecisionHeadTests(unittest.TestCase):
    def test_candidate_encoding_records_marker_positions(self):
        row = {"id": "row", "state": "state", "question": "question", "options": ["a", "b"]}
        encoded = encode(row, MarkerTokenizer(), max_length=32, max_options=4, decision_head="candidate_masks")
        self.assertEqual(len(encoded["candidate_positions"]), 2)
        self.assertEqual(encoded["candidate_positions"], [1, 2])

    def test_fixed_slot_preserves_bounded_slot_logits(self):
        model = DecisionModel(TinyEncoder(), head_hidden=3, max_options=4)
        batch = collate([example(2)], pad_token_id=0, max_options=4)
        output = model(**batch).logits
        self.assertEqual(output.shape, (1, 4))
        self.assertTrue(torch.isfinite(output[0, :2]).all())
        self.assertTrue(torch.isneginf(output[0, 2:]).all())

    def test_candidate_masks_scores_each_marker_with_shared_head(self):
        model = DecisionModel(TinyEncoder(), head_hidden=3, max_options=4, decision_head="candidate_masks")
        batch = collate([example(2)], pad_token_id=0, max_options=4)
        output = model(**batch).logits
        self.assertEqual(output.shape, (1, 4))
        self.assertTrue(torch.isfinite(output[0, :2]).all())
        self.assertTrue(torch.isneginf(output[0, 2:]).all())
        self.assertEqual(model.head[-1].out_features, 1)

    def test_config_validates_decision_head(self):
        self.assertEqual(Config(decision_head="candidate_masks").validate().decision_head, "candidate_masks")
        with self.assertRaisesRegex(ValueError, "unsupported decision head"):
            Config(decision_head="unknown").validate()


if __name__ == "__main__":
    unittest.main()
