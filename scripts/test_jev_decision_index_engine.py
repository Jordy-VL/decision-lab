"""Adapter tests; the pinned Decision Index kit must be importable."""
import types
import unittest
from unittest.mock import patch

import torch
from decision_index.engines import Unsupported, validate
from jev_decision_index_engine import ModernBERTDecisionEngine


class FakeModel:
    max_options = 128
    decision_head = "fixed_slot"

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, input_ids, **kwargs):
        self.batch_sizes.append(len(input_ids))
        logits = torch.arange(128, dtype=torch.float32).expand(len(input_ids), -1)
        return types.SimpleNamespace(logits=logits)


def fake_encode(row, tokenizer, max_length, max_options, head):
    if row["state"] == "too long":
        raise ValueError("example exceeds max_length=2048")
    return dict(row=row, input_ids=[1, 2, 3], candidate_positions=[], token_counts={})


def engine(batch_size):
    obj = ModernBERTDecisionEngine.__new__(ModernBERTDecisionEngine)
    obj.model = FakeModel()
    obj.tokenizer = types.SimpleNamespace(pad_token_id=0)
    obj.device = torch.device("cpu")
    obj.max_length = 2048
    obj.question_batch_size = batch_size
    return obj


class AdapterTests(unittest.TestCase):
    @patch("jev_decision_index_engine.encode", side_effect=fake_encode)
    def test_question_batching_preserves_keys_probabilities_and_noul(self, _):
        questions = {f"question-{i}": dict(type="choice", instructions="choose",
                     criteria={f"a-{i}": "first", f"b-{i}": "second"}) for i in range(18)}
        questions["boolean"] = dict(type="noul", instructions="true?")
        batched, sequential = engine(8), engine(1)
        result, raw = batched({"text": "state"}, questions)
        expected, _ = sequential({"text": "state"}, questions)
        self.assertEqual(result, expected)
        self.assertIsNone(raw)
        self.assertEqual(batched.model.batch_sizes, [8, 8, 3])
        self.assertEqual(list(result["answers"]), list(questions))
        validate(questions, result)

    def test_option_overflow_is_explicitly_unsupported(self):
        questions = {"q": dict(type="choice", criteria={str(i): "x" for i in range(129)})}
        with self.assertRaises(Unsupported):
            engine(8)("state", questions)

    @patch("jev_decision_index_engine.encode", side_effect=fake_encode)
    def test_token_overflow_is_explicitly_unsupported(self, _):
        questions = {"q": dict(type="choice", criteria={"a": "a", "b": "b"})}
        with self.assertRaises(Unsupported):
            engine(8)("too long", questions)

    def test_unknown_type_is_explicitly_unsupported(self):
        with self.assertRaises(Unsupported):
            engine(8)("state", {"q": {"type": "free_text"}})


if __name__ == "__main__":
    unittest.main()
