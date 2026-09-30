import types
import unittest
import tempfile

import torch
from torch import nn

from decisions.config import Config
from decisions.model import CANDIDATE_MARKER, DecisionModel, collate, encode
from decisions.hf_trainer import make_collator
from decisions.loss import decision_loss, AUGRCLoss
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, BertConfig, BertModel


def real_tokenizer():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]")
    tokenizer.add_special_tokens({"additional_special_tokens": [CANDIDATE_MARKER]})
    return tokenizer


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
    def test_real_tokenization_padding_labels_and_gradient_alignment(self):
        tokenizer = real_tokenizer()
        rows = [dict(id=str(k), state="some state", question="pick one", options=[str(i) for i in range(k)], target_index=k-1, type="categorical") for k in (2, 3)]
        encoded = [encode(r, tokenizer, 100, 4, "candidate_masks") for r in rows]
        for e in encoded:
            self.assertEqual([e["input_ids"][p] for p in e["candidate_positions"]], [tokenizer.convert_tokens_to_ids(CANDIDATE_MARKER)] * len(e["row"]["options"]))
        batch = make_collator(tokenizer, Config(max_options=4))(encoded)
        labels = batch.pop("labels")
        batch.pop("task_types")
        self.assertEqual(labels.tolist(), [1, 2])
        model = DecisionModel(TinyEncoder(), 3, 4, "candidate_masks")
        logits = model(**batch).logits
        expected = model.head(model.encoder(**{k: batch[k] for k in ("input_ids", "attention_mask")}).last_hidden_state[0, encoded[0]["candidate_positions"]]).squeeze(-1)
        torch.testing.assert_close(logits[0, :2], expected)
        self.assertEqual(logits.softmax(-1)[~batch["option_mask"]].sum().item(), 0)
        for loss in (decision_loss(logits, labels), decision_loss(logits, labels, 1), AUGRCLoss()(logits, labels)):
            self.assertTrue(torch.isfinite(loss))
            gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            self.assertTrue((gradient[torch.arange(2), labels] < 0).all())
            self.assertEqual(gradient[~batch["option_mask"]].abs().sum().item(), 0)

    def test_marker_collisions_and_unregistered_tokenizer_fail(self):
        row = dict(id="collision", state=CANDIDATE_MARKER, question="pick", options=["a", "b"])
        with self.assertRaisesRegex(ValueError, "marker count"):
            encode(row, real_tokenizer(), 100, 4, "candidate_masks")
        tokenizer = MarkerTokenizer()
        tokenizer.unk_token_id = 99
        row["state"] = "state"
        with self.assertRaisesRegex(ValueError, "register"):
            encode(row, tokenizer, 100, 4, "candidate_masks")

    def test_collator_rejects_missing_positions_in_candidate_batch(self):
        missing = example(2)
        missing["candidate_positions"] = []
        with self.assertRaisesRegex(ValueError, "candidate position"):
            collate([example(3), missing], 0, 4)

    def test_candidate_checkpoint_roundtrip_preserves_logits(self):
        tokenizer = real_tokenizer()
        encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2))
        model = DecisionModel(encoder, 4, 4, "candidate_masks").eval()
        row = dict(id="roundtrip", state="state", question="pick", options=["a", "b"])
        encoded = encode(row, tokenizer, 100, 4, "candidate_masks")
        batch = collate([encoded], tokenizer.pad_token_id, 4)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory, tokenizer)
            restored, restored_tokenizer = DecisionModel.load(directory)
            restored.eval()
            self.assertEqual(restored.decision_head, "candidate_masks")
            self.assertEqual(encode(row, restored_tokenizer, 100, 4, "candidate_masks")["candidate_positions"], encoded["candidate_positions"])
            torch.testing.assert_close(restored(**batch).logits, model(**batch).logits)

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
