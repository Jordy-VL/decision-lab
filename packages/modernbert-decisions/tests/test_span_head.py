import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel

from decisions.model import DecisionModel, collate, encode
from test_decision_head import TinyEncoder, real_tokenizer


class SpanHeadTests(unittest.TestCase):
    def test_pooling_gradient_padding_and_option_permutation(self):
        torch.manual_seed(17)
        model = DecisionModel(TinyEncoder(), 8, 256, 'candidate_spans')
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        spans = torch.tensor([[0, 1, 1, 2, 2, 3, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
        mask = torch.arange(256)[None, :] < 2
        logits = model(ids, attention, mask, span_ids=spans).logits
        hidden = model.encoder.embedding(ids)[0]
        q = hidden[1:3].mean(0)
        opts = torch.stack([hidden[3:5].mean(0), hidden[5]])
        expected = model.head(torch.cat([q.expand_as(opts), opts, q * opts], -1)).flatten()
        torch.testing.assert_close(logits[0, :2], expected)
        self.assertEqual(logits.softmax(-1)[~mask].sum().item(), 0)
        swapped = spans.clone()
        swapped[spans == 2], swapped[spans == 3] = 3, 2
        permuted = model(ids, attention, mask, span_ids=swapped).logits
        torch.testing.assert_close(logits[0, :2].flip(0), permuted[0, :2])
        torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()
        self.assertGreater(model.encoder.embedding.weight.grad[1:6].abs().sum().item(), 0)
        self.assertEqual(model.encoder.embedding.weight.grad[7].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(ids, attention, mask, span_ids=torch.zeros_like(spans))

    def test_text_spans_and_checkpoint_roundtrip(self):
        tokenizer = real_tokenizer()
        row = dict(id='unicode', state='state words', question='which topic?', options=['café science', 'two\nlines'], target_index=0, type='categorical')
        encoded = encode(row, tokenizer, 256, 256, 'candidate_spans')
        marker = encode(row, tokenizer, 256, 256, 'candidate_masks')
        self.assertEqual(encoded['input_ids'], marker['input_ids'])
        self.assertTrue(all(encoded['span_ids'][p] == 0 for p in encoded['candidate_positions']))
        self.assertEqual(set(encoded['span_ids']), {0, 1, 2, 3})
        batch = collate([encoded], tokenizer.pad_token_id, 256)
        encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
        model = DecisionModel(encoder, 8, 256, 'candidate_spans').eval()
        expected = model(**batch).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory, tokenizer)
            loaded, _ = DecisionModel.load(directory)
            torch.testing.assert_close(loaded.eval()(**batch).logits, expected)
        with self.assertRaises(ValueError):
            encode(dict(row, options=['', 'valid']), tokenizer, 256, 256, 'candidate_spans')
        with self.assertRaises(ValueError):
            collate([encoded, marker], tokenizer.pad_token_id, 256)
