import copy
import types
import unittest
import tempfile
from transformers import BertConfig, BertModel
from decisions.permutation import permute_options
from decisions.config import Config
from decisions.model import encode, DecisionModel
from decisions.hf_trainer import PermutedRows, PermutationEpochCallback, make_collator, create_trainer
from test_decision_head import real_tokenizer


class PermutationTests(unittest.TestCase):
    def row(self):
        return dict(id='sample', split='train', type='categorical', state='context',
                    question='choose', options=['alpha','beta','gamma','delta'], target_index=2)

    def test_semantics_immutability_and_epoch_determinism(self):
        row=self.row(); original=copy.deepcopy(row)
        variants=[permute_options(row,17,e) for e in range(10)]
        self.assertGreater(len({tuple(r['options']) for r in variants}),1)
        for epoch,r in enumerate(variants):
            self.assertEqual(r['options'][r['target_index']], 'gamma')
            self.assertEqual(sorted(r['options']),sorted(row['options']))
            self.assertEqual(r,permute_options(row,17,epoch))
        self.assertEqual(row,original)

    def test_nontraining_rejected_and_ordered_types_preserved(self):
        row=self.row();row['split']='development'
        with self.assertRaises(ValueError):permute_options(row,17,0)
        for kind in ('boolean','ordinal'):
            row=self.row();row['type']=kind
            self.assertEqual(row,permute_options(row,17,0))

    def test_markers_and_labels_follow_epoch_and_resume(self):
        tok=real_tokenizer(); config=Config(decision_head='candidate_masks',max_options=256)
        row=self.row(); ds=PermutedRows([encode(row,tok,2048,256,'candidate_masks')],tok,config)
        callback=PermutationEpochCallback(ds)
        callback.on_epoch_begin(None,types.SimpleNamespace(epoch=3.5),None)
        first=ds[0]
        self.assertEqual(first['row']['options'][first['row']['target_index']],'gamma')
        batch=make_collator(tok,config)([first])
        self.assertEqual(batch['option_mask'].shape,(1,256))
        self.assertEqual(batch['labels'].item(),first['row']['target_index'])
        callback.on_epoch_begin(None,types.SimpleNamespace(epoch=3.0),None)
        self.assertEqual(ds[0],first)

    def test_real_trainer_updates_training_epoch_only(self):
        tok=real_tokenizer()
        model=DecisionModel(BertModel(BertConfig(vocab_size=len(tok),hidden_size=8,
            intermediate_size=16,num_hidden_layers=1,num_attention_heads=2)),4,4,'candidate_masks')
        row=self.row(); dev={**row,'id':'dev','split':'development'}
        config=Config(decision_head='candidate_masks',max_options=4,epochs=2,
            batch_size=1,save_every=1,logging_steps=1,gradient_checkpointing=False,
            device='cpu',loss_type='ce',permute_training_options=True)
        with tempfile.TemporaryDirectory() as out:
            trainer=create_trainer(model,tok,[encode(row,tok,2048,4,'candidate_masks')],
                [encode(dev,tok,2048,4,'candidate_masks')],config,out)
            trainer.train()
            self.assertEqual(trainer.train_dataset.epoch,1)
            self.assertEqual(trainer.eval_dataset[0]['row'],dev)
            self.assertEqual(trainer.state.global_step,2)

    def test_fixed_slot_256_capacity_and_checkpoint_roundtrip(self):
        tok=real_tokenizer()
        model=DecisionModel(BertModel(BertConfig(vocab_size=len(tok),hidden_size=8,
            intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
            max_position_embeddings=2048)),4,256).eval()
        row=self.row();row['options']=[f'choice {i}' for i in range(256)];row['target_index']=255
        batch=make_collator(tok,Config(max_options=256))([encode(row,tok,2048,256)])
        batch.pop('labels');batch.pop('task_types')
        import torch
        with torch.no_grad():
            logits=model(**batch).logits
        self.assertEqual(logits.shape,(1,256))
        self.assertTrue(torch.isfinite(logits).all())
        with tempfile.TemporaryDirectory() as out:
            model.save(out,tok); restored,_=DecisionModel.load(out);restored.eval()
            self.assertEqual(restored.max_options,256)
            with torch.no_grad():torch.testing.assert_close(logits,restored(**batch).logits)


if __name__ == '__main__':unittest.main()
