"""Frozen development-only candidate order, content and padding diagnostics."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/modernbert-decisions'))
import torch
from decisions.model import DecisionModel, encode, collate


def main():
    torch.manual_seed(17)
    out = ROOT / 'runs/candidate-invariance-20261002.json'
    if out.exists():
        raise FileExistsError(out)
    data = ROOT / 'data/kev-decision-v7/development.jsonl'
    groups = collections.defaultdict(list)
    for line in data.open():
        r = json.loads(line)
        assert r['split'] == 'development'
        if r['type'] == 'categorical':
            groups[r['source']].append(r)
    rows = [r for source in sorted(groups) for r in sorted(groups[source], key=lambda r:
            hashlib.sha256(r['id'].encode()).hexdigest())[:12]]
    report = {'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
              'selection':'12 smallest SHA256(id) per categorical development source', 'models':{}}
    for arm in ('candidate-masks-ce', 'ce'):
        path = ROOT / f'runs/x2-{arm}-10epoch-linear-1e5/checkpoint'
        model, tokenizer = DecisionModel.load(path)
        model.cuda().eval()
        def enc(r):
            return encode(r, tokenizer, 2048, model.max_options, model.decision_head)
        def predict(examples):
            batch = {k:v.cuda() for k,v in collate(examples,tokenizer.pad_token_id,model.max_options).items()}
            with torch.inference_mode():
                logits = model(**batch).logits
            return logits.softmax(-1).cpu(), batch
        encoded = [enc(r) for r in rows]
        result = {'checkpoint':str(path), 'rows':[], 'max_batch_probability_delta':0.,
                  'invalid_slot_probability':0., 'masked_position_delta':0.}
        for start in range(0,len(rows),8):
            batch_probs,batch = predict(encoded[start:start+8])
            result['invalid_slot_probability'] = max(result['invalid_slot_probability'],batch_probs[~batch['option_mask'].cpu()].sum().item())
            if model.decision_head == 'candidate_masks':
                changed = {k:v.clone() for k,v in batch.items()}
                changed['candidate_positions'][~changed['option_mask']] = 1
                with torch.inference_mode():
                    changed_probs = model(**changed).logits.softmax(-1).cpu()
                result['masked_position_delta'] = max(result['masked_position_delta'],(changed_probs-batch_probs).abs().max().item())
            for j,r in enumerate(rows[start:start+8]):
                p,_ = predict([encoded[start+j]])
                result['max_batch_probability_delta'] = max(result['max_batch_probability_delta'],(p[0]-batch_probs[j]).abs().max().item())
                rev = copy.deepcopy(r);rev['options']=list(reversed(r['options']));rev['target_index']=len(r['options'])-1-r['target_index']
                rp,_=predict([enc(rev)])
                # Replace all candidate text equally, preserving the question and state.
                blank = copy.deepcopy(r);blank['options']=['candidate']*len(r['options'])
                bp,_=predict([enc(blank)])
                pred=int(p[0].argmax());rpred=int(rp[0].argmax());k=len(r['options'])
                result['rows'].append({'id':r['id'],'source':r['source'],'k':k,'target':r['target_index'],
                    'prediction':pred,'reversed_prediction':rpred,'blank_prediction':int(bp[0].argmax()),
                    'correct':pred==r['target_index'],'reversed_correct':rpred==rev['target_index'],
                    'same_slot':pred==rpred,'same_answer':pred==k-1-rpred,
                    'blank_probability_delta':float((p-bp).abs().max()),
                    'reversal_aligned_probability_delta':float((p[0,:k]-rp[0,:k].flip(0)).abs().max())})
        result['summary'] = {s:{'n':len(v),'correct':sum(x['correct'] for x in v),
            'reversed_correct':sum(x['reversed_correct'] for x in v),'same_slot':sum(x['same_slot'] for x in v),
            'same_answer':sum(x['same_answer'] for x in v),
            'blank_same_prediction':sum(x['prediction']==x['blank_prediction'] for x in v)}
            for s in sorted(groups) if (v := [x for x in result['rows'] if x['source']==s])}
        report['models'][arm]=result
        print(json.dumps({'arm':arm,**{k:v for k,v in result.items() if k not in ('rows','checkpoint')}}),flush=True)
        del model
        torch.cuda.empty_cache()
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
