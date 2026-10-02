"""Post-evaluation diagnostic; does not change the official run or select a model."""
import collections
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'packages/modernbert-decisions'),
                str(ROOT / 'runs/decision-index-kit-20260930')]
import torch
import jev_decision_index_engine as adapter
from decisions.data import adapt, text
from decisions.model import encode, collate
from decision_index.suite.io import Suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structured-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    suffix = '-structured' if args.structured_only else ''
    out = ROOT / f'runs/decision-index-transfer-diagnostic-20261001{suffix}.json'
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(17)
    suite = Suite('runs/decision-index-suite-0.2.1', '0.2.1')
    # Hash selection is independent of correctness and confidence.
    pools = collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        if r['_evaluation']['catalog_id'] in ((61, 62) if args.structured_only else (21, 24, 28, 29, 4)):
            if len(r['questions']) == 1:
                pools[r['_evaluation']['catalog_id']].append(r)
    selected = {str(k): sorted(v, key=lambda r: hashlib.sha256(
        r['_evaluation']['run_id'].encode()).hexdigest())[:64] for k, v in pools.items()}
    report = {'selection': '64 smallest SHA256(run_id) per chosen benchmark; post-evaluation exploratory',
              'sample_ids': {k: [r['_evaluation']['run_id'] for r in v] for k, v in selected.items()},
              'models': {}}
    original_text = adapter._text
    for arm in ('aurc-only', 'ce'):
        checkpoint = f'runs/x2-{arm}-10epoch-linear-1e5/checkpoint'
        engine = adapter.ModernBERTDecisionEngine(checkpoint, device='cuda')
        result = {'checkpoint': checkpoint, 'external': {}, 'development': {}}
        dev = [json.loads(l) for l in Path('data/kev-decision-v7/development.jsonl').open()]
        dev = sorted(dev, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:128]
        correct = 0
        with torch.inference_mode():
            for start in range(0, len(dev), 8):
                rows = dev[start:start+8]
                examples = [encode(r, engine.tokenizer, 2048, 128) for r in rows]
                batch = {k:v.cuda() for k,v in collate(examples, engine.tokenizer.pad_token_id, 128).items()}
                preds = engine.model(**batch).logits.argmax(-1).tolist()
                correct += sum(p == r['target_index'] for p,r in zip(preds, rows))
        result['development'] = {'n':len(dev), 'correct':correct, 'accuracy':correct/len(dev)}
        for name, rows in selected.items():
            records = []
            for row in rows:
                key, q = next(iter(row['questions'].items()))
                gold = row['expected'][key]
                entry = {'id':row['_evaluation']['run_id'], 'gold':gold, 'variants':{}}
                for variant in ('original', 'training_spaces', 'reversed'):
                    adapter._text = text if variant == 'training_spaces' else original_text
                    questions = copy.deepcopy(row['questions'])
                    if variant == 'reversed':
                        questions[key]['criteria'] = dict(reversed(list(q['criteria'].items())))
                    try:
                        response, _ = engine(row['state'], questions)
                        a = response['answers'][key]
                        entry['variants'][variant] = {'prediction':a['choice'],
                            'slot':list(questions[key]['criteria']).index(a['choice']),
                            'confidence':max(a['probabilities'].values()), 'correct':a['choice']==gold}
                    except adapter.Unsupported as e:
                        entry['variants'][variant] = {'unsupported':str(e)}
                records.append(entry)
            summary = {}
            for variant in ('original','training_spaces','reversed'):
                values = [r['variants'][variant] for r in records if 'prediction' in r['variants'][variant]]
                summary[variant] = {'n':len(values), 'correct':sum(v['correct'] for v in values),
                    'mean_confidence':sum(v['confidence'] for v in values)/len(values) if values else None,
                    'slots':dict(collections.Counter(v['slot'] for v in values))}
            pairs = [r for r in records if all('prediction' in r['variants'][v] for v in ('original','reversed'))]
            summary['reversal'] = {'n':len(pairs), 'same_semantic_answer':sum(r['variants']['original']['prediction']==r['variants']['reversed']['prediction'] for r in pairs),
                'same_slot':sum(r['variants']['original']['slot']==r['variants']['reversed']['slot'] for r in pairs)}
            result['external'][name] = {'summary':summary,'records':records}
            print(json.dumps({'arm':arm,'dataset':name,'summary':summary}),flush=True)
        report['models'][arm] = result
        del engine
        torch.cuda.empty_cache()
    adapter._text = original_text
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('SAVED',out,flush=True)


if __name__ == '__main__':
    main()
