"""Deterministic training-only option permutations with remapped categorical labels."""
import hashlib
import random


def permute_options(row, seed, epoch):
    if row.get('split') != 'train':
        raise ValueError('option permutation requires a training row')
    # Boolean and ordinal encodings have fixed semantic ordering contracts.
    if row['type'] != 'categorical':
        return dict(row)
    target = row['target_index']
    order = list(range(len(row['options'])))
    if type(target) is not int or not 0 <= target < len(order):
        raise ValueError('permutation requires a valid target index')
    key = f'{seed}:{epoch}:{row["id"]}'.encode()
    random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big')).shuffle(order)
    return {**row, 'options':[row['options'][i] for i in order],
            'target_index':order.index(target)}
