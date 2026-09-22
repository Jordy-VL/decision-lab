"""Harmonic empirical-AURC surrogate, blended with ordinary CE."""
import torch
from torch.nn import functional as F


def decision_loss(logits, targets, aurc_lambda=0.0, types=None):
    ce = F.cross_entropy(logits.float(), targets, reduction="none")
    if aurc_lambda == 0:
        return ce.mean()
    with torch.no_grad():
        confidence = logits.float().softmax(-1).max(-1).values
        weights = torch.empty_like(ce)
        groups = [list(range(len(ce)))] if types is None else [[i for i, t in enumerate(types) if t == kind] for kind in sorted(set(types))]
        for indices in groups:
            idx = torch.tensor(indices, device=logits.device)
            order = torch.argsort(confidence[idx], stable=True)  # ascending; ties keep batch order
            n = len(idx)
            # alpha_r = H_n - H_(n-r), for ranks 1..n. Actual microbatch size.
            alphas = (1.0 / torch.arange(n, 0, -1, device=logits.device, dtype=torch.float32)).cumsum(0)
            weights[idx[order]] = alphas
    return (1 - aurc_lambda) * ce.mean() + aurc_lambda * (weights * ce).mean()
