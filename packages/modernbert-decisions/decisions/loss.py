"""Rank-weighted AURC and AUGRC cross-entropy surrogates."""
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


class AUGRCLoss(torch.nn.Module):
    """Linear rank-weighted CE surrogate for area under generalized risk.

    AURC averages accepted errors / accepted count; AUGRC integrates accepted
    errors / total count over coverage (Traub et al., arXiv:2407.01032, Eq. 6).
    With descending confidence rank r=1..n, trapezoidal AUGRC assigns each
    error weight (n-r+0.5)/n inside a mean. We replace binary error with CE and
    double those weights so their mean is one, matching ordinary CE's scale.
    Thus this normalized surrogate is not a numerical AUGRC metric.

    Unlike decision_loss's harmonic AURC weights, these weights vary linearly
    with rank and remain below two: high-confidence errors receive less extreme
    emphasis. Maximum-softmax confidence ranks are detached; gradients flow
    through CE only, not sorting. Exact ties receive their average rank weight,
    making the result independent of row ordering (the existing AURC loss keeps
    batch order for ties).

    augrc_lambda blends ordinary CE (0) with the full surrogate (1). Optional
    per-row types rank separately within each type, using actual microbatch
    group sizes; singleton groups reduce to CE. Gradient accumulation does not
    enlarge the ranking population. This is a training surrogate, with no
    guarantee of improving discrete AUGRC or coverage at a fixed risk target.

    Usage: AUGRCLoss(augrc_lambda=1.0)(logits, targets, types=None).
    """

    def __init__(self, augrc_lambda=1.0):
        super().__init__()
        if not 0 <= augrc_lambda <= 1:
            raise ValueError("augrc_lambda must be in [0,1]")
        self.augrc_lambda = augrc_lambda

    def forward(self, logits, targets, types=None):
        if len(targets) == 0:
            raise ValueError("cannot compute AUGRC loss on an empty batch")
        if types is not None and len(types) != len(targets):
            raise ValueError("types must contain one entry per target")
        ce = F.cross_entropy(logits.float(), targets, reduction="none")
        if self.augrc_lambda == 0:
            return ce.mean()
        with torch.no_grad():
            confidence = logits.float().softmax(-1).max(-1).values
            weights = torch.empty_like(ce)
            groups = [list(range(len(ce)))] if types is None else [
                [i for i, t in enumerate(types) if t == kind]
                for kind in sorted(set(types))
            ]
            for indices in groups:
                idx = torch.tensor(indices, device=logits.device)
                values, order = confidence[idx].sort()  # ascending rank
                _, inverse, counts = torch.unique_consecutive(
                    values, return_inverse=True, return_counts=True
                )
                ends = counts.cumsum(0)
                # Mean ascending rank is (start + end + 1)/2, one-based.
                # Normalized weight (2*rank-1)/n simplifies as below.
                tie_weights = (2 * ends - counts).to(ce.dtype) / len(idx)
                weights[idx[order]] = tie_weights[inverse]
        return ((1 - self.augrc_lambda) * ce.mean()
                + self.augrc_lambda * (weights * ce).mean())
