import unittest

import torch
from torch.nn import functional as F

from decisions.loss import AUGRCLoss


class AUGRCLossTests(unittest.TestCase):
    def test_matches_trapezoidal_generalized_ce_curve_and_gradient(self):
        logits = torch.tensor([[3., 0.], [1., 0.], [0.2, 0.]], requires_grad=True)
        targets = torch.tensor([1, 0, 1])
        ce = F.cross_entropy(logits, targets, reduction="none")
        # Confidence already descends. Integrate cumulative CE / total count,
        # including the zero-coverage origin; normalize the area by two.
        generalized_risk = torch.cat([ce.new_zeros(1), ce.cumsum(0) / 3])
        expected = 2 * torch.trapezoid(generalized_risk, dx=1 / 3)
        actual = AUGRCLoss()(logits, targets)
        torch.testing.assert_close(actual, expected)
        expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)
        actual.backward()
        torch.testing.assert_close(logits.grad, expected_grad)

    def test_ties_average_weights_and_are_permutation_invariant(self):
        logits = torch.tensor([[2., 0.], [2., 0.], [0., 0.]])
        targets = torch.tensor([0, 1, 0])
        ce = F.cross_entropy(logits, targets, reduction="none")
        expected = (ce * torch.tensor([4 / 3, 4 / 3, 1 / 3])).mean()
        loss = AUGRCLoss()
        torch.testing.assert_close(loss(logits, targets), expected)
        order = torch.tensor([2, 1, 0])
        torch.testing.assert_close(loss(logits[order], targets[order]), expected)

    def test_group_sizes_blending_and_masked_options(self):
        logits = torch.tensor([[3., 0., -torch.inf], [0.2, 0., -torch.inf],
                               [1., 0., -torch.inf]], requires_grad=True)
        targets = torch.tensor([1, 0, 0])
        ce = F.cross_entropy(logits, targets, reduction="none")
        weighted = (ce * torch.tensor([1.5, 0.5, 1.])).mean()
        for blend in (0., 0.5, 1.):
            actual = AUGRCLoss(blend)(logits, targets, ["a", "a", "b"])
            torch.testing.assert_close(actual, (1 - blend) * ce.mean() + blend * weighted)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_singleton_and_all_tied_reduce_to_ce(self):
        for n in (1, 4):
            logits = torch.tensor([[1., 0.]]).repeat(n, 1)
            targets = torch.arange(n) % 2
            torch.testing.assert_close(AUGRCLoss()(logits, targets),
                                       F.cross_entropy(logits, targets))

    def test_invalid_inputs(self):
        for blend in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                AUGRCLoss(blend)
        with self.assertRaises(ValueError):
            AUGRCLoss()(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        with self.assertRaises(ValueError):
            AUGRCLoss()(torch.ones(2, 2), torch.zeros(2, dtype=torch.long), ["a"])
