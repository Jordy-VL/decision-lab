"""Focused synthetic checks for frozen scalar calibration implementation."""
import unittest
from calibrate_confidence import fit, predict, pava, partition, metrics


def rows(scores, correct):
    return [{"confidence": s, "error": 1-c} for s, c in zip(scores, correct)]


class CalibrationTests(unittest.TestCase):
    def test_weighted_pooling(self):
        self.assertEqual(pava([1, 0, 1], [1, 3, 1]), [.25, .25, 1])

    def test_ties_and_step_boundary(self):
        model = fit(rows([.2, .2, .8], [0, 1, 1]), "isotonic")
        self.assertEqual(model["y"], [.5, 1])
        self.assertEqual(predict(model, .799), .5)
        self.assertEqual(predict(model, .8), 1)
        self.assertEqual(predict(model, 0), .5)

    def test_spline_monotone_bounded_and_flat(self):
        model = fit(rows([.1,.2,.3,.4,.6,.8,.9], [0,1,0,1,0,1,1]), "spline")
        values = [predict(model, i/1000) for i in range(1001)]
        self.assertTrue(all(0 <= v <= 1 for v in values))
        self.assertTrue(all(a <= b+1e-12 for a,b in zip(values,values[1:])))
        self.assertEqual(predict(model, 0), model["y"][0])
        self.assertEqual(predict(model, 1), model["y"][-1])

    def test_constant_spline(self):
        model = fit(rows([.25,.26], [0,1]), "spline")
        self.assertEqual(predict(model, .99), .5)

    def test_group_partition_stable_and_fit_independent(self):
        groups = ["g"+str(i) for i in range(30)]
        expected = {g: partition(g) for g in groups}
        self.assertEqual(expected, {g: partition(g) for g in reversed(groups)})
        fit_rows = rows([.2,.8], [0,1])
        before = fit(fit_rows, "isotonic")
        assessment = rows([.9], [0])
        assessment[0]["error"] = 0
        self.assertEqual(before, fit(fit_rows, "isotonic"))

    def test_generalized_risk_tie_and_order(self):
        self.assertAlmostEqual(metrics(rows([.9,.1], [1,0]))["augrc"], .125)
        self.assertAlmostEqual(metrics(rows([.5,.5], [1,0]))["augrc"], .25)
        self.assertAlmostEqual(metrics(rows([.9,.1], [0,1]))["augrc"], .375)


if __name__ == "__main__":
    unittest.main()
