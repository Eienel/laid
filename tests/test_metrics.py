import unittest

from bakeoff.metrics import Prediction, grouped_scores, score


class MetricsTests(unittest.TestCase):
    def test_fixed_threshold_is_inclusive(self):
        rows = [
            Prediction("r1", 0, 0.64),
            Prediction("r2", 0, 0.66),
            Prediction("a1", 1, 0.65),
            Prediction("a2", 1, 0.20),
        ]
        result = score(rows, 0.65)
        self.assertEqual(result.true_real, 1)
        self.assertEqual(result.true_ai, 1)
        self.assertEqual(result.balanced_accuracy, 0.5)

    def test_perfect_balanced_accuracy(self):
        result = score(
            [Prediction("r", 0, 0.01), Prediction("a", 1, 0.99)], 0.65
        )
        self.assertEqual(result.balanced_accuracy, 1.0)

    def test_rejects_single_class(self):
        with self.assertRaises(ValueError):
            score([Prediction("a", 1, 0.99)])

    def test_single_class_slice_is_marked_not_scored(self):
        rows = [
            Prediction("r", 0, 0.01, generator="camera"),
            Prediction("a", 1, 0.99, generator="flux"),
        ]
        report = grouped_scores(rows, "generator", 0.65)
        self.assertEqual(report["flux"]["status"], "not_scored")


if __name__ == "__main__":
    unittest.main()
