"""Unit tests for EvalEngine benchmark calculations, Wilson CI, and confusion matrix."""
import unittest
from backend.app.eval_engine import EvalEngine
from backend.app.database import init_db

class TestEvalEngine(unittest.TestCase):
    def setUp(self):
        init_db()
        self.engine = EvalEngine()

    def test_wilson_score_interval(self):
        # 10 successes out of 10
        low, high = self.engine.wilson_score_interval(10, 10)
        self.assertGreater(low, 0.6)
        self.assertEqual(high, 1.0)

        # 0 successes out of 10
        low0, high0 = self.engine.wilson_score_interval(0, 10)
        self.assertEqual(low0, 0.0)
        self.assertLess(high0, 0.4)

        # Zero total
        self.assertEqual(self.engine.wilson_score_interval(0, 0), (0.0, 0.0))

    def test_run_benchmark_standard(self):
        results = self.engine.run_benchmark(is_curveball_run=False)
        self.assertEqual(results["total_test_cases"], 20)
        self.assertIn("overall_accuracy", results)
        self.assertIn("category_breakdown", results)
        self.assertIn("confusion_matrix", results)
        self.assertIn("latency_percentiles_ms", results)

        acc = results["overall_accuracy"]
        self.assertGreaterEqual(acc["precision"], 0.80)
        self.assertGreaterEqual(acc["recall"], 0.85)
        self.assertGreaterEqual(acc["f1_score"], 0.85)

        # Check Wilson CI bounds exist and are valid
        ci = acc["wilson_ci_95"]
        self.assertLessEqual(ci["precision"][0], ci["precision"][1])
        self.assertLessEqual(ci["recall"][0], ci["recall"][1])

    def test_run_benchmark_curveball(self):
        results = self.engine.run_benchmark(is_curveball_run=True)
        self.assertTrue(results["is_curveball_run"])
        self.assertEqual(results["total_test_cases"], 20)

if __name__ == "__main__":
    unittest.main()
