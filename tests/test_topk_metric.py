from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.null_audit import hypergeometric_tail_probability, topk_null_audit


class TopKNullAuditTests(unittest.TestCase):
    def test_perfect_ranking_has_lift_above_null(self):
        observed = np.array([-5., -4., -3., -2., -1.])
        result = topk_null_audit(observed, observed.copy(), np.array([1, 1, 2, 2, 3]), 2, 40, 200, 5)
        self.assertGreater(result["lift"], 1.0)
        self.assertEqual(result["overlap"], 2)

    def test_hypergeometric_probability_is_bounded(self):
        value = hypergeometric_tail_probability(20, 5, 5, 2)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
