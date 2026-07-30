from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crop_yield_xai.audit_rules import top_k_recovery


class TailMetricTests(unittest.TestCase):
    def test_top_k_recovery_uses_most_negative_values(self):
        observed = np.array([-4.0, -3.0, -2.0, -1.0])
        predicted = np.array([-3.5, -2.5, -0.1, -4.0])
        self.assertEqual(top_k_recovery(observed, predicted, 2), 0.5)

    def test_top_k_recovery_rejects_invalid_k(self):
        with self.assertRaises(ValueError):
            top_k_recovery(np.array([-1.0]), np.array([-1.0]), 2)
