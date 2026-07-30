from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.null_audit import rank_null_audit


class RankNullAuditTests(unittest.TestCase):
    def test_perfect_rank_has_positive_interval(self):
        values = np.array([-4., -3., -2., -1., 0., 1.])
        audit = rank_null_audit(values, values.copy(), np.array([1, 1, 2, 2, 3, 3]), 40, 200, 9)
        self.assertGreater(audit["spearman_ci95_low"], 0.0)
