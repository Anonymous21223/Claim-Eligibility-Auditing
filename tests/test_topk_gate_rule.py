from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_audit import topk_pass


class TopKGateRuleTests(unittest.TestCase):
    def test_equal_to_chance_cannot_pass(self):
        result = {"lift": 1.0, "lift_ci95_low": 1.1, "hypergeometric_pvalue": .01, "permutation_pvalue": .01}
        self.assertFalse(topk_pass(result, .05))

    def test_each_conjunct_is_required(self):
        result = {"lift": 1.2, "lift_ci95_low": 1.1, "hypergeometric_pvalue": .01, "permutation_pvalue": .01}
        self.assertTrue(topk_pass(result, .05))
        for field, value in (("lift_ci95_low", 1.0), ("hypergeometric_pvalue", .05), ("permutation_pvalue", .05)):
            broken = dict(result); broken[field] = value
            self.assertFalse(topk_pass(broken, .05))
