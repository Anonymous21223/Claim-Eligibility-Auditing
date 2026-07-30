from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crop_yield_xai.audit_rules import final_gate_status, paired_error_pass, tail_component_pass


class GateLogicTests(unittest.TestCase):
    def setUp(self):
        self.config = {"tail_policy": {"primary_tail": "z<-1"}}

    def test_paired_rule_requires_upper_interval_below_zero(self):
        self.assertTrue(paired_error_pass(-0.001))
        self.assertFalse(paired_error_pass(0.0))

    def test_tail_rule_and_final_conjunction(self):
        row = {"threshold": "z<-1", "paired_delta_rmse_ci95_high": -0.01, "paired_delta_mae_ci95_high": -0.01, "rank_recovery_status": "PASS", "topk_recovery_status": "PASS"}
        self.assertTrue(tail_component_pass(row, self.config))
        row["threshold"] = "z<-1.5"
        self.assertFalse(tail_component_pass(row, self.config))
        self.assertEqual(final_gate_status({"overall": True, "tail": True}), "PASS")
        self.assertEqual(final_gate_status({"overall": True, "tail": False}), "FAIL")
