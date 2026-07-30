import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_main8_audit import fmt


class NumericFormattingTests(unittest.TestCase):
    def test_negative_zero_is_normalized_at_three_decimals(self):
        self.assertEqual(fmt(-0.0004), "0.000")
        self.assertEqual(fmt(0.0004), "0.000")
        self.assertEqual(fmt(-0.0006), "-0.001")
        self.assertEqual(fmt(0.0006), "0.001")


if __name__ == "__main__":
    unittest.main()
