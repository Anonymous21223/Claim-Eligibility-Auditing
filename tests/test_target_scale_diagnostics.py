from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class TargetScaleDiagnosticsTests(unittest.TestCase):
    def test_training_scales_are_finite_and_guarded(self):
        values = pd.read_csv(ROOT / "artifacts" / "targets" / "train_scale_diagnostics.csv")
        self.assertTrue(values.finite.all())
        self.assertTrue(values.above_minimum.all())
        self.assertTrue((values.sigma_train > 1e-8).all())
