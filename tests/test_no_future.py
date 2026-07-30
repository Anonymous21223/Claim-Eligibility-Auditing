from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.core import detrend_train_test  # noqa: E402


class TrainOnlyDetrendingTests(unittest.TestCase):
    def test_future_yield_cannot_change_evaluation_target(self) -> None:
        train = pd.DataFrame({"crop": ["Wheat"] * 4, "region": ["Texas"] * 4, "year": [2000, 2001, 2002, 2003], "yield_t_ha": [1.0, 1.2, 1.4, 1.6]})
        evaluation = pd.DataFrame({"crop": ["Wheat", "Wheat"], "region": ["Texas", "Texas"], "year": [2004, 2005], "yield_t_ha": [1.5, 9.0]})
        _, scored_a, audit_a = detrend_train_test(train, evaluation)
        changed = evaluation.copy()
        changed.loc[1, "yield_t_ha"] = 99.0
        _, scored_b, audit_b = detrend_train_test(train, changed)
        self.assertEqual(float(scored_a.loc[0, "trend_yield_t_ha"]), float(scored_b.loc[0, "trend_yield_t_ha"]))
        self.assertEqual(float(audit_a.loc[0, "residual_scale_t_ha"]), float(audit_b.loc[0, "residual_scale_t_ha"]))
        self.assertTrue((audit_a["fit_year_max"] < audit_a["evaluation_year_min"]).all())
        self.assertTrue((audit_b["future_access"] == False).all())  # noqa: E712


if __name__ == "__main__":
    unittest.main()
