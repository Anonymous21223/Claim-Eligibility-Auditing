from pathlib import Path
import json
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class FeatureContractTests(unittest.TestCase):
    def test_primary_feature_availability_is_post_season_audit(self) -> None:
        availability = pd.read_csv(ROOT / "artifacts" / "data" / "feature_availability.csv")
        primary_weather = availability[(availability.feature_group == "weather_full_season") & (availability.used_in_primary == True)]  # noqa: E712
        self.assertEqual(len(primary_weather), 35)
        self.assertTrue(primary_weather.available_at_prediction_time.str.contains("after the complete crop-season weather window").all())
        metadata = availability[availability.feature_group == "metadata"]
        self.assertEqual(set(metadata.feature), {"lat", "lon", "region", "crop"})

    def test_forbidden_target_derived_columns_are_absent_from_model_matrices(self) -> None:
        schema = json.loads((ROOT / "artifacts" / "data" / "feature_matrix_schema.json").read_text(encoding="utf-8"))
        model_columns = set(schema["metadata_only"]["numeric"])
        model_columns.update(schema["metadata_only"]["categorical"])
        model_columns.update(schema["weather_only"]["numeric"])
        model_columns.update(schema["full"]["numeric"])
        model_columns.update(schema["full"]["categorical"])
        forbidden = set(schema["forbidden_target_derived_columns"])
        self.assertTrue(forbidden)
        self.assertFalse(model_columns & forbidden)
        self.assertNotIn("year", model_columns)
        self.assertNotIn(schema["target"], model_columns)

    def test_target_feature_overlap_and_no_shortcut_audit_pass(self) -> None:
        overlap = pd.read_csv(ROOT / "artifacts" / "audit_records" / "target_feature_overlap.csv")
        no_shortcut = pd.read_csv(ROOT / "artifacts" / "audit_records" / "no_shortcut_ablation.csv")
        self.assertEqual(set(overlap.status), {"PASS"})
        self.assertEqual(set(no_shortcut.status), {"PASS"})
        self.assertIn("target_derived_columns", set(no_shortcut.ablation_group))
        self.assertIn("calendar_year", set(no_shortcut.ablation_group))


if __name__ == "__main__":
    unittest.main()
