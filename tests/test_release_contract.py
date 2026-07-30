"""Integration contracts for the generated audit records used in release QA."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_main8_audit import paired_delta  # noqa: E402


class ReleaseContractTests(unittest.TestCase):
    def test_locked_selection_and_baseline_records_are_auditable(self) -> None:
        selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text(encoding="utf-8"))
        access = json.loads((ROOT / "artifacts" / "audit_records" / "locked_test_access.json").read_text(encoding="utf-8"))
        baseline = pd.read_csv(ROOT / "artifacts" / "audit_records" / "baseline_vector_hashes.csv")
        self.assertEqual(selected["validation_years"], "2012-2015")
        self.assertFalse(selected["final_test_accessed_for_selection"])
        self.assertFalse(access["locked_final_test_access_for_selection"])
        self.assertTrue((baseline[baseline.record_type == "vector"].prediction_sha256.str.len() == 64).all())

    def test_gate_b1_b2_and_figure_inputs_use_same_paired_records(self) -> None:
        paired = pd.read_csv(ROOT / "artifacts" / "audit_records" / "paired_comparisons.csv").query("metric == 'rmse_t_ha'").set_index("comparison")
        figure = json.loads((ROOT / "artifacts" / "gates" / "figure2_three_comparisons.json").read_text(encoding="utf-8"))["comparisons"]
        self.assertEqual([item["role"] for item in figure], ["primary", "primary", "diagnostic"])
        self.assertEqual([item["n_rows"] for item in figure], [333, 333, 333])
        self.assertTrue(all(item["n_boot"] == 2000 and item["resampling_unit"] == "year_block" for item in figure))
        self.assertEqual(len({item["row_id_sha256"] for item in figure}), 1)
        self.assertEqual(len({item["target_sha256"] for item in figure}), 1)
        for item in figure:
            source = paired.loc[item["comparison"]]
            self.assertAlmostEqual(item["estimate"], float(source.delta_left_minus_right), places=12)
            self.assertAlmostEqual(item["ci95_low"], float(source.ci95_low), places=12)
            self.assertAlmostEqual(item["ci95_high"], float(source.ci95_high), places=12)

    def test_gate_b1_representatives_are_locked_fixed_architecture_vectors(self) -> None:
        manifest = json.loads((ROOT / "artifacts" / "audit" / "selection" / "gate_b1_representatives.json").read_text(encoding="utf-8"))
        rows = pd.read_csv(ROOT / "artifacts" / "audit" / "selection" / "gate_b1_representatives.csv")
        self.assertEqual(manifest["gate_a_selected_config_id"], "extra_trees_leaf_1")
        self.assertFalse(manifest["final_test_accessed_for_selection"])
        self.assertIn("Fixed architecture", manifest["protocol"])
        self.assertEqual(set(rows["feature_family"]), {"weather_only", "full", "metadata_only"})
        self.assertTrue(rows["locked_test_access_for_selection"].eq(False).all())
        self.assertTrue(rows["prediction_sha256"].str.len().eq(64).all())
        self.assertEqual(rows["row_id_sha256"].nunique(), 1)
        self.assertEqual(rows["target_sha256"].nunique(), 1)

    def test_invalid_bootstrap_draw_hard_fails(self) -> None:
        predictions = pd.DataFrame({"row_id": ["a", "b"], "year": [2016, 2016], "trend_residual_t_ha": [0.0, 1.0], "prediction": [0.0, 0.5]})
        invalid_draws = pd.DataFrame({"replicate": [0], "sampled_years": ["9999"]})
        with self.assertRaises(KeyError):
            paired_delta(predictions, predictions, invalid_draws, "invalid")

    def test_generated_gate_table_is_record_derived(self) -> None:
        records = pd.read_csv(ROOT / "artifacts" / "audit_records" / "fidelity_gate_components.csv")
        table = (ROOT / "paper" / "generated" / "table_gate_ab.tex").read_text(encoding="utf-8")
        self.assertIn("Full vs. Metadata-only", table)
        self.assertIn("Weather-only vs. Metadata-only", table)
        self.assertIn("Gate B1", table)
        self.assertEqual(records[records.component == "FINAL GATE A"].iloc[0].status, "FAIL")
        self.assertEqual(records[records.component == "FINAL GATE B1"].iloc[0].status, "FAIL")


if __name__ == "__main__":
    unittest.main()
