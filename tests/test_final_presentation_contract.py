import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class FinalPresentationContractTests(unittest.TestCase):
    def test_history_roles_and_hash_explanation_are_consistent(self):
        history = pd.read_csv(ROOT / "artifacts" / "audit_records" / "min_history_sensitivity.csv").set_index("min_history")
        self.assertEqual(history.loc[3, "gate_a_overall_status"], "PRIMARY FAIL")
        self.assertTrue(str(history.loc[8, "gate_a_overall_status"]).startswith("SENSITIVITY PASS"))
        self.assertTrue(str(history.loc[10, "gate_a_overall_status"]).startswith("SENSITIVITY PASS"))
        audit = json.loads((ROOT / "artifacts" / "sensitivity" / "history_8_vs_10_hash_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["same_row_membership"])
        self.assertTrue(audit["same_target_hash"])
        self.assertLessEqual(audit["prediction_max_abs_difference"], 2.5e-16)

    def test_gate_policy_keeps_b1_primary_and_b2_diagnostic(self):
        config = json.loads((ROOT / "configs" / "fidelity_gate.yaml").read_text(encoding="utf-8"))
        self.assertIn("Gate B1", config["decision_rule"])
        self.assertIn("full-minus-metadata-only", config["gate_b"]["decision_rule"].lower())
        self.assertEqual(config["gate_a"]["requires"], ["overall paired RMSE upper 95% CI below zero"])
        self.assertIn("module_e", config)
        generator = (ROOT / "scripts" / "build_audit_v2_assets.py").read_text(encoding="utf-8")
        self.assertIn("Full vs. Metadata only", generator)
        self.assertIn("Weather only vs. Metadata only; exploratory representation contrast only", generator)

    def test_remaining_presentation_fixes_are_reflected_in_source_assets(self):
        tex = (ROOT / "paper_versions" / "v3_method_benchmark" / "source" / "fidelity_gated_xai_method_benchmark_v3.tex").read_text(encoding="utf-8")
        tex_flat = " ".join(tex.split())
        trace_table = (ROOT / "paper" / "generated" / "table_audit_traceability.tex").read_text(encoding="utf-8")
        release_notes = (ROOT / "reviewer_materials" / "RELEASE_NOTES.md").read_text(encoding="utf-8")

        self.assertIn("Sequential claim-eligibility workflow", tex_flat)
        self.assertIn("Because the primary features require a completed crop-season weather window", tex_flat)
        self.assertIn("Supplementary Table S2 defines each weather feature", tex_flat)
        self.assertIn("Module D is diagnostic only", tex_flat)
        self.assertIn("Module A estimates", tex_flat)
        self.assertIn("The event module requires", tex_flat)
        self.assertNotIn("Gate A", tex)
        self.assertNotIn("Gate B1", tex)
        self.assertNotIn("Gate B2", tex)
        self.assertNotIn("CHECKED", tex)
        self.assertNotIn("Gate B2 is sensitivity only", tex)
        self.assertIn("Gate B2 is an exploratory diagnostic only", release_notes)
        self.assertIn("Full paths, commands, and hashes are in the released acceptance audit", tex_flat)
        self.assertIn("ID & Artifact & Status", trace_table)
        self.assertNotIn("Reproduce", trace_table)

    def test_supplementary_weather_feature_table_has_required_columns(self):
        table = pd.read_csv(ROOT / "artifacts" / "supplement" / "table_s2_weather_feature_definitions.csv")
        required = {
            "feature",
            "feature_family",
            "nasa_power_variable",
            "aggregation",
            "window",
            "unit",
            "spatial_rule",
            "missing_handling",
            "availability",
        }
        self.assertEqual(len(table), 35)
        self.assertTrue(required.issubset(table.columns))
        self.assertTrue(table[list(required)].notna().all().all())

    def test_figure_two_has_three_row_aligned_paired_comparisons(self):
        figure = json.loads((ROOT / "artifacts" / "gates" / "figure2_three_comparisons.json").read_text(encoding="utf-8"))
        comparisons = figure["comparisons"]
        self.assertEqual(len(comparisons), 3)
        self.assertEqual(
            [(item["gate"], item["role"], item["comparison"]) for item in comparisons],
            [
                ("Gate A", "primary", "extra_trees_leaf_1_weather_only_vs_zero"),
                ("Gate B1 PRIMARY", "primary", "extra_trees_leaf_1_full_vs_metadata_only"),
                ("Gate B2 DIAGNOSTIC", "diagnostic", "extra_trees_leaf_1_weather_only_vs_metadata_only"),
            ],
        )
        self.assertTrue(all(item["n_rows"] == 333 and item["n_boot"] == 2000 and item["resampling_unit"] == "year_block" for item in comparisons))
        self.assertEqual(len({item["row_id_sha256"] for item in comparisons}), 1)
        self.assertEqual(len({item["target_sha256"] for item in comparisons}), 1)
        self.assertEqual(comparisons[1]["left"]["config_id"], figure["selected_config"]["config_id"])
        self.assertEqual(comparisons[1]["left"]["feature_family"], "full")
        self.assertEqual(comparisons[1]["right"]["feature_family"], "metadata_only")
        self.assertEqual(comparisons[2]["left"]["feature_family"], "weather_only")
        self.assertEqual(comparisons[2]["right"]["feature_family"], "metadata_only")

    def test_synthetic_observable_policy_numbers_and_limitations_are_current(self):
        summary = json.loads((ROOT / "artifacts" / "experiments" / "synthetic-gate-benchmark" / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["policy_uses_gt_or_oracle"])
        self.assertEqual(summary["fp"], 171)
        self.assertEqual(summary["fn"], 20)
        self.assertEqual(summary["invalid_ground_truth_runs"], 240)
        self.assertEqual(summary["valid_ground_truth_runs"], 180)
        self.assertEqual(f"{summary['observable_policy_sensitivity'] * 100:.1f}%", "88.9%")
        self.assertEqual(f"{summary['observable_policy_specificity'] * 100:.1f}%", "28.7%")
        self.assertEqual(f"{summary['observable_policy_permission_rate'] * 100:.1f}%", "78.8%")

        macros = (ROOT / "paper" / "generated" / "synthetic_numbers.tex").read_text(encoding="utf-8")
        self.assertIn(r"\newcommand{\SyntheticPolicyFalsePermissions}{171/240}", macros)
        self.assertIn(r"\newcommand{\SyntheticPolicyFalseAbstentions}{20/180}", macros)
        self.assertIn(r"\newcommand{\SyntheticPolicySensitivity}{88.9\%}", macros)
        self.assertIn(r"\newcommand{\SyntheticPolicySpecificity}{28.7\%}", macros)

        tex = (ROOT / "paper_versions" / "v3_method_benchmark" / "source" / "fidelity_gated_xai_method_benchmark_v3.tex").read_text(encoding="utf-8")
        tex_flat = " ".join(tex.split())
        self.assertIn("observable A+B+E policy", tex_flat)
        self.assertIn("necessary for claim discipline but not sufficient", tex_flat)
        self.assertIn("GT labels are used only for evaluation, not for policy decisions", tex_flat)
        for obsolete in ["0/240", "100% specificity", "full policy rejects omitted confounding", "oracle-rejected"]:
            self.assertNotIn(obsolete, tex)


if __name__ == "__main__":
    unittest.main()
