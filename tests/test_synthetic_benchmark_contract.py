from pathlib import Path
import inspect
import json
import unittest

import pandas as pd

from scripts.run_synthetic_gate_benchmark import decide_policy


ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "artifacts" / "experiments" / "synthetic-gate-benchmark"


class SyntheticBenchmarkContractTest(unittest.TestCase):
    def test_required_artifacts_exist(self):
        expected = {
            "synthetic_scenarios.yaml",
            "synthetic_ground_truth.csv",
            "synthetic_runs_long.csv",
            "synthetic_summary.csv",
            "synthetic_summary_ci.csv",
            "synthetic_component_ablation.csv",
            "gate_component_ablation.csv",
            "scenario_level_decisions.csv",
            "observable_policy_schema.json",
            "summary.json",
        }
        missing = [name for name in expected if not (SYNTH / name).exists()]
        self.assertEqual(missing, [])

    def test_repeated_ground_truth_benchmark_has_denominators_and_ci(self):
        payload = json.loads((SYNTH / "summary.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(SYNTH / "synthetic_runs_long.csv")
        ground_truth = pd.read_csv(SYNTH / "synthetic_ground_truth.csv")
        ablation = pd.read_csv(SYNTH / "gate_component_ablation.csv")

        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["scenarios"], 14)
        self.assertGreaterEqual(payload["repeats_per_scenario"], 30)
        self.assertEqual(len(runs), payload["runs"])
        self.assertEqual(ground_truth["scenario"].nunique(), payload["scenarios"])
        self.assertEqual(int(ground_truth["ground_truth_permission"].sum()), 6)
        self.assertEqual(int((~ground_truth["ground_truth_permission"]).sum()), 8)
        self.assertEqual(payload["valid_ground_truth_runs"], 6 * 30)
        self.assertEqual(payload["invalid_ground_truth_runs"], 8 * 30)
        self.assertEqual(int((~runs["ground_truth_permission"]).sum()), payload["invalid_ground_truth_runs"])
        self.assertEqual(int(runs["ground_truth_permission"].sum()), payload["valid_ground_truth_runs"])

        required_rules = {
            "ungated",
            "Validation-only policy",
            "Module A only",
            "Module A + Module B",
            "Module A + Module E",
            "Module A + Module B + Module E",
            "Observable policy",
        }
        self.assertEqual(set(ablation["rule"]), required_rules)
        for column in [
            "false_permission_count",
            "false_permission_rate",
            "false_permission_ci95_low",
            "false_permission_ci95_high",
            "false_abstention_count",
            "false_abstention_rate",
            "false_abstention_ci95_low",
            "false_abstention_ci95_high",
            "sensitivity",
            "specificity",
            "permission_rate",
        ]:
            self.assertIn(column, ablation.columns)

    def test_observable_policy_is_scored_without_oracle_inputs(self):
        payload = json.loads((SYNTH / "summary.json").read_text(encoding="utf-8"))
        ablation = pd.read_csv(SYNTH / "gate_component_ablation.csv").set_index("rule")
        runs = pd.read_csv(SYNTH / "synthetic_runs_long.csv")
        schema = json.loads((SYNTH / "observable_policy_schema.json").read_text(encoding="utf-8"))
        full = ablation.loc["Observable policy"]

        self.assertFalse(payload["policy_uses_gt_or_oracle"])
        self.assertEqual(payload["policy_formula"], "module_a_pass AND module_b_pass AND module_e_pass")
        self.assertEqual(set(schema["allowed_inputs"]), {"module_a_pass", "module_b_pass", "module_e_pass", "requested_claim_level"})
        signature = inspect.signature(decide_policy)
        self.assertEqual(list(signature.parameters), ["observed_module_results", "requested_claim_level"])
        forbidden = {"ground_truth_permission", "scenario", "effect", "hidden_dgp", "admissibility_label", "driver", "valid"}
        self.assertTrue(forbidden.isdisjoint(signature.parameters))

        regenerated = runs.apply(
            lambda row: decide_policy(
                {
                    "module_a_pass": bool(row["module_a_pass"]),
                    "module_b_pass": bool(row["module_b_pass"]),
                    "module_e_pass": bool(row["module_e_pass"]),
                },
                row["requested_claim_level"],
            ),
            axis=1,
        )
        self.assertTrue((regenerated == runs["policy_permit"]).all())
        self.assertEqual(int(runs["false_permission"].sum()), payload["fp"])
        self.assertEqual(int(runs["false_abstention"].sum()), payload["fn"])
        self.assertEqual(int((runs["policy_permit"] & runs["ground_truth_permission"]).sum()), payload["tp"])
        self.assertEqual(int((~runs["policy_permit"] & ~runs["ground_truth_permission"]).sum()), payload["tn"])
        self.assertEqual(int(full["false_permission_count"]), payload["fp"])
        self.assertEqual(int(full["false_abstention_count"]), payload["fn"])
        self.assertGreater(full["permission_rate"], 0)
        self.assertGreater(full["sensitivity"], 0)

    def test_generated_synthetic_table_matches_observable_policy_artifact(self):
        table = (ROOT / "paper" / "generated" / "table_synthetic_scenario_decisions_compact.tex").read_text(encoding="utf-8")
        scenario = pd.read_csv(SYNTH / "scenario_level_decisions.csv")
        body = [line for line in table.splitlines() if " & " in line and not line.startswith("Scenario")]
        yes_rows = [line for line in body if " & yes & " in line]
        no_rows = [line for line in body if " & no & " in line]
        row_by_label = {line.split(" & ")[0]: line for line in body}
        self.assertEqual(len(body), 14)
        self.assertEqual(len(yes_rows), 6)
        self.assertEqual(len(no_rows), 8)
        self.assertIn("Scenario & GT & Claim & A & B & E & Policy", table)
        self.assertIn("Omitted conf. & no & event & 100\\% & 100\\% & 100\\% & 100\\%", table)
        self.assertIn("Temporal drift & no & event & 97\\% & 100\\% & 100\\% & 97\\%", table)
        self.assertIn("Spatial mismatch & yes & event & 43\\% & 100\\% & 80\\% & 40\\%", table)

        labels = {
            "correlated_features": "Corr. feat.",
            "geographic_shift": "Geo. shift",
            "measurement_error": "Meas. error",
            "omitted_confounder": "Omitted conf.",
            "spatial_resolution_mismatch": "Spatial mismatch",
            "temporal_drift": "Temporal drift",
            "train_only_detrending": "Train-only detr.",
            "imbalanced_tail": "Imbalanced tail",
            "moderate_signal": "Moderate signal",
            "no_signal": "No signal",
            "small_sample": "Small sample",
            "strong_signal": "Strong signal",
            "weak_signal": "Weak signal",
            "leakage": "Leakage",
        }
        for _, row in scenario.iterrows():
            line = row_by_label[labels[row["scenario"]]]
            self.assertIn(f"{100 * row['policy_permit_rate']:.0f}\\%", line)

    def test_manuscript_synthetic_metrics_match_artifact(self):
        tex = (ROOT / "paper_versions" / "v3_method_benchmark" / "source" / "fidelity_gated_xai_method_benchmark_v3.tex").read_text(encoding="utf-8")
        payload = json.loads((SYNTH / "summary.json").read_text(encoding="utf-8"))
        self.assertIn(f"{payload['fp']}/240", tex)
        self.assertIn(f"{100 * payload['observable_policy_false_permission_rate']:.1f}\\%", tex)
        self.assertIn(f"{payload['fn']}/180", tex)
        self.assertIn(f"{100 * payload['observable_policy_false_abstention_rate']:.1f}\\%", tex)
        self.assertIn(f"permission rate is {100 * payload['observable_policy_permission_rate']:.1f}\\%", tex)
        self.assertIn(f"sensitivity {100 * payload['observable_policy_sensitivity']:.1f}\\%", tex)
        self.assertIn(f"specificity {100 * payload['observable_policy_specificity']:.1f}\\%", tex)
        self.assertNotIn("permission to 0/240", tex)
        self.assertNotIn("38.8\\%", tex)
        self.assertNotIn("specificity 100.0\\%", tex)


if __name__ == "__main__":
    unittest.main()
