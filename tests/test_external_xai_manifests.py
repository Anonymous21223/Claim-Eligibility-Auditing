import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COUNTY = ROOT / "artifacts" / "experiments" / "county-v2-weather-models"
PJM = ROOT / "artifacts" / "experiments" / "external-domain-eia"
XAI = ROOT / "artifacts" / "xai"
AUDIT = ROOT / "artifacts" / "audit_records"


class ExternalAndXaiManifestTest(unittest.TestCase):
    def test_county_case_has_protocol_predictions_and_uncertainty(self):
        for name in ["county_protocol.yaml", "county_predictions.csv", "county_bootstrap.csv"]:
            self.assertTrue((COUNTY / name).exists(), name)
        protocol = (COUNTY / "county_protocol.yaml").read_text(encoding="utf-8")
        predictions = pd.read_csv(COUNTY / "county_predictions.csv", dtype={"county_fips": str})
        bootstrap = pd.read_csv(COUNTY / "county_bootstrap.csv")

        self.assertIn("selected_on_validation: extra_trees_weather", protocol)
        self.assertIn("explanation_availability: ABSTAIN", protocol)
        self.assertIn("gate_a_status: FAIL", protocol)
        self.assertIn("gate_b1_status: PASS", protocol)
        self.assertEqual(len(predictions), 1024)
        self.assertEqual(set(bootstrap["comparison"]), {
            "Gate A selected weather model vs zero residual baseline",
            "Gate B1 full vs metadata-only",
        })
        gate_a = bootstrap.set_index("comparison").loc["Gate A selected weather model vs zero residual baseline"]
        self.assertGreater(gate_a["ci95_high"], 0)

    def test_pjm_case_has_protocol_predictions_bootstrap_and_predictive_boundary(self):
        for name in ["pjm_protocol.yaml", "pjm_predictions.csv", "pjm_bootstrap.csv", "pjm_gate_decisions.json", "gated_feature_importance.csv"]:
            self.assertTrue((PJM / name).exists(), name)
        protocol = (PJM / "pjm_protocol.yaml").read_text(encoding="utf-8")
        predictions = pd.read_csv(PJM / "pjm_predictions.csv")
        bootstrap = pd.read_csv(PJM / "pjm_bootstrap.csv")
        gate_decisions = json.loads((PJM / "pjm_gate_decisions.json").read_text(encoding="utf-8"))
        gates = {row["gate"]: row for row in gate_decisions["gates"]}

        self.assertIn("gate_a_baseline: train-period mean demand naive baseline", protocol)
        self.assertIn("gate_a_status: PASS", protocol)
        self.assertIn("gate_b1_status: PASS", protocol)
        self.assertIn("xai_release_requires: Gate A and Gate B1", protocol)
        self.assertIn("feature_group_gate: PASS", protocol)
        self.assertIn("explanation_availability: INTERPRET", protocol)
        self.assertIn("predictive feature-group reliance only", protocol)
        self.assertNotIn("causal attribution", protocol.lower())
        self.assertEqual(len(predictions), 92)
        self.assertEqual(set(bootstrap["gate"]), {"Gate A", "Gate B1"})
        self.assertEqual(len(bootstrap), 4000)
        for gate in ["Gate A", "Gate B1"]:
            self.assertEqual(gates[gate]["status"], "PASS")
            self.assertLess(bootstrap[bootstrap.gate == gate]["delta_rmse_left_minus_right"].quantile(0.975), 0)
        self.assertEqual(gate_decisions["xai_release"], "INTERPRET")

    def test_xai_manifest_binds_methods_rows_hashes_and_output_scale(self):
        for name in ["xai_manifest.csv", "model_hashes.json", "explanation_row_ids.csv", "lime_config.yaml"]:
            self.assertTrue((XAI / name).exists(), name)
        manifest = pd.read_csv(XAI / "xai_manifest.csv")
        rows = pd.read_csv(XAI / "explanation_row_ids.csv")
        evidence = pd.read_csv(AUDIT / "external_xai_claim_evidence.csv")

        required_methods = {
            "SHAP",
            "LIME",
            "Group permutation",
            "Group ablation",
            "ALE response curves",
            "PJM permutation importance",
        }
        self.assertEqual(set(manifest["method"]), required_methods)
        self.assertTrue(manifest["output_scale"].str.contains("residual|demand|RMSE", regex=True).all())
        self.assertTrue(manifest["model_hash_path"].eq("artifacts/xai/model_hashes.json").all())
        self.assertGreaterEqual(rows["row_id"].nunique(), 333)
        self.assertTrue(rows["present_in_scored_panel"].all())
        self.assertIn("XAI outputs are descriptive unless the corresponding gate passes", set(evidence["claim"]))
        self.assertIn("PJM external-domain Gate A and Gate B1 permit predictive interpretation", set(evidence["claim"]))

    def test_pjm_xai_release_requires_both_gates(self):
        gate_decisions = json.loads((PJM / "pjm_gate_decisions.json").read_text(encoding="utf-8"))
        gates = {row["gate"]: row["status"] for row in gate_decisions["gates"]}
        expected_release = "INTERPRET" if gates["Gate A"] == "PASS" and gates["Gate B1"] == "PASS" else "ABSTAIN"
        self.assertEqual(gate_decisions["xai_release"], expected_release)


if __name__ == "__main__":
    unittest.main()
