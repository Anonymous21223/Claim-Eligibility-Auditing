"""Build external-case protocols and XAI provenance manifests from existing artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COUNTY = ROOT / "artifacts" / "experiments" / "county-v2-weather-models"
PJM = ROOT / "artifacts" / "experiments" / "external-domain-eia"
XAI = ROOT / "artifacts" / "xai"
XAI_OUT = ROOT / "outputs" / "xai"
AUDIT = ROOT / "artifacts" / "audit_records"
REVIEWER = ROOT / "reviewer_materials"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - pred))))


def bootstrap_delta(y: np.ndarray, left: np.ndarray, right: np.ndarray, seed: int, draws: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        idx = rng.integers(0, len(y), len(y))
        rows.append({"draw": draw, "delta_rmse_left_minus_right": rmse(y[idx], left[idx]) - rmse(y[idx], right[idx])})
    return pd.DataFrame(rows)


def build_county_case() -> None:
    summary = json.loads((COUNTY / "summary.json").read_text(encoding="utf-8"))
    model_spec = json.loads((ROOT / "configs" / "experiments" / "county_v2_model_spec.json").read_text(encoding="utf-8"))
    panel_spec = json.loads((ROOT / "configs" / "experiments" / "county_v2_panel_spec.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(COUNTY / "locked_holdout_predictions.csv", dtype={"county_fips": str})
    predictions.to_csv(COUNTY / "county_predictions.csv", index=False)

    rows = [
        {
            "comparison": "Gate A selected weather model vs zero residual baseline",
            "point_delta_rmse": summary["gate_a_selected_vs_zero"]["point_delta_rmse"],
            "ci95_low": summary["gate_a_selected_vs_zero"]["ci95_low"],
            "ci95_high": summary["gate_a_selected_vs_zero"]["ci95_high"],
            "draws": summary["gate_a_selected_vs_zero"]["draws"],
            "resampling_unit": "calendar_year_block",
            "status": "FAIL" if summary["gate_a_selected_vs_zero"]["ci95_high"] >= 0 else "PASS",
        },
        {
            "comparison": "Gate B1 full vs metadata-only",
            "point_delta_rmse": summary["gate_b1_weather_increment"]["point_delta_rmse"],
            "ci95_low": summary["gate_b1_weather_increment"]["ci95_low"],
            "ci95_high": summary["gate_b1_weather_increment"]["ci95_high"],
            "draws": summary["gate_b1_weather_increment"]["draws"],
            "resampling_unit": "calendar_year_block",
            "status": "FAIL" if summary["gate_b1_weather_increment"]["ci95_high"] >= 0 else "PASS",
        },
    ]
    pd.DataFrame(rows).to_csv(COUNTY / "county_bootstrap.csv", index=False)
    yaml = f"""experiment_id: county-v2-weather-models
case_name: Agricultural external-resolution check
population: WHEAT__WINTER county-year panel with >=10 train-period observations per retained county
target: {model_spec["target"]}
source_scope: {panel_spec["source_scope"]}
train_split: "{panel_spec["split_rule"]["train"]}"
validation_split: "{panel_spec["split_rule"]["validation"]}"
locked_holdout_split: "{panel_spec["split_rule"]["locked_holdout"]}"
selected_on_validation: {summary["selected_on_validation"]}
selection_rule: {model_spec["selection"]}
feature_groups:
  metadata: {model_spec["metadata_features"]}
  weather: {model_spec["weather_features"]}
same_task_baseline: zero residual after train/validation detrending
holdout_rows: {summary["holdout_rows"]}
holdout_years: {summary["holdout_years"]}
gate_a: selected model vs zero residual baseline; upper paired 95% CI must be < 0
gate_b1: full vs metadata-only; upper paired 95% CI must be < 0 for weather-specific explanations
gate_a_status: FAIL
gate_b1_status: PASS
explanation_availability: {summary["explanation_availability"]}
claim_boundary: county-level agricultural abstention case; no successful agricultural prediction or transfer claim
predictions: {rel(COUNTY / "county_predictions.csv")}
bootstrap: {rel(COUNTY / "county_bootstrap.csv")}
"""
    write_text(COUNTY / "county_protocol.yaml", yaml)


def build_pjm_case() -> None:
    summary = json.loads((PJM / "summary.json").read_text(encoding="utf-8"))
    gate_decisions = json.loads((PJM / "pjm_gate_decisions.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(PJM / "locked_predictions.csv")
    predictions.to_csv(PJM / "pjm_predictions.csv", index=False)
    draws_path = PJM / "pjm_bootstrap_draws.csv"
    if draws_path.exists():
        pd.read_csv(draws_path).to_csv(PJM / "pjm_bootstrap.csv", index=False)
    else:
        y = predictions["observed_demand"].to_numpy(dtype=float)
        full = predictions["calendar_forecast_prediction"].to_numpy(dtype=float)
        calendar = predictions["calendar_prediction"].to_numpy(dtype=float)
        draws = bootstrap_delta(y, full, calendar, seed=23)
        draws["gate"] = "Gate B1"
        draws["comparison"] = "full vs calendar-only"
        draws.to_csv(PJM / "pjm_bootstrap.csv", index=False)
    gates = {row["gate"]: row for row in gate_decisions["gates"]}
    yaml = f"""experiment_id: external-domain-eia-pjm
case_name: Cross-domain permission case
domain: {summary["domain"]}
target: daily PJM demand in MWh
source_url: {summary["source_url"]}
train_split: "2024-01-01 through 2024-09-30"
locked_holdout_split: "2024-10-01 through 2024-12-31"
train_rows: {summary["train_rows"]}
locked_rows: {summary["locked_rows"]}
calendar_features: [dow, sin_doy, cos_doy]
full_features: [dow, sin_doy, cos_doy, forecast]
model: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=23)
gate_a_baseline: train-period mean demand naive baseline
gate_a: full model vs train-mean naive demand baseline paired bootstrap RMSE; upper 95% CI must be < 0
gate_a_delta_rmse: {gates["Gate A"]["point_delta_rmse"]}
gate_a_ci95: [{gates["Gate A"]["ci95_low"]}, {gates["Gate A"]["ci95_high"]}]
gate_a_status: {gates["Gate A"]["status"]}
gate_b1: calendar-plus-forecast full model vs calendar-only ExtraTrees paired bootstrap RMSE; upper 95% CI must be < 0
gate_b1_delta_rmse: {gates["Gate B1"]["point_delta_rmse"]}
gate_b1_ci95: [{gates["Gate B1"]["ci95_low"]}, {gates["Gate B1"]["ci95_high"]}]
gate_b1_status: {gates["Gate B1"]["status"]}
feature_group_gate: {summary["feature_group_gate"]}
xai_release_requires: Gate A and Gate B1
explanation_availability: {summary["explanation_availability"]}
output_scale: predicted daily demand in MWh
xai_method: permutation importance on locked holdout after Gate A and Gate B1 PASS
claim_boundary: predictive feature-group reliance only; not causal and not agricultural transfer validation
predictions: {rel(PJM / "pjm_predictions.csv")}
bootstrap: {rel(PJM / "pjm_bootstrap.csv")}
gate_decisions: {rel(PJM / "pjm_gate_decisions.json")}
importance: {rel(PJM / "gated_feature_importance.csv")}
"""
    write_text(PJM / "pjm_protocol.yaml", yaml)


def build_xai_provenance() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    XAI.mkdir(parents=True, exist_ok=True)
    REVIEWER.mkdir(parents=True, exist_ok=True)
    method_settings = pd.read_csv(XAI_OUT / "method_settings.csv")
    scored = pd.read_csv(XAI_OUT / "anomaly_scores_all_rows.csv")
    scored["row_id"] = scored["crop"].astype(str) + "|" + scored["region"].astype(str) + "|" + scored["year"].astype(str) + "|" + scored["window"].astype(str)
    scored_ids = set(scored["row_id"])
    shap_rows = pd.read_csv(XAI / "shap_reconstruction_checks.csv")
    lime = pd.read_csv(XAI_OUT / "lime_event_explanations.csv")
    lime_events = lime[["event_key"]].drop_duplicates().copy()
    def lime_row_id(key: object) -> str:
        country, region, crop, year, window = str(key).split("|")
        return f"{crop}|{region}|{year}|{window}"
    lime_events["row_id"] = lime_events["event_key"].map(lime_row_id)

    row_records = []
    for row_id in shap_rows["row_id"].drop_duplicates():
        row_records.append({
            "method": "SHAP",
            "row_id": row_id,
            "event_key": "",
            "row_scope": "locked_2016_2025",
            "present_in_scored_panel": row_id in scored_ids,
            "output_scale": "predicted raw residual t/ha",
            "interpretation_status": "descriptive_only_gate_failed",
        })
    for row in lime_events.itertuples(index=False):
        row_records.append({
            "method": "LIME",
            "row_id": row.row_id,
            "event_key": row.event_key,
            "row_scope": "pre-specified selected anomaly cases",
            "present_in_scored_panel": row.row_id in scored_ids,
            "output_scale": "predicted raw residual t/ha",
            "interpretation_status": "descriptive_only_gate_failed",
        })
    pd.DataFrame(row_records).to_csv(XAI / "explanation_row_ids.csv", index=False)

    lime_config = "\n".join(
        [
            "explainer: LimeTabularExplainer",
            "model: ExtraTreesRegressor",
            "feature_space: raw weather feature matrix",
            "output: residual_t_ha",
            "split: selected anomaly cases",
            f"selection_rule: {method_settings.loc[method_settings.setting.eq('lime_selection_rule'), 'value'].iloc[0]}",
            f"random_state: {method_settings.loc[method_settings.setting.eq('random_state'), 'value'].iloc[0]}",
        ]
    )
    write_text(XAI / "lime_config.yaml", lime_config + "\n")

    hashes = {
        "selected_config_json": digest(ROOT / "artifacts" / "audit" / "selection" / "selected_config.json"),
        "seed_aggregated_predictions_csv": digest(ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv"),
        "row_level_predictions_csv": digest(ROOT / "artifacts" / "audit" / "final_test" / "row_level_predictions.csv"),
        "feature_matrix_schema_json": digest(ROOT / "artifacts" / "data" / "feature_matrix_schema.json"),
        "xai_method_settings_csv": digest(XAI_OUT / "method_settings.csv"),
        "xai_scored_panel_csv": digest(XAI_OUT / "anomaly_scores_all_rows.csv"),
        "pjm_importance_csv": digest(PJM / "gated_feature_importance.csv"),
        "pjm_gate_decisions_json": digest(PJM / "pjm_gate_decisions.json"),
        "note": "No pickle is released; hashes bind selected config, feature schema, row-level predictions, and XAI output tables.",
    }
    write_text(XAI / "model_hashes.json", json.dumps(hashes, indent=2) + "\n")

    manifest_rows = [
        {
            "method": "SHAP",
            "artifact_path": rel(XAI_OUT / "shap_feature_ranking.csv"),
            "config_path": rel(XAI / "shap_config.yaml"),
            "row_id_path": rel(XAI / "explanation_row_ids.csv"),
            "model_hash_path": rel(XAI / "model_hashes.json"),
            "output_scale": "predicted raw residual t/ha",
            "row_scope": "locked and panel fitted-function diagnostics",
            "seed_handling": "five selected seeds aggregated by mean prediction where applicable",
            "status": "DESCRIPTIVE_ONLY_GATE_FAILED",
        },
        {
            "method": "LIME",
            "artifact_path": rel(XAI_OUT / "lime_event_explanations.csv"),
            "config_path": rel(XAI / "lime_config.yaml"),
            "row_id_path": rel(XAI / "explanation_row_ids.csv"),
            "model_hash_path": rel(XAI / "model_hashes.json"),
            "output_scale": "predicted raw residual t/ha",
            "row_scope": "pre-specified selected anomaly cases",
            "seed_handling": "random_state from method_settings.csv",
            "status": "DESCRIPTIVE_ONLY_GATE_FAILED",
        },
        {
            "method": "Group permutation",
            "artifact_path": rel(XAI_OUT / "group_permutation_importance.csv"),
            "config_path": rel(XAI_OUT / "method_settings.csv"),
            "row_id_path": rel(XAI / "explanation_row_ids.csv"),
            "model_hash_path": rel(XAI / "model_hashes.json"),
            "output_scale": "RMSE increase on predicted raw residual t/ha",
            "row_scope": "panel fitted-function diagnostic",
            "seed_handling": "permutation repeats from method_settings.csv",
            "status": "DESCRIPTIVE_ONLY_GATE_FAILED",
        },
        {
            "method": "Group ablation",
            "artifact_path": rel(XAI_OUT / "group_ablation_importance.csv"),
            "config_path": rel(XAI_OUT / "method_settings.csv"),
            "row_id_path": rel(XAI / "explanation_row_ids.csv"),
            "model_hash_path": rel(XAI / "model_hashes.json"),
            "output_scale": "RMSE increase on predicted raw residual t/ha",
            "row_scope": "panel fitted-function diagnostic",
            "seed_handling": "random_state from method_settings.csv",
            "status": "DESCRIPTIVE_ONLY_GATE_FAILED",
        },
        {
            "method": "ALE response curves",
            "artifact_path": rel(XAI_OUT / "ale_curves.csv"),
            "config_path": rel(XAI_OUT / "method_settings.csv"),
            "row_id_path": rel(XAI / "explanation_row_ids.csv"),
            "model_hash_path": rel(XAI / "model_hashes.json"),
            "output_scale": "accumulated local effect in residual t/ha",
            "row_scope": "panel fitted-function diagnostic",
            "seed_handling": "random_state from method_settings.csv",
            "status": "DESCRIPTIVE_ONLY_GATE_FAILED",
        },
        {
            "method": "PJM permutation importance",
            "artifact_path": rel(PJM / "gated_feature_importance.csv"),
            "config_path": rel(PJM / "pjm_protocol.yaml"),
            "row_id_path": rel(PJM / "pjm_predictions.csv"),
            "model_hash_path": rel(XAI / "model_hashes.json"),
            "output_scale": "predicted daily demand MWh",
            "row_scope": "locked October-December 2024 PJM holdout",
            "seed_handling": "ExtraTrees random_state=23; permutation random_state=23",
            "status": "INTERPRET_PREDICTIVE_RELIANCE_ONLY" if json.loads((PJM / "summary.json").read_text(encoding="utf-8"))["explanation_availability"] == "INTERPRET" else "ABSTAIN_GATE_NOT_COMPLETE",
        },
    ]
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(XAI / "xai_manifest.csv", index=False)
    lines = ["# XAI Provenance Manifest", "", "| Method | Artifact | Output scale | Status |", "|---|---|---|---|"]
    for row in manifest.itertuples(index=False):
        lines.append(f"| {row.method} | `{row.artifact_path}` | {row.output_scale} | {row.status} |")
    write_text(XAI / "xai_manifest.md", "\n".join(lines) + "\n")


def build_claim_evidence() -> None:
    county = json.loads((COUNTY / "summary.json").read_text(encoding="utf-8"))
    pjm = json.loads((PJM / "summary.json").read_text(encoding="utf-8"))
    rows = [
        {
            "claim": "County-level agricultural external-resolution check is inconclusive",
            "paper_location": "Synthetic and External-Domain Evidence",
            "artifact_path": rel(COUNTY / "summary.json"),
            "protocol_path": rel(COUNTY / "county_protocol.yaml"),
            "prediction_path": rel(COUNTY / "county_predictions.csv"),
            "bootstrap_path": rel(COUNTY / "county_bootstrap.csv"),
            "status": county["status"],
            "claim_boundary": "abstention case only",
        },
        {
            "claim": "PJM external-domain Gate A and Gate B1 permit predictive interpretation",
            "paper_location": "Synthetic and External-Domain Evidence",
            "artifact_path": rel(PJM / "summary.json"),
            "protocol_path": rel(PJM / "pjm_protocol.yaml"),
            "prediction_path": rel(PJM / "pjm_predictions.csv"),
            "bootstrap_path": rel(PJM / "pjm_bootstrap.csv"),
            "status": pjm["status"],
            "claim_boundary": "predictive reliance only after Gate A and Gate B1 pass; not causal or agricultural transfer",
        },
        {
            "claim": "XAI outputs are descriptive unless the corresponding gate passes",
            "paper_location": "Introduction; Related Work; Synthetic and External-Domain Evidence",
            "artifact_path": rel(XAI / "xai_manifest.csv"),
            "protocol_path": rel(XAI / "model_hashes.json"),
            "prediction_path": rel(XAI / "explanation_row_ids.csv"),
            "bootstrap_path": "",
            "status": "PASS",
            "claim_boundary": "predicted residual/demand output scales; no causal claim",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT / "external_xai_claim_evidence.csv", index=False)
    md = ["# External and XAI Claim Evidence", "", "| Claim | Artifact | Protocol | Status | Boundary |", "|---|---|---|---|---|"]
    for row in df.itertuples(index=False):
        md.append(f"| {row.claim} | `{row.artifact_path}` | `{row.protocol_path}` | {row.status} | {row.claim_boundary} |")
    write_text(REVIEWER / "EXTERNAL_XAI_CLAIM_EVIDENCE.md", "\n".join(md) + "\n")


def main() -> None:
    build_county_case()
    build_pjm_case()
    build_xai_provenance()
    build_claim_evidence()
    print(json.dumps({"status": "PASS", "county_protocol": rel(COUNTY / "county_protocol.yaml"), "pjm_protocol": rel(PJM / "pjm_protocol.yaml"), "xai_manifest": rel(XAI / "xai_manifest.csv")}))


if __name__ == "__main__":
    main()
