"""Verify the V2 county pipeline's provenance, temporal contract, and gate-consistent claims."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2_county"
REPORTS = ROOT / "reports"


def main() -> None:
    panel_report = json.loads((REPORTS / "experiments" / "county-panel-v2.json").read_text(encoding="utf-8"))
    weather_report = json.loads((REPORTS / "v2" / "weather_feature_coverage.json").read_text(encoding="utf-8"))
    model_report = json.loads((REPORTS / "experiments" / "county-v2-weather-models.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(DATA / "processed" / "county_winter_wheat_weather_panel.csv", dtype={"county_fips": str})
    nasser = sorted((DATA / "manifests").glob("nass_request_*.json"))
    redacted = all(json.loads(path.read_text(encoding="utf-8")).get("request", {}).get("key") == "<REDACTED>" for path in nasser)
    split_ok = (
        panel.loc[panel["split_role"].eq("train"), "year"].le(2018).all()
        and panel.loc[panel["split_role"].eq("validation"), "year"].between(2019, 2021).all()
        and panel.loc[panel["split_role"].eq("locked_holdout"), "year"].ge(2022).all()
    )
    temporal_ok = (pd.to_datetime(panel["weather_feature_date_max"]) <= pd.to_datetime(panel["target_available_date"])).all()
    feature_columns = weather_report["feature_columns"]
    feature_ok = not panel[feature_columns].isna().any(axis=1).any()
    unique_panel_rows = not panel.duplicated(["county_fips", "year"]).any()
    validation = pd.read_csv(ROOT / "artifacts" / "experiments" / "county-v2-weather-models" / "validation_metrics.csv")
    validation_selected_ok = str(validation.sort_values("rmse_bu_acre").iloc[0]["config_id"]) == model_report["selected_on_validation"]
    gate_a_fail = model_report["gate_a_selected_vs_zero"]["ci95_high"] >= 0
    claim_ok = model_report["explanation_availability"] == "ABSTAIN" if gate_a_fail else True
    checks = {
        "nass_manifest_count": len(nasser),
        "nass_keys_redacted": redacted,
        "canonical_panel_status": panel_report["status"] == "PASS_PRE_MODEL_PANEL",
        "weather_status": weather_report["status"] == "PASS",
        "weather_county_coverage": weather_report["weather_counties"] == panel_report["selected_counties"],
        "unique_county_year_panel": unique_panel_rows,
        "split_lock": split_ok,
        "temporal_contract": temporal_ok,
        "complete_feature_rows": feature_ok,
        "validation_only_selection": validation_selected_ok,
        "gate_consistent_claim": claim_ok,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "model_status": model_report["status"],
        "explanation_availability": model_report["explanation_availability"],
        "next_action": "Do not promote the V2 agricultural model; retain the locked inconclusive result and improve only through a newly registered experiment.",
    }
    output_json = REPORTS / "v2" / "v2_pipeline_audit.json"
    output_md = REPORTS / "v2" / "v2_pipeline_audit.md"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text("# V2 Pipeline Audit\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in payload.items()) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if status != "PASS":
        raise SystemExit("FAILED_V2_PIPELINE_AUDIT")


if __name__ == "__main__":
    main()
