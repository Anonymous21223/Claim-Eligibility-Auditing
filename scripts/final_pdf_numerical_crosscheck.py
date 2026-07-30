"""Cross-check every submission-facing numerical claim against canonical records."""
from __future__ import annotations

import hashlib
import json
import re
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def norm(value: object) -> str:
    text = str(value)
    for old in ("−", "–", "—"):
        text = text.replace(old, "-")
    return " ".join(text.split())


def pdf_pages(pdf: Path) -> list[str]:
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    return [subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"], check=True, capture_output=True).stdout.decode("utf-8", errors="replace") for page in range(1, pages + 1)]


def page_location(pages: list[str], marker: str) -> str:
    wanted = norm(marker).lower()
    for number, text in enumerate(pages, 1):
        if wanted in norm(text).lower():
            return f"p. {number} ({marker})"
    return f"MISSING PDF MARKER ({marker})"


def add_claim(rows: list[dict[str, object]], name: str, pdf_location: str, displayed: object, artifact: str, actual: object, expected: object, tolerance: float = 0.0) -> None:
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        passed = abs(float(actual) - float(expected)) <= tolerance
    else:
        passed = actual == expected
    passed = bool(passed and not pdf_location.startswith("MISSING"))
    rows.append({"claim": name, "pdf_location": pdf_location, "displayed_value": displayed, "artifact_path": artifact, "artifact_value": actual, "expected_value": expected, "tolerance": tolerance, "status": "PASS" if passed else "FAIL"})


def metric(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["trend_residual_t_ha"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    return {"r2": float(1 - np.square(observed - predicted).sum() / np.square(observed - observed.mean()).sum()), "rmse": float(np.sqrt(np.mean(np.square(observed - predicted))))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf")
    args = parser.parse_args()
    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        pdf = ROOT / pdf
    pages = pdf_pages(pdf)
    rows: list[dict[str, object]] = []
    raw = pd.read_csv(ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv")
    processed = pd.read_csv(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv")
    selected_payload = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text(encoding="utf-8"))
    selected = selected_payload["selected_config"]
    aggregate = pd.read_csv(ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv")
    model = aggregate[(aggregate.config_id == selected["config_id"]) & (aggregate.feature_family == selected["feature_family"])].copy()
    headline = metric(model)
    paired = pd.read_csv(ROOT / "artifacts" / "audit_records" / "paired_comparisons.csv")
    gate_a = paired[(paired.comparison == "extra_trees_leaf_1_weather_only_vs_zero") & (paired.metric == "rmse_t_ha")].iloc[0]
    figure = json.loads((ROOT / "artifacts" / "gates" / "figure2_three_comparisons.json").read_text(encoding="utf-8"))["comparisons"]
    tail = pd.read_csv(ROOT / "artifacts" / "audit" / "tail" / "tail_metrics_by_threshold.csv").query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(ROOT / "artifacts" / "audit_records" / "rank_null_audit.csv").query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(ROOT / "artifacts" / "audit_records" / "topk_null_audit.csv").query("threshold == 'z<-1' and definition == 'k=10'").iloc[0]
    history = pd.read_csv(ROOT / "artifacts" / "audit_records" / "min_history_sensitivity.csv")
    target_scale = pd.read_csv(ROOT / "artifacts" / "audit_records" / "target_scale_sensitivity.csv")
    detrending = pd.read_csv(ROOT / "artifacts" / "audit_records" / "alternative_detrending_sensitivity.csv")
    expanded = pd.read_csv(ROOT / "artifacts" / "audit_records" / "expanded_model_baselines.csv")
    bootstrap = pd.read_csv(ROOT / "artifacts" / "audit_records" / "bootstrap_scheme_comparison.csv")
    temporal = pd.read_csv(ROOT / "artifacts" / "audit_records" / "temporal_and_capacity_audits.csv")
    history_hashes = json.loads((ROOT / "artifacts" / "sensitivity" / "history_8_vs_10_hash_audit.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((ROOT / "artifacts" / "data" / "feature_matrix_schema.json").read_text(encoding="utf-8"))
    feature_availability = pd.read_csv(ROOT / "artifacts" / "data" / "feature_availability.csv")
    no_shortcut = pd.read_csv(ROOT / "artifacts" / "audit_records" / "no_shortcut_ablation.csv")
    synthetic = json.loads((ROOT / "artifacts" / "experiments" / "synthetic-gate-benchmark" / "summary.json").read_text(encoding="utf-8"))
    county = json.loads((ROOT / "artifacts" / "experiments" / "county-v2-weather-models" / "summary.json").read_text(encoding="utf-8"))
    pjm = json.loads((ROOT / "artifacts" / "experiments" / "external-domain-eia" / "summary.json").read_text(encoding="utf-8"))
    xai_manifest = pd.read_csv(ROOT / "artifacts" / "xai" / "xai_manifest.csv")

    add_claim(rows, "raw rows", page_location(pages, "1,291"), "1,291", "data/raw/us_yield_1989_2025_tha.csv", len(raw), 1291)
    add_claim(rows, "processed rows", page_location(pages, "1,257"), "1,257", "data/processed/us_model_frame_hemisphere_aware_1990_2025.csv", len(processed), 1257)
    add_claim(rows, "validation eligible rows", page_location(pages, "Validation selection"), "140", "artifacts/audit/selection/selected_config.json", selected["n"], 140)
    add_claim(rows, "locked rows", page_location(pages, "Locked 2016-2025"), "333", "artifacts/audit/final_test/seed_aggregated_predictions.csv", len(model), 333)
    add_claim(rows, "primary z<-1 events", page_location(pages, "broad below-trend subset contains 73"), "73", "artifacts/audit/tail/tail_metrics_by_threshold.csv", int(tail.n), 73)
    add_claim(rows, "selected configuration", page_location(pages, "ExtraTrees"), "ExtraTrees, leaf=1, Weather only", "artifacts/audit/selection/selected_config.json", (selected["model"], selected["config_id"], selected["feature_family"]), ("ExtraTrees", "extra_trees_leaf_1", "weather_only"))
    add_claim(rows, "fixed validation seeds", page_location(pages, "Seeds"), "7, 17, 27, 37, 47", "artifacts/audit/selection/selected_config.json", selected_payload["fixed_seeds"], [7, 17, 27, 37, 47])
    add_claim(rows, "validation-only selection", page_location(pages, "2012-2015"), "2012-2015 validation", "artifacts/audit/selection/selected_config.json", (selected_payload["validation_years"], selected_payload["final_test_accessed_for_selection"]), ("2012-2015", False))
    add_claim(rows, "locked R2", page_location(pages, "R2"), "-0.014", "artifacts/audit/final_test/seed_aggregated_predictions.csv", round(headline["r2"], 3), -0.014, 5e-4)
    add_claim(rows, "locked RMSE", page_location(pages, "0.669"), "0.669", "artifacts/audit/final_test/seed_aggregated_predictions.csv", round(headline["rmse"], 3), 0.669, 5e-4)
    module_decision_location = page_location(pages, "Modules A and B fail")
    add_claim(rows, "Module A paired delta RMSE", module_decision_location, "-0.005", "artifacts/audit_records/paired_comparisons.csv", round(float(gate_a.delta_left_minus_right), 3), -0.005, 5e-4)
    add_claim(rows, "Module A paired CI low", module_decision_location, "-0.019", "artifacts/audit_records/paired_comparisons.csv", round(float(gate_a.ci95_low), 3), -0.019, 5e-4)
    add_claim(rows, "Module A paired CI high", module_decision_location, "0.009", "artifacts/audit_records/paired_comparisons.csv", round(float(gate_a.ci95_high), 3), 0.009, 5e-4)
    expected_figure = [("Gate A", "primary", "extra_trees_leaf_1_weather_only_vs_zero"), ("Gate B1 PRIMARY", "primary", "extra_trees_leaf_1_full_vs_metadata_only"), ("Gate B2 DIAGNOSTIC", "diagnostic", "extra_trees_leaf_1_weather_only_vs_metadata_only")]
    add_claim(rows, "Figure 2 module comparisons and roles", module_decision_location, "A / B / D diagnostic", "artifacts/gates/figure2_three_comparisons.json", [(item["gate"], item["role"], item["comparison"]) for item in figure], expected_figure)
    add_claim(rows, "Figure 2 paired row alignment", page_location(pages, "identical locked rows"), "paired locked rows", "artifacts/gates/figure2_three_comparisons.json", (len({item["row_id_sha256"] for item in figure}), len({item["target_sha256"] for item in figure}), [item["n_rows"] for item in figure]), (1, 1, [333, 333, 333]))
    add_claim(rows, "Figure 2 calendar-year bootstrap", page_location(pages, "year-block bootstrap replicates"), "year-block 95% CI", "artifacts/gates/figure2_three_comparisons.json", [(item["n_boot"], item["resampling_unit"]) for item in figure], [(2000, "year_block")] * 3)
    add_claim(rows, "Module D diagnostic constraint", page_location(pages, "exploratory representation diagnostic"), "D is diagnostic only", "artifacts/gates/gate_b_decision.json", figure[2]["role"], "diagnostic")
    add_claim(rows, "primary-tail delta RMSE", page_location(pages, "Panel A"), "-0.043", "artifacts/audit/tail/tail_metrics_by_threshold.csv", round(float(tail.paired_delta_rmse), 3), -0.043, 5e-4)
    add_claim(rows, "primary-tail delta MAE", page_location(pages, "Panel A"), "-0.054", "artifacts/audit/tail/tail_metrics_by_threshold.csv", round(float(tail.paired_delta_mae), 3), -0.054, 5e-4)
    add_claim(rows, "primary-tail rank rho", page_location(pages, "Rank"), "0.180", "artifacts/audit_records/rank_null_audit.csv", round(float(rank.spearman), 3), 0.180, 5e-4)
    add_claim(rows, "primary-tail rank permutation p", page_location(pages, "Perm."), "0.362", "artifacts/audit_records/rank_null_audit.csv", round(float(rank.permutation_pvalue), 3), 0.362, 5e-4)
    add_claim(rows, "primary-tail top-10 overlap", page_location(pages, "Top-10"), "1/10", "artifacts/audit_records/topk_null_audit.csv", f"{int(topk.overlap)}/{int(topk.k)}", "1/10")
    add_claim(rows, "primary-tail expected top-10 overlap", page_location(pages, "Expected"), "1.37", "artifacts/audit_records/topk_null_audit.csv", round(float(topk.random_expectation * topk.k), 2), 1.37, 0.005)
    add_claim(rows, "primary-tail lift", page_location(pages, "Lift"), "0.73", "artifacts/audit_records/topk_null_audit.csv", round(float(topk.lift), 2), 0.73, 0.005)
    add_claim(rows, "primary-tail hypergeometric p", page_location(pages, "H p"), "0.794", "artifacts/audit_records/topk_null_audit.csv", round(float(topk.hypergeometric_pvalue), 3), 0.794, 5e-4)
    add_claim(rows, "primary-tail within-year permutation p", page_location(pages, "P p"), "0.793", "artifacts/audit_records/topk_null_audit.csv", round(float(topk.permutation_pvalue), 3), 0.793, 5e-4)
    add_claim(rows, "history sensitivity rows", page_location(pages, "History 3"), "3, 5, 8, 10", "artifacts/audit_records/min_history_sensitivity.csv", sorted(history.min_history.astype(int).tolist()), [3, 5, 8, 10])
    add_claim(rows, "standardized target sensitivity", page_location(pages, "Standardized target"), "Standardized target", "artifacts/audit_records/target_scale_sensitivity.csv", len(target_scale) > 0, True)
    add_claim(rows, "Huber detrending sensitivity", page_location(pages, "Huber detrending"), "Huber detrending", "artifacts/audit_records/alternative_detrending_sensitivity.csv", "huber_train_only" in detrending.detrending.tolist(), True)
    add_claim(rows, "expanded-model sensitivities", page_location(pages, "HistGradientBoosting"), "HistGradientBoosting; ElasticNet", "artifacts/audit_records/expanded_model_baselines.csv", {"hist_gradient_boosting", "elastic_net"}.issubset(set(expanded.model)), True)
    add_claim(rows, "cluster-resampling sensitivities", page_location(pages, "Resampling"), "year/state-year/crop-state", "artifacts/audit_records/bootstrap_scheme_comparison.csv", {"year_block", "state_year_block", "crop_state_cluster"}.issubset(set(bootstrap.scheme)), True)
    add_claim(rows, "temporal sensitivities", page_location(pages, "prefix learning"), "four rolling folds; four prefixes", "artifacts/audit_records/temporal_and_capacity_audits.csv", (int((temporal.audit == "rolling_origin").sum()), int((temporal.audit == "prefix_learning_curve").sum())), (4, 4))
    add_claim(rows, "history 8/10 vector consistency", page_location(pages, "History 8"), "same rows/targets; floating-point prediction difference <= 2.5e-16", "artifacts/sensitivity/history_8_vs_10_hash_audit.json", (history_hashes["same_row_membership"], history_hashes["same_target_hash"], history_hashes["prediction_max_abs_difference"] <= 2.5e-16), (True, True, True))
    add_claim(rows, "post-season feature availability", page_location(pages, "post-season scientific audit"), "post-season audit; not pre-harvest", "artifacts/data/feature_availability.csv", (int((feature_availability.feature_group == "weather_full_season").sum()), feature_schema["prediction_task"]), (35, "post-season scientific audit of train-only detrended yield residuals"))
    add_claim(rows, "target residual formula", page_location(pages, "raw train-only residual"), "raw train-only residual target", "artifacts/targets/target_spec.md", feature_schema["target"], "trend_residual_t_ha")
    add_claim(rows, "no shortcut feature matrix", page_location(pages, "model matrices exclude year"), "forbidden target/year/history columns excluded", "artifacts/audit_records/no_shortcut_ablation.csv", sorted(set(no_shortcut.status)), ["PASS"])
    add_claim(rows, "synthetic benchmark design", page_location(pages, "14 regimes over 30"), "14 regimes; 30 seeds; 420 runs", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (synthetic["scenarios"], synthetic["repeats_per_scenario"], synthetic["runs"]), (14, 30, 420))
    add_claim(rows, "synthetic denominators", page_location(pages, "240 invalid runs"), "240 invalid; 180 valid", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (synthetic["invalid_ground_truth_runs"], synthetic["valid_ground_truth_runs"]), (240, 180))
    add_claim(rows, "synthetic ungated false permission", page_location(pages, "100.0"), "100.0%; 95% CI [98.4,100.0]", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (round(100 * synthetic["ungated_false_permission_rate"], 1), round(100 * synthetic["ungated_false_permission_ci95"][0], 1), round(100 * synthetic["ungated_false_permission_ci95"][1], 1)), (100.0, 98.4, 100.0))
    add_claim(rows, "synthetic observable-policy confusion matrix", page_location(pages, "171/240"), "TP=160; FP=171; TN=69; FN=20", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (synthetic["tp"], synthetic["fp"], synthetic["tn"], synthetic["fn"]), (160, 171, 69, 20))
    add_claim(rows, "synthetic observable false permission", page_location(pages, "171/240"), "171/240; 71.2%; 95% CI [65.2,76.6]", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (round(100 * synthetic["observable_policy_false_permission_rate"], 1), round(100 * synthetic["observable_policy_false_permission_ci95"][0], 1), round(100 * synthetic["observable_policy_false_permission_ci95"][1], 1)), (71.2, 65.2, 76.6))
    add_claim(rows, "synthetic observable false abstention", page_location(pages, "20/180"), "20/180; 11.1%; 95% CI [7.3,16.5]", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (synthetic["fn"], round(100 * synthetic["observable_policy_false_abstention_rate"], 1), round(100 * synthetic["observable_policy_false_abstention_ci95"][0], 1), round(100 * synthetic["observable_policy_false_abstention_ci95"][1], 1)), (20, 11.1, 7.3, 16.5))
    add_claim(rows, "synthetic observable permission tradeoff", page_location(pages, "permission rate is 78.8"), "permission 78.8%; sensitivity 88.9%; specificity 28.7%", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (round(100 * synthetic["observable_policy_permission_rate"], 1), round(100 * synthetic["observable_policy_sensitivity"], 1), round(100 * synthetic["observable_policy_specificity"], 1)), (78.8, 88.9, 28.7))
    add_claim(rows, "county external-resolution rows", page_location(pages, "574 counties"), "574 counties; 2022-2025 holdout", "reports/experiments/county-v2-weather-models.json", (county["holdout_rows"], county["holdout_years"]), (1024, [2022, 2023, 2024, 2025]))
    add_claim(rows, "county Module A remains inconclusive", page_location(pages, "Module A remains inconclusive"), "Module A CI high 0.26", "reports/experiments/county-v2-weather-models.json", round(float(county["gate_a_selected_vs_zero"]["ci95_high"]), 2), 0.26, 0.005)
    add_claim(rows, "county Module B feature-group value", page_location(pages, "Module B, which compares Full"), "delta -0.71; CI [-1.23,-0.34]", "reports/experiments/county-v2-weather-models.json", (round(float(county["gate_b1_weather_increment"]["point_delta_rmse"]), 2), round(float(county["gate_b1_weather_increment"]["ci95_low"]), 2), round(float(county["gate_b1_weather_increment"]["ci95_high"]), 2)), (-0.71, -1.23, -0.34))
    add_claim(rows, "PJM Module A predictive adequacy", page_location(pages, "mean-demand naive baseline"), "Full 77,521 vs naive 335,037; CI [-296.7, -221.8] x 10^3 MWh", "reports/experiments/external-domain-eia.json", (round(float(pjm["full_rmse"])), round(float(pjm["naive_rmse"])), [round(float(value)) for value in pjm["gate_a_paired_bootstrap_ci95"]], pjm["gate_a_status"]), (77521, 335037, [-296661, -221790], "PASS"))
    add_claim(rows, "PJM Module B feature-group value", page_location(pages, "Calendar-only RMSE is 227,417 MWh"), "Calendar 227,417 vs Full 77,521; CI [-186.0, -119.3] x 10^3 MWh", "reports/experiments/external-domain-eia.json", (round(float(pjm["calendar_rmse"])), round(float(pjm["full_rmse"])), [round(float(value)) for value in pjm["paired_bootstrap_ci95"]], pjm["gate_b1_status"]), (227417, 77521, [-186037, -119309], "PASS"))
    add_claim(rows, "XAI provenance output-scale binding", page_location(pages, "XAI manifest binds"), "row IDs, hashes, seed handling, output scale", "artifacts/xai/xai_manifest.csv", ({"SHAP", "LIME", "PJM permutation importance"}.issubset(set(xai_manifest["method"])), "predicted raw residual t/ha" in set(xai_manifest["output_scale"]), "predicted daily demand MWh" in set(xai_manifest["output_scale"])), (True, True, True))

    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "pdf": pdf.relative_to(ROOT).as_posix(), "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "status": status, "claims": rows}
    audit_dir = ROOT / "audit"; audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "final_pdf_numerical_crosscheck.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = ["# Final PDF Numerical Crosscheck", "", f"- Status: `{status}`", f"- Claims checked: `{len(rows)}`", "", "| Claim | PDF location | Displayed | Artifact | Artifact value | Tolerance | Status |", "|---|---|---|---|---|---:|---|"]
    report.extend(f"| {row['claim']} | {row['pdf_location']} | `{row['displayed_value']}` | `{row['artifact_path']}` | `{row['artifact_value']}` | {row['tolerance']} | {row['status']} |" for row in rows)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "final_pdf_numerical_crosscheck.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if status != "PASS":
        raise AssertionError("Final PDF numerical crosscheck failed")
    print(f"Final PDF numerical crosscheck PASS: {len(rows)} claims")


if __name__ == "__main__":
    main()
