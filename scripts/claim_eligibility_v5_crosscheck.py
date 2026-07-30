"""Cross-check V5 claim-eligibility manuscript against frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return " ".join(text.replace("\u2212", "-").split())


def add(rows: list[dict], claim: str, displayed: object, artifact: str, actual: object, expected: object) -> None:
    rows.append(
        {
            "claim": claim,
            "displayed_or_checked": displayed,
            "artifact": artifact,
            "actual": actual,
            "expected": expected,
            "status": "PASS" if actual == expected else "FAIL",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default="paper_versions/v5_claim_eligibility_audit/source/fidelity_gated_xai_method_benchmark_v3.pdf",
    )
    args = parser.parse_args()
    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        pdf = ROOT / pdf

    reader = PdfReader(str(pdf))
    text = norm("\n".join(page.extract_text() or "" for page in reader.pages))
    meta = reader.metadata or {}
    rows: list[dict] = []

    add(rows, "PDF page count", "8 pages", str(pdf), len(reader.pages), 8)
    add(rows, "PDF metadata Author", "Anonymous", str(pdf), meta.get("/Author"), "Anonymous")
    add(rows, "PDF title", "claim-eligibility title", str(pdf), meta.get("/Title"), "Claim-Eligibility Auditing for Post-hoc Explanations: Synthetic Calibration and Cross-Domain Cases")

    forbidden = [
        "fidelity-gated",
        "stop-gate",
        "claim-module",
        "pre-specified admissibility labels",
        "supplementary",
        "supplemental",
        "Table S",
        "Fig. S",
        "Algorithm 1",
        "Neither local explanations nor global/group diagnostics support",
    ]
    forbidden_hits = []
    for term in forbidden:
        if term in {"Table S", "Fig. S"}:
            if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                forbidden_hits.append(term)
        elif term.lower() in text.lower():
            forbidden_hits.append(term)
    add(rows, "Forbidden wording absent", forbidden, str(pdf), forbidden_hits, [])

    figure2 = read_json(ROOT / "artifacts/gates/figure2_three_comparisons.json")
    comparisons = {item["comparison"]: item for item in figure2["comparisons"]}
    gate_a = comparisons["extra_trees_leaf_1_weather_only_vs_zero"]
    gate_b = comparisons["extra_trees_leaf_1_full_vs_metadata_only"]
    add(rows, "Module A delta and CI", "-0.005 [-0.019, 0.009]", "artifacts/gates/figure2_three_comparisons.json", (round(gate_a["estimate"], 3), round(gate_a["ci95_low"], 3), round(gate_a["ci95_high"], 3), gate_a["n_rows"]), (-0.005, -0.019, 0.009, 333))
    add(rows, "Module B delta and CI", "-0.012 [-0.029, 0.002]", "artifacts/gates/figure2_three_comparisons.json", (round(gate_b["estimate"], 3), round(gate_b["ci95_low"], 3), round(gate_b["ci95_high"], 3), gate_b["n_rows"]), (-0.012, -0.029, 0.002, 333))

    local_frame = pd.read_csv(ROOT / "artifacts/xai/local_case_decomposition.csv")
    local = local_frame.iloc[0]
    row_groups = local_frame[local_frame["row_id"] == "Barley|Colorado|2016|spring"]
    add(rows, "Corrected Fig. 2 local case", "Barley-Colorado 2016; pred +0.209; obs -0.510", "artifacts/xai/local_case_decomposition.csv", (local["row_id"], round(float(local["predicted_residual"]), 3), round(float(local["observed_residual"]), 3)), ("Barley|Colorado|2016|spring", 0.209, -0.510))
    add(rows, "Corrected Fig. 2 displayed base", "base approximately -0.008", "artifacts/xai/local_case_decomposition.csv", round(float(local["predicted_residual"] - row_groups["signed_group_shap"].sum()), 3), -0.007)

    synth = read_json(ROOT / "artifacts/experiments/synthetic-gate-benchmark/summary.json")
    add(rows, "Synthetic benchmark dimensions", "14 regimes x 30 seeds", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (synth["scenarios"], synth["repeats_per_scenario"], synth["runs"]), (14, 30, 420))
    ungated_fp = round(synth["ungated_false_permission_rate"] * synth["invalid_ground_truth_runs"])
    add(rows, "Synthetic false permission", "240/240 ungated; 171/240 audited", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (ungated_fp, synth["invalid_ground_truth_runs"], synth["fp"], round(100 * synth["observable_policy_false_permission_rate"], 1)), (240, 240, 171, 71.2))
    add(rows, "Synthetic false abstention", "20/180 valid runs", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (synth["fn"], synth["valid_ground_truth_runs"], round(100 * synth["observable_policy_false_abstention_rate"], 1)), (20, 180, 11.1))
    add(rows, "Synthetic sensitivity/specificity", "88.9% / 28.7%", "artifacts/experiments/synthetic-gate-benchmark/summary.json", (round(100 * synth["observable_policy_sensitivity"], 1), round(100 * synth["observable_policy_specificity"], 1)), (88.9, 28.7))

    gt = pd.read_csv(ROOT / "artifacts/experiments/synthetic-gate-benchmark/synthetic_ground_truth.csv")
    permissible = sorted(gt.loc[gt["ground_truth_permission"] == True, "scenario"].tolist())
    add(rows, "Synthetic GT labels evaluation only", "six permissible regimes", "synthetic_ground_truth.csv", permissible, ["imbalanced_tail", "moderate_signal", "small_sample", "spatial_resolution_mismatch", "strong_signal", "train_only_detrending"])

    sens = read_json(ROOT / "reports/claim_eligibility_v5/sensitivity_count_v5.json")
    add(rows, "Pre-specified sensitivity count", "11 categories / 51 rows", "reports/claim_eligibility_v5/sensitivity_count_v5.json", (sens["category_count"], sens["row_count"], sens["assertion_status"]), (11, 51, "PASS"))

    state_map = pd.read_csv(ROOT / "artifacts/maps/state_level_locked_delta_rmse.csv")
    add(rows, "State map locked-row count", "state labels sum to 333", "artifacts/maps/state_level_locked_delta_rmse.csv", int(state_map["n_locked_rows"].sum()), 333)

    county = read_json(ROOT / "artifacts/experiments/county-v2-weather-models/summary.json")
    add(rows, "County external-resolution case", "A inconclusive; B passes", "artifacts/experiments/county-v2-weather-models/summary.json", (round(float(county["holdout_selected_rmse_bu_acre"]), 2), round(float(county["holdout_zero_rmse_bu_acre"]), 2), round(float(county["gate_a_selected_vs_zero"]["ci95_high"]), 2), round(float(county["gate_b1_weather_increment"]["point_delta_rmse"]), 2)), (13.47, 13.78, 0.26, -0.71))

    pjm = read_json(ROOT / "artifacts/experiments/external-domain-eia/summary.json")
    add(rows, "PJM Module A CI", "[-296.7, -221.8] x 10^3 MWh", "artifacts/experiments/external-domain-eia/summary.json", (round(float(pjm["full_rmse"])), round(float(pjm["naive_rmse"])), [round(float(v)) for v in pjm["gate_a_paired_bootstrap_ci95"]], pjm["gate_a_status"]), (77521, 335037, [-296661, -221790], "PASS"))
    add(rows, "PJM Module B CI", "[-186.0, -119.3] x 10^3 MWh", "artifacts/experiments/external-domain-eia/summary.json", (round(float(pjm["calendar_rmse"])), [round(float(v)) for v in pjm["paired_bootstrap_ci95"]], pjm["gate_b1_status"]), (227417, [-186037, -119309], "PASS"))

    required_markers = [
        "Claim-Eligibility Auditing for Post-hoc Explanations",
        "pre-specified evaluation labels",
        "GT labels define six permissible regimes and are used only for evaluation",
        "Modules A, B, and E do not pass",
        "[-296.7,-221.8]",
        "[-186.0,-119.3]",
        "MWh",
    ]
    text_for_markers = norm(text)
    marker_hits = [m for m in required_markers if norm(m) in text_for_markers]
    add(rows, "Required rendered-text markers", required_markers, str(pdf), marker_hits, required_markers)

    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf.relative_to(ROOT)),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "status": status,
        "checks": rows,
    }
    out_dir = ROOT / "reports/claim_eligibility_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v5_numerical_crosscheck.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V5 Numerical Crosscheck",
        "",
        f"- Status: `{status}`",
        f"- PDF: `{payload['pdf']}`",
        f"- SHA-256: `{payload['pdf_sha256']}`",
        f"- Checks: `{len(rows)}`",
        "",
        "| Claim | Displayed/check | Artifact | Actual | Expected | Status |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['claim']} | `{row['displayed_or_checked']}` | `{row['artifact']}` | `{row['actual']}` | `{row['expected']}` | {row['status']} |"
        for row in rows
    )
    (out_dir / "v5_numerical_crosscheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise AssertionError("V5 numerical crosscheck failed")
    print(f"V5 numerical crosscheck PASS: {len(rows)} checks")


if __name__ == "__main__":
    main()
