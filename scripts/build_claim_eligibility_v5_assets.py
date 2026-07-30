from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_GEN = ROOT / "paper" / "generated"
REPORT_DIR = ROOT / "reports" / "claim_eligibility_v5"
ARTIFACTS = ROOT / "artifacts"

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * float(value):.0f}\\%"


def tex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def build_corrected_fig2() -> dict[str, object]:
    local_path = ARTIFACTS / "xai" / "local_case_decomposition.csv"
    model_hash_path = ARTIFACTS / "xai" / "model_hashes.json"
    row_ids_path = ARTIFACTS / "xai" / "explanation_row_ids.csv"
    gates_path = ARTIFACTS / "gates" / "figure2_three_comparisons.json"
    local = pd.read_csv(local_path)
    row_id = "Barley|Colorado|2016|spring"
    case = local[local["row_id"].eq(row_id)].copy()
    if case.empty:
        raise AssertionError(f"Missing local SHAP case: {row_id}")

    expected_groups = ["heat", "drought", "frost_cold", "excess_rain", "radiation"]
    actual_groups = case["driver_group"].tolist()
    if actual_groups != expected_groups:
        raise AssertionError(f"Unexpected group order: {actual_groups}")

    predicted = float(case["predicted_residual"].iloc[0])
    observed = float(case["observed_residual"].iloc[0])
    display_base = predicted - float(case["signed_group_shap"].sum())
    if not (abs(predicted - 0.20949014122422643) < 1e-12 and abs(observed + 0.5095273846155033) < 1e-12):
        raise AssertionError("Fig. 2 local prediction/observation invariant changed")
    if not abs(display_base + 0.007052091484794798) < 1e-12:
        raise AssertionError("Fig. 2 display base invariant changed")

    gates = json.loads(gates_path.read_text(encoding="utf-8"))["comparisons"]
    gate_a = next(item for item in gates if item["gate"] == "Gate A")
    gate_b = next(item for item in gates if item["gate"] == "Gate B1 PRIMARY")
    tail = pd.read_csv(ARTIFACTS / "audit" / "tail" / "tail_metrics_by_threshold.csv").query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(ARTIFACTS / "audit_records" / "rank_null_audit.csv").query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(ARTIFACTS / "audit_records" / "topk_null_audit.csv").query("threshold == 'z<-1' and definition == 'k=10'").iloc[0]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.45, 1.0]})

    x_min, x_max = -0.56, 0.25
    y0 = 0
    ax_l.axvline(observed, color="#111111", linestyle="--", linewidth=1.2)
    ax_l.axvline(predicted, color="#111111", linestyle="-", linewidth=1.2)
    ax_l.axvspan(observed, predicted, color="#d9d9d9", alpha=0.45, zorder=0)
    ax_l.text(observed, 5.75, "observed\n-0.510", ha="center", va="bottom", fontsize=7.2)
    ax_l.text(predicted, 5.75, "prediction\n+0.209", ha="center", va="bottom", fontsize=7.2)
    ax_l.scatter([display_base], [y0], marker="o", s=26, color="#444444", zorder=3)
    ax_l.text(display_base, y0 - 0.33, "base -0.007", ha="center", va="top", fontsize=7.0)

    running = display_base
    labels = {
        "heat": "heat",
        "drought": "drought",
        "frost_cold": "frost/cold",
        "excess_rain": "excess rain",
        "radiation": "radiation",
    }
    for idx, row in enumerate(case.itertuples(), start=1):
        value = float(row.signed_group_shap)
        new = running + value
        color = "#ffffff" if value >= 0 else "#999999"
        edge = "#222222"
        hatch = "" if value >= 0 else "///"
        left = min(running, new)
        ax_l.barh(idx, abs(value), left=left, height=0.52, color=color, edgecolor=edge, hatch=hatch, linewidth=0.9)
        ax_l.plot([running, running], [idx - 0.26, idx + 0.26], color=edge, linewidth=0.8)
        ax_l.plot([new, new], [idx - 0.26, idx + 0.26], color=edge, linewidth=0.8)
        ax_l.text(new + (0.012 if value >= 0 else -0.012), idx, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=7.2)
        running = new
    ax_l.scatter([predicted], [6], marker="s", s=28, color="#222222", zorder=3)
    ax_l.set_yticks(range(1, 6), [labels[x] for x in case["driver_group"]], fontsize=7.5)
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(-0.6, 6.4)
    ax_l.set_xlabel("raw residual t ha$^{-1}$", fontsize=8)
    ax_l.set_title("A. Coherent explanation, wrong event", loc="left", fontsize=9)
    ax_l.tick_params(axis="x", labelsize=7.4)
    ax_l.spines[["top", "right", "left"]].set_visible(False)
    ax_l.tick_params(axis="y", length=0)

    ax_r.set_axis_off()
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    rows = [
        ("Ungated XAI", "PERMIT\nweather attribution", "X", 0.74),
        ("Claim-eligibility audit", "ABSTAIN\nclaim not eligible", "A/B/E", 0.38),
    ]
    for title, verdict, marker, y in rows:
        ax_r.add_patch(plt.Rectangle((0.04, y - 0.12), 0.92, 0.24, fill=False, edgecolor="#222222", linewidth=1.0))
        ax_r.text(0.08, y + 0.055, title, fontsize=8.0, fontweight="bold", va="center")
        ax_r.text(0.08, y - 0.045, verdict, fontsize=8.0, va="center")
        ax_r.text(0.79, y, marker, fontsize=8.0, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#eeeeee", ec="#333333"))
    module_text = (
        f"A: {gate_a['estimate']:.3f} [{gate_a['ci95_low']:.3f}, {gate_a['ci95_high']:.3f}]\n"
        f"B: {gate_b['estimate']:.3f} [{gate_b['ci95_low']:.3f}, {gate_b['ci95_high']:.3f}]\n"
        f"E: RMSE {tail.paired_delta_rmse:.3f}; rank {rank.spearman:.3f}; top-10 {int(topk.overlap)}/{int(topk.k)}"
    )
    ax_r.text(0.04, 0.11, module_text, fontsize=7.0, va="bottom")
    ax_r.set_title("B. Study-level verdict", loc="left", fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(PAPER_GEN / f"figure_xai_claim_eligibility_v5.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    provenance = {
        "artifact_id": "figure_xai_claim_eligibility_v5",
        "row_id": row_id,
        "model_family": "ExtraTrees",
        "configuration_id": "extra_trees_leaf_1",
        "feature_family": "weather_only",
        "split_id": "locked_2016_2025",
        "seed_or_seed_aggregation": "five selected seeds aggregated by mean prediction where applicable",
        "display_base": display_base,
        "predicted_residual": predicted,
        "observed_residual": observed,
        "groups": case[["driver_group", "signed_group_shap"]].to_dict(orient="records"),
        "local_case_source": str(local_path.relative_to(ROOT)),
        "local_case_sha256": sha256(local_path),
        "row_id_source": str(row_ids_path.relative_to(ROOT)),
        "row_id_source_sha256": sha256(row_ids_path),
        "model_hash_source": str(model_hash_path.relative_to(ROOT)),
        "model_hash_source_sha256": sha256(model_hash_path),
        "gate_source": str(gates_path.relative_to(ROOT)),
        "gate_source_sha256": sha256(gates_path),
        "assertion_status": "PASS",
    }
    return provenance


def build_synthetic_dumbbell() -> dict[str, object]:
    source_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "scenario_level_decisions.csv"
    summary_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "summary.json"
    df = pd.read_csv(source_path)
    invalid = df[~df["ground_truth_permission"].astype(bool)].copy()
    order = [
        "leakage",
        "omitted_confounder",
        "correlated_features",
        "geographic_shift",
        "temporal_drift",
        "measurement_error",
        "weak_signal",
        "no_signal",
    ]
    labels = {
        "leakage": "Leakage",
        "omitted_confounder": "Omitted conf.",
        "correlated_features": "Corr. feat.",
        "geographic_shift": "Geo. shift",
        "temporal_drift": "Temporal drift",
        "measurement_error": "Meas. error",
        "weak_signal": "Weak signal",
        "no_signal": "No signal",
    }
    invalid["order"] = invalid["scenario"].map({name: i for i, name in enumerate(order)})
    invalid = invalid.sort_values("order")
    if invalid["scenario"].tolist() != order:
        raise AssertionError(f"Unexpected invalid scenario set: {invalid['scenario'].tolist()}")

    y = np.arange(len(invalid))
    audited = invalid["policy_permit_rate"].to_numpy(dtype=float) * 100
    ungated = np.full_like(audited, 100.0)

    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    for yi, a, u in zip(y, audited, ungated):
        ax.plot([a, u], [yi, yi], color="#666666", linewidth=0.9, linestyle="-")
    ax.scatter(ungated, y, marker="o", facecolor="white", edgecolor="#111111", s=30, label="Ungated")
    ax.scatter(audited, y, marker="s", facecolor="#444444", edgecolor="#111111", s=28, label="Audit")
    ax.set_yticks(y, [labels[s] for s in invalid["scenario"]], fontsize=7.3)
    ax.invert_yaxis()
    ax.set_xlim(-3, 104)
    ax.set_xlabel("false-permission rate (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.4)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    ax.set_title("Synthetic invalid regimes", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(PAPER_GEN / f"figure_synthetic_dumbbell_v5.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["fp"] != 171 or summary["invalid_ground_truth_runs"] != 240:
        raise AssertionError("Synthetic false-permission invariant changed")
    return {
        "artifact_id": "figure_synthetic_dumbbell_v5",
        "source": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256(source_path),
        "summary_source": str(summary_path.relative_to(ROOT)),
        "summary_source_sha256": sha256(summary_path),
        "invalid_regimes": invalid[["scenario", "policy_permit_rate", "false_permission_count"]].to_dict(orient="records"),
        "ungated_false_permission_rate": 1.0,
        "observable_false_permissions": "171/240",
        "specificity": 0.2875,
        "assertion_status": "PASS",
    }


def build_module_table() -> dict[str, object]:
    gates = json.loads((ARTIFACTS / "gates" / "figure2_three_comparisons.json").read_text(encoding="utf-8"))["comparisons"]
    gate_a = next(item for item in gates if item["gate"] == "Gate A")
    gate_b = next(item for item in gates if item["gate"] == "Gate B1 PRIMARY")
    tail = pd.read_csv(ARTIFACTS / "audit" / "tail" / "tail_metrics_by_threshold.csv").query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(ARTIFACTS / "audit_records" / "rank_null_audit.csv").query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(ARTIFACTS / "audit_records" / "topk_null_audit.csv").query("threshold == 'z<-1' and definition == 'k=10'").iloc[0]
    rows = [
        ("A", "Selected Weather-only vs zero", "upper paired CI $<0$",
         f"{fmt(gate_a['estimate'])} [{fmt(gate_a['ci95_low'])}, {fmt(gate_a['ci95_high'])}]",
         "DOES NOT PASS", "model description only"),
        ("B", "Full vs Metadata-only", "upper paired CI $<0$",
         f"{fmt(gate_b['estimate'])} [{fmt(gate_b['ci95_low'])}, {fmt(gate_b['ci95_high'])}]",
         "DOES NOT PASS", "no weather-specific reliance claim"),
        ("E", "Primary tail error + rank + top-$k$", "all checks pass",
         f"RMSE {fmt(tail.paired_delta_rmse)} [{fmt(tail.paired_delta_rmse_ci95_low)}, {fmt(tail.paired_delta_rmse_ci95_high)}]; rank {fmt(rank.spearman)}; top-10 {int(topk.overlap)}/{int(topk.k)}",
         "DOES NOT PASS", "no event-recovery claim"),
    ]
    lines = [
        "\\begin{tabular}{p{0.035\\textwidth}p{0.18\\textwidth}p{0.16\\textwidth}p{0.24\\textwidth}p{0.13\\textwidth}p{0.17\\textwidth}}",
        "\\toprule",
        "Mod. & Question/contrast & Pass rule & Current estimate & V5 verdict & Claim consequence \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])
    out = PAPER_GEN / "table_claim_eligibility_modules_v5.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return {
        "artifact_id": "table_claim_eligibility_modules_v5",
        "output": str(out.relative_to(ROOT)),
        "output_sha256": sha256(out),
        "assertion_status": "PASS",
    }


def build_synthetic_label_check() -> dict[str, object]:
    gt_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "synthetic_ground_truth.csv"
    gt = pd.read_csv(gt_path)
    permissible = {
        "moderate_signal",
        "strong_signal",
        "train_only_detrending",
        "small_sample",
        "imbalanced_tail",
        "spatial_resolution_mismatch",
    }
    actual = set(gt.loc[gt["ground_truth_permission"].astype(bool), "scenario"])
    if actual != permissible:
        raise AssertionError(f"Synthetic permissible labels do not match definition: {sorted(actual)}")
    return {
        "artifact_id": "synthetic_gt_label_definition_check",
        "source": str(gt_path.relative_to(ROOT)),
        "source_sha256": sha256(gt_path),
        "permissible_regimes": sorted(actual),
        "impermissible_regimes": sorted(set(gt["scenario"]) - actual),
        "definition": "permissible when requested feature-group/event claim is present in the DGP and conceptually valid; impermissible when signal is absent or design makes interpretation structurally invalid",
        "assertion_status": "PASS",
    }


def build_sensitivity_count() -> dict[str, object]:
    sources = {
        "history thresholds": ARTIFACTS / "audit_records" / "min_history_sensitivity.csv",
        "target scale": ARTIFACTS / "audit_records" / "target_scale_sensitivity.csv",
        "detrending variants": ARTIFACTS / "audit_records" / "alternative_detrending_sensitivity.csv",
        "expanded model families": ARTIFACTS / "audit_records" / "expanded_model_baselines.csv",
        "resampling schemes": ARTIFACTS / "audit_records" / "bootstrap_scheme_comparison.csv",
        "temporal-stage features": ARTIFACTS / "audit" / "stage_features" / "stage_feature_sensitivity.csv",
        "crop checks": ARTIFACTS / "audit" / "crop" / "crop_specific_metrics.csv",
        "spatial checks": ARTIFACTS / "audit" / "spatial" / "leave_one_state_out.csv",
        "temporal rolling/prefix analyses": ARTIFACTS / "audit_records" / "temporal_and_capacity_audits.csv",
        "retrospective target": ARTIFACTS / "audit_records" / "retrospective_target_comparison.csv",
        "state heterogeneity": ARTIFACTS / "maps" / "state_level_locked_delta_rmse.csv",
    }
    categories = []
    total_rows = 0
    for label, path in sources.items():
        frame = pd.read_csv(path)
        count = len(frame)
        categories.append({"category": label, "source": str(path.relative_to(ROOT)), "rows": count, "sha256": sha256(path)})
        total_rows += count
    out = {
        "artifact_id": "prespecified_sensitivity_count_v5",
        "categories": categories,
        "category_count": len(categories),
        "row_count": total_rows,
        "statement": f"{len(categories)} pre-specified sensitivity categories covering {total_rows} recorded rows",
        "assertion_status": "PASS",
    }
    (REPORT_DIR / "sensitivity_count_v5.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        build_corrected_fig2(),
        build_synthetic_dumbbell(),
        build_module_table(),
        build_synthetic_label_check(),
        build_sensitivity_count(),
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    (REPORT_DIR / "v5_generated_artifact_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Built V5 claim-eligibility assets")


if __name__ == "__main__":
    main()
