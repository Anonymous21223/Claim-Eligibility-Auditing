from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_GEN = ROOT / "paper" / "generated"
REPORT_DIR = ROOT / "reports" / "claim_eligibility_v5_1"
ARTIFACTS = ROOT / "artifacts"

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * float(value):.0f}\\%"


def tex_escape(value: object) -> str:
    text = str(value).replace("_", " ")
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def provenance_entry(
    artifact_id: str,
    manuscript_location: str,
    model_family: str,
    feature_family: str,
    configuration_id: str,
    split_id: str,
    row_id_or_population_hash: str,
    seed_or_seed_aggregation: str,
    prediction_hash: str,
    target_hash: str,
    source_metric_file: Path,
    generation_script: str,
    expected_unrounded_value: object,
    displayed_value: object,
    rounding_rule: str,
    assertion_status: str = "PASS",
    **extra: object,
) -> dict[str, object]:
    item = {
        "artifact_id": artifact_id,
        "manuscript_location": manuscript_location,
        "model_family": model_family,
        "feature_family": feature_family,
        "configuration_id": configuration_id,
        "split_id": split_id,
        "row_id_or_population_hash": row_id_or_population_hash,
        "seed_or_seed_aggregation": seed_or_seed_aggregation,
        "prediction_hash": prediction_hash,
        "target_hash": target_hash,
        "source_metric_file": rel(source_metric_file),
        "source_metric_sha256": sha256(source_metric_file),
        "generation_script": generation_script,
        "expected_unrounded_value": expected_unrounded_value,
        "displayed_value": displayed_value,
        "rounding_rule": rounding_rule,
        "assertion_status": assertion_status,
    }
    item.update(extra)
    return item


def rounded_sum_assertion(values: list[float], displayed_prediction: float, digits: int = 3) -> dict[str, object]:
    rounded_values = [round(v, digits) for v in values]
    rounded_prediction = round(displayed_prediction, digits)
    displayed_sum = round(sum(rounded_values), digits)
    tolerance = 0.5 * (10 ** -digits) * len(values)
    passed = abs(displayed_sum - rounded_prediction) <= tolerance
    return {
        "rounded_values": rounded_values,
        "rounded_sum": displayed_sum,
        "rounded_prediction": rounded_prediction,
        "tolerance": tolerance,
        "assertion_status": "PASS" if passed else "FAIL",
    }


def box(ax, x: float, y: float, w: float, h: float, text: str, fs: float = 7.0, dashed: bool = False) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            fc="white",
            ec="#111111",
            linewidth=0.8,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def diamond(ax, cx: float, cy: float, w: float, h: float, text: str, fs: float = 6.7) -> None:
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec="#111111", linewidth=0.8))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1: float, y1: float, x2: float, y2: float, label: str = "", fs: float = 6.3) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.8, color="#111111"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label, ha="center", va="bottom", fontsize=fs)


def build_workflow() -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    top = [
        (0.04, "Raw snapshots\n+ hashes"),
        (0.23, "Train-only target,\nscale, preprocessing"),
        (0.42, "Validation model\nand feature selection"),
        (0.61, "Lock config\n+ claim type"),
        (0.80, "Predict locked\nfuture rows"),
    ]
    for x, text in top:
        box(ax, x, 0.80, 0.14, 0.10, text, fs=6.8)
    for x1, x2 in [(0.18, 0.23), (0.37, 0.42), (0.56, 0.61), (0.75, 0.80)]:
        arrow(ax, x1, 0.85, x2, 0.85)

    diamond(ax, 0.48, 0.58, 0.19, 0.14, "A\nOverall\nadequacy", fs=5.9)
    diamond(ax, 0.48, 0.38, 0.19, 0.14, "B\nIncremental\nvalue", fs=5.9)
    diamond(ax, 0.48, 0.19, 0.19, 0.14, "E\nEvent recovery\nif claimed", fs=5.7)
    diamond(ax, 0.86, 0.58, 0.13, 0.10, "D\nDiagnostic", fs=6.8)
    box(ax, 0.69, 0.31, 0.24, 0.12, "AUTHORIZED\nPREDICTIVE-RELIANCE\nCLAIM", fs=6.8)
    box(ax, 0.69, 0.12, 0.24, 0.12, "AUTHORIZED\nEVENT-RECOVERY\nCLAIM", fs=6.8)
    box(ax, 0.05, 0.52, 0.25, 0.12, "DOES NOT PASS:\nMODEL DESCRIPTION\nONLY", fs=6.4)
    box(ax, 0.05, 0.32, 0.25, 0.12, "DOES NOT PASS:\nNO WEATHER-SPECIFIC\nCLAIM", fs=6.2)
    box(ax, 0.05, 0.13, 0.25, 0.12, "DOES NOT PASS:\nNO EVENT-RECOVERY\nCLAIM", fs=6.2)

    arrow(ax, 0.87, 0.80, 0.87, 0.64)
    arrow(ax, 0.87, 0.80, 0.54, 0.62)
    arrow(ax, 0.48, 0.51, 0.48, 0.45)
    arrow(ax, 0.39, 0.58, 0.30, 0.58)
    arrow(ax, 0.48, 0.31, 0.48, 0.26)
    arrow(ax, 0.39, 0.38, 0.30, 0.38)
    arrow(ax, 0.57, 0.38, 0.69, 0.37, "PASS")
    arrow(ax, 0.57, 0.19, 0.69, 0.18, "PASS")
    arrow(ax, 0.39, 0.19, 0.30, 0.19)
    ax.plot([0.86, 0.86], [0.53, 0.44], color="#111111", lw=0.8, ls="--")

    out_pdf = PAPER_GEN / "figure_workflow_us_v5_1.pdf"
    out_png = PAPER_GEN / "figure_workflow_us_v5_1.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return provenance_entry(
        "fig1_workflow_v5_1",
        "Fig. 1",
        "not_applicable",
        "not_applicable",
        "claim_eligibility_workflow_v5_1",
        "not_applicable",
        "not_applicable",
        "not_applicable",
        "not_applicable",
        "not_applicable",
        out_pdf,
        "scripts/build_claim_eligibility_v5_1_assets.py::build_workflow",
        "workflow labels include PASS/DOES NOT PASS; Module D outside path",
        "figure_workflow_us_v5_1.pdf",
        "programmatic vector figure; no scientific rounding",
    )


def build_fig2() -> dict[str, object]:
    local_path = ARTIFACTS / "xai" / "local_case_decomposition.csv"
    gates_path = ARTIFACTS / "gates" / "figure2_three_comparisons.json"
    shap_config_path = ARTIFACTS / "xai" / "shap_config.yaml"
    model_hash_path = ARTIFACTS / "xai" / "model_hashes.json"
    row_ids_path = ARTIFACTS / "xai" / "explanation_row_ids.csv"
    local = pd.read_csv(local_path)
    row_id = "Barley|Colorado|2016|spring"
    case = local[local["row_id"].eq(row_id)].copy()
    expected_groups = ["heat", "drought", "frost_cold", "excess_rain", "radiation"]
    if case["driver_group"].tolist() != expected_groups:
        raise AssertionError("Unexpected Fig. 2 SHAP group order")

    predicted = float(case["predicted_residual"].iloc[0])
    observed = float(case["observed_residual"].iloc[0])
    exact_base = float(case["base_value"].iloc[0])
    shown_sum = float(case["signed_group_shap"].sum())
    other_remainder = predicted - exact_base - shown_sum
    reconstructed = exact_base + shown_sum + other_remainder
    if abs(predicted - 0.20949014122422643) > 1e-12 or abs(observed + 0.5095273846155033) > 1e-12:
        raise AssertionError("Fig. 2 prediction/observation invariant changed")
    if abs(reconstructed - predicted) > 1e-12:
        raise AssertionError("Fig. 2 base + grouped + remainder does not reconstruct prediction")

    values = [exact_base] + case["signed_group_shap"].astype(float).tolist() + [other_remainder]
    display_assert = rounded_sum_assertion(values, predicted, digits=3)
    if display_assert["assertion_status"] != "PASS":
        raise AssertionError("Fig. 2 rounded display arithmetic failed")

    gates = json.loads(gates_path.read_text(encoding="utf-8"))["comparisons"]
    gate_a = next(item for item in gates if item["gate"] == "Gate A")
    gate_b = next(item for item in gates if item["gate"] == "Gate B1 PRIMARY")
    tail = pd.read_csv(ARTIFACTS / "audit" / "tail" / "tail_metrics_by_threshold.csv").query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(ARTIFACTS / "audit_records" / "rank_null_audit.csv").query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(ARTIFACTS / "audit_records" / "topk_null_audit.csv").query("threshold == 'z<-1' and definition == 'k=10'").iloc[0]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.45, 1.0]})
    x_min, x_max = -0.56, 0.25
    ax_l.axvline(observed, color="#111111", linestyle="--", linewidth=1.2)
    ax_l.axvline(predicted, color="#111111", linestyle="-", linewidth=1.2)
    ax_l.axvspan(observed, predicted, color="#d9d9d9", alpha=0.45, zorder=0)
    ax_l.text(observed, 6.85, "observed\n-0.510", ha="center", va="bottom", fontsize=7.0)
    ax_l.text(predicted, 6.85, "prediction\n+0.209", ha="center", va="bottom", fontsize=7.0)
    ax_l.scatter([exact_base], [0], marker="o", s=24, color="#444444", zorder=3)
    ax_l.text(exact_base, -0.28, "base +0.000", ha="center", va="top", fontsize=6.8)

    display_rows = case[["driver_group", "signed_group_shap"]].values.tolist()
    display_rows.append(["other/remainder", other_remainder])
    labels = {
        "heat": "heat",
        "drought": "drought",
        "frost_cold": "frost/cold",
        "excess_rain": "excess rain",
        "radiation": "radiation",
        "other/remainder": "other/remainder",
    }
    running = exact_base
    for idx, (name, value) in enumerate(display_rows, start=1):
        value = float(value)
        new = running + value
        left = min(running, new)
        color = "#ffffff" if value >= 0 else "#999999"
        hatch = "" if value >= 0 else "///"
        ax_l.barh(idx, abs(value), left=left, height=0.48, color=color, edgecolor="#222222", hatch=hatch, linewidth=0.8)
        ax_l.plot([running, running], [idx - 0.24, idx + 0.24], color="#222222", linewidth=0.7)
        ax_l.plot([new, new], [idx - 0.24, idx + 0.24], color="#222222", linewidth=0.7)
        ax_l.text(new + (0.012 if value >= 0 else -0.012), idx, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=6.9)
        running = new
    ax_l.scatter([predicted], [7], marker="s", s=26, color="#222222", zorder=3)
    ax_l.set_yticks(range(1, 7), [labels[x[0]] for x in display_rows], fontsize=7.1)
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(-0.6, 7.35)
    ax_l.set_xlabel("raw residual t ha$^{-1}$", fontsize=8)
    ax_l.set_title("A. Coherent explanation, wrong event", loc="left", fontsize=9)
    ax_l.tick_params(axis="x", labelsize=7.2)
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
        ax_r.text(0.79, y, marker, fontsize=8.0, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="#eeeeee", ec="#333333"))
    module_text = (
        f"A: {gate_a['estimate']:.3f} [{gate_a['ci95_low']:.3f}, {gate_a['ci95_high']:.3f}]\n"
        f"B: {gate_b['estimate']:.3f} [{gate_b['ci95_low']:.3f}, {gate_b['ci95_high']:.3f}]\n"
        f"E: RMSE {tail.paired_delta_rmse:.3f}; rank {rank.spearman:.3f}; top-10 {int(topk.overlap)}/{int(topk.k)}"
    )
    ax_r.text(0.04, 0.11, module_text, fontsize=7.0, va="bottom")
    ax_r.set_title("B. Study-level verdict", loc="left", fontsize=9)

    out_pdf = PAPER_GEN / "figure_xai_claim_eligibility_v5_1.pdf"
    out_png = PAPER_GEN / "figure_xai_claim_eligibility_v5_1.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    row_hash = gate_a["row_id_sha256"]
    target_hash = gate_a["target_sha256"]
    prediction_hash = json.loads(model_hash_path.read_text(encoding="utf-8"))["seed_aggregated_predictions_csv"]
    return provenance_entry(
        "fig2_xai_claim_eligibility_v5_1",
        "Fig. 2",
        "ExtraTrees",
        "weather_only",
        "extra_trees_leaf_1",
        "locked_2016_2025",
        row_hash,
        "five selected seeds aggregated by mean prediction where applicable",
        prediction_hash,
        target_hash,
        local_path,
        "scripts/build_claim_eligibility_v5_1_assets.py::build_fig2",
        {
            "exact_shap_base_value": exact_base,
            "grouped_weather_sum": shown_sum,
            "other_remainder": other_remainder,
            "prediction": predicted,
            "observed": observed,
        },
        {
            "base": "+0.000",
            "prediction": "+0.209",
            "observed": "-0.510",
            "other_remainder": f"{other_remainder:+.3f}",
        },
        "round displayed numeric labels to three decimals; rounded-sum tolerance is half-unit times number of displayed components",
        shap_config=shap_config_path.read_text(encoding="utf-8"),
        shap_config_sha256=sha256(shap_config_path),
        model_hash_source=rel(model_hash_path),
        row_id_source=rel(row_ids_path),
        arithmetic_assertion=display_assert,
        output_pdf=rel(out_pdf),
        output_pdf_sha256=sha256(out_pdf),
    )


def build_synthetic_outputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "scenario_level_decisions.csv"
    gt_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "synthetic_ground_truth.csv"
    summary_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "summary.json"
    df = pd.read_csv(source_path)
    gt = pd.read_csv(gt_path)
    merged = df.merge(gt[["scenario", "effect", "ground_truth_permission", "reason"]], on=["scenario", "ground_truth_permission"], how="left", suffixes=("", "_gt"))
    if len(merged) != 14 or merged["reason_gt"].isna().any():
        raise AssertionError("Synthetic GT audit merge failed")

    permissible = set(merged.loc[merged["ground_truth_permission"].astype(bool), "scenario"])
    expected_permissible = {
        "moderate_signal",
        "strong_signal",
        "train_only_detrending",
        "small_sample",
        "imbalanced_tail",
        "spatial_resolution_mismatch",
    }
    if permissible != expected_permissible:
        raise AssertionError(f"GT label conflict: {sorted(permissible)}")

    rationale_map = {
        "moderate_signal": "permissible: requested event claim is present in the DGP with a moderate nonzero effect.",
        "strong_signal": "permissible: requested event claim is present in the DGP with a strong nonzero effect.",
        "train_only_detrending": "permissible: requested signal is present and target construction remains conceptually valid under train-only detrending.",
        "small_sample": "permissible: signal is present and conceptually valid; limited sample size makes estimation difficult but not structurally invalid.",
        "imbalanced_tail": "permissible: requested event signal is present; tail imbalance makes recovery difficult but does not invalidate the claim.",
        "spatial_resolution_mismatch": "permissible: signal is present and the requested claim remains conceptually valid; representation mismatch makes estimation difficult.",
        "weak_signal": "impermissible: requested event signal is too weak to support the requested claim under the benchmark definition.",
        "no_signal": "impermissible: requested signal is absent in the data-generating process.",
        "correlated_features": "impermissible: correlated-feature structure makes the requested feature-group interpretation structurally invalid.",
        "geographic_shift": "impermissible: distribution shift makes the locked interpretation structurally invalid.",
        "leakage": "impermissible: leakage makes predictive evidence invalid for the requested interpretation.",
        "measurement_error": "impermissible: measurement mechanism invalidates the requested feature-group/event interpretation.",
        "omitted_confounder": "impermissible: omitted confounding makes the requested interpretation structurally invalid.",
        "temporal_drift": "impermissible: temporal drift violates the requested locked-period interpretation.",
    }
    merged["label"] = np.where(merged["ground_truth_permission"].astype(bool), "permissible", "impermissible")
    merged["rationale"] = merged["scenario"].map(rationale_map)
    if merged["rationale"].isna().any():
        raise AssertionError("Missing synthetic rationale")

    gt_audit = {
        "artifact_id": "synthetic_gt_label_audit_v5_1",
        "definition": {
            "permissible": "requested feature-group or event claim is present in the data-generating process and remains conceptually valid, even if estimation is difficult because of limited sample size or representation mismatch",
            "impermissible": "requested signal is absent or the data-generating or study-design mechanism makes that interpretation structurally invalid",
            "usage": "labels are used only to score the observable audit and are never inputs to Modules A, B, or E",
        },
        "source_configuration": rel(gt_path),
        "source_configuration_sha256": sha256(gt_path),
        "scenario_decision_source": rel(source_path),
        "scenario_decision_sha256": sha256(source_path),
        "records": merged[["scenario", "label", "ground_truth_permission", "requested_claim_level", "effect", "rationale", "reason", "module_a_pass_rate", "module_b_pass_rate", "module_e_pass_rate", "policy_permit_rate"]].to_dict(orient="records"),
        "assertion_status": "PASS",
    }
    (REPORT_DIR / "synthetic_gt_label_audit_v5_1.json").write_text(json.dumps(gt_audit, indent=2) + "\n", encoding="utf-8")

    order = [
        "correlated_features",
        "geographic_shift",
        "imbalanced_tail",
        "leakage",
        "measurement_error",
        "moderate_signal",
        "no_signal",
        "omitted_confounder",
        "small_sample",
        "spatial_resolution_mismatch",
        "strong_signal",
        "temporal_drift",
        "train_only_detrending",
        "weak_signal",
    ]
    label_map = {
        "correlated_features": "Corr. feat.",
        "geographic_shift": "Geo. shift",
        "imbalanced_tail": "Imbalanced tail",
        "leakage": "Leakage",
        "measurement_error": "Meas. error",
        "moderate_signal": "Moderate signal",
        "no_signal": "No signal",
        "omitted_confounder": "Omitted conf.",
        "small_sample": "Small sample",
        "spatial_resolution_mismatch": "Spatial mismatch",
        "strong_signal": "Strong signal",
        "temporal_drift": "Temporal drift",
        "train_only_detrending": "Train-only detr.",
        "weak_signal": "Weak signal",
    }
    table_df = df.set_index("scenario").loc[order].reset_index()
    lines = [
        "\\begin{tabular}{@{}p{0.29\\columnwidth}cccccc@{}}",
        "\\toprule",
        "Scenario & GT & Claim & A & B & E & Rule \\\\",
        "\\midrule",
    ]
    for row in table_df.itertuples():
        cells = [
            tex_escape(label_map[row.scenario]),
            "yes" if bool(row.ground_truth_permission) else "no",
            "event",
            pct(row.module_a_pass_rate),
            pct(row.module_b_pass_rate),
            pct(row.module_e_pass_rate),
            pct(row.policy_permit_rate),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_table = PAPER_GEN / "table_synthetic_scenario_decisions_v5_1.tex"
    out_table.write_text("\n".join(lines), encoding="utf-8")

    invalid = df[~df["ground_truth_permission"].astype(bool)].copy()
    dumbbell_order = ["leakage", "omitted_confounder", "correlated_features", "geographic_shift", "temporal_drift", "measurement_error", "weak_signal", "no_signal"]
    invalid["order"] = invalid["scenario"].map({name: i for i, name in enumerate(dumbbell_order)})
    invalid = invalid.sort_values("order")
    y = np.arange(len(invalid))
    audited = invalid["policy_permit_rate"].to_numpy(dtype=float) * 100
    ungated = np.full_like(audited, 100.0)
    dumb_labels = {
        "leakage": "Leakage",
        "omitted_confounder": "Omitted conf.",
        "correlated_features": "Corr. feat.",
        "geographic_shift": "Geo. shift",
        "temporal_drift": "Temporal drift",
        "measurement_error": "Meas. error",
        "weak_signal": "Weak signal",
        "no_signal": "No signal",
    }
    fig, ax = plt.subplots(figsize=(3.55, 2.85))
    for yi, a, u in zip(y, audited, ungated):
        ax.plot([a, u], [yi, yi], color="#666666", linewidth=0.9)
    ax.scatter(ungated, y, marker="o", facecolor="white", edgecolor="#111111", s=30, label="Ungated")
    ax.scatter(audited, y, marker="s", facecolor="#444444", edgecolor="#111111", s=28, label="Audit")
    ax.set_yticks(y, [dumb_labels[s] for s in invalid["scenario"]], fontsize=7.3)
    ax.invert_yaxis()
    ax.set_xlim(-3, 104)
    ax.set_xlabel("false-permission rate (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.4)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    ax.set_title("Synthetic invalid regimes", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    out_pdf = PAPER_GEN / "figure_synthetic_dumbbell_v5_1.pdf"
    out_png = PAPER_GEN / "figure_synthetic_dumbbell_v5_1.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["fp"], summary["invalid_ground_truth_runs"], summary["fn"], summary["valid_ground_truth_runs"]) != (171, 240, 20, 180):
        raise AssertionError("Synthetic summary invariant changed")

    dumbbell_prov = provenance_entry(
        "fig3_synthetic_dumbbell_v5_1",
        "Fig. 3",
        "synthetic_generator",
        "observable_modules_A_B_E",
        "synthetic_gate_benchmark",
        "30_seeds_per_regime",
        sha256(source_path),
        "30 seeds per scenario",
        "not_applicable",
        sha256(gt_path),
        source_path,
        "scripts/build_claim_eligibility_v5_1_assets.py::build_synthetic_outputs",
        {"ungated_false_permission_rate": 1.0, "observable_false_permissions": "171/240"},
        "dumbbell invalid-regime false-permission rates",
        "percentages rounded to nearest integer in figure axes",
        output_pdf=rel(out_pdf),
        output_pdf_sha256=sha256(out_pdf),
    )
    table_prov = provenance_entry(
        "table_synthetic_14_regimes_v5_1",
        "Table III",
        "synthetic_generator",
        "observable_modules_A_B_E",
        "synthetic_gate_benchmark",
        "30_seeds_per_regime",
        sha256(source_path),
        "30 seeds per scenario",
        "not_applicable",
        sha256(gt_path),
        source_metric_file=source_path,
        generation_script="scripts/build_claim_eligibility_v5_1_assets.py::build_synthetic_outputs",
        expected_unrounded_value=table_df[["scenario", "module_a_pass_rate", "module_b_pass_rate", "module_e_pass_rate", "policy_permit_rate"]].to_dict(orient="records"),
        displayed_value=rel(out_table),
        rounding_rule="rates displayed as whole percentages from exact pass-rate columns",
        output_tex_sha256=sha256(out_table),
        shares_source_with_dumbbell=True,
    )
    return gt_audit, dumbbell_prov, table_prov


def build_module_table() -> dict[str, object]:
    gates_path = ARTIFACTS / "gates" / "figure2_three_comparisons.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))["comparisons"]
    gate_a = next(item for item in gates if item["gate"] == "Gate A")
    gate_b = next(item for item in gates if item["gate"] == "Gate B1 PRIMARY")
    tail = pd.read_csv(ARTIFACTS / "audit" / "tail" / "tail_metrics_by_threshold.csv").query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(ARTIFACTS / "audit_records" / "rank_null_audit.csv").query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(ARTIFACTS / "audit_records" / "topk_null_audit.csv").query("threshold == 'z<-1' and definition == 'k=10'").iloc[0]
    rows = [
        ("A", "Selected Weather-only vs zero", "upper paired CI $<0$", f"{fmt(gate_a['estimate'])} [{fmt(gate_a['ci95_low'])}, {fmt(gate_a['ci95_high'])}]", "DOES NOT PASS", "model description only"),
        ("B", "Full vs Metadata-only", "upper paired CI $<0$", f"{fmt(gate_b['estimate'])} [{fmt(gate_b['ci95_low'])}, {fmt(gate_b['ci95_high'])}]", "DOES NOT PASS", "no weather-specific reliance claim"),
        ("E", "Primary tail error + rank + top-$k$", "all checks pass", f"RMSE {fmt(tail.paired_delta_rmse)} [{fmt(tail.paired_delta_rmse_ci95_low)}, {fmt(tail.paired_delta_rmse_ci95_high)}]; rank {fmt(rank.spearman)}; top-10 {int(topk.overlap)}/{int(topk.k)}", "DOES NOT PASS", "no event-recovery claim"),
    ]
    lines = [
        "\\begin{tabular}{p{0.035\\textwidth}p{0.18\\textwidth}p{0.16\\textwidth}p{0.24\\textwidth}p{0.13\\textwidth}p{0.17\\textwidth}}",
        "\\toprule",
        "Mod. & Question/contrast & Pass rule & Current estimate & Verdict & Claim consequence \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out = PAPER_GEN / "table_claim_eligibility_modules_v5_1.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return provenance_entry(
        "table1_claim_eligibility_modules_v5_1",
        "Table I",
        "ExtraTrees",
        "weather_only/full/metadata_only",
        "extra_trees_leaf_1",
        "locked_2016_2025",
        gate_a["row_id_sha256"],
        "five selected seeds aggregated by mean prediction where applicable",
        json.loads((ARTIFACTS / "xai" / "model_hashes.json").read_text(encoding="utf-8"))["seed_aggregated_predictions_csv"],
        gate_a["target_sha256"],
        gates_path,
        "scripts/build_claim_eligibility_v5_1_assets.py::build_module_table",
        {"A": gate_a, "B": gate_b, "E": {"tail_delta_rmse": float(tail.paired_delta_rmse), "rank": float(rank.spearman), "top10": f"{int(topk.overlap)}/{int(topk.k)}"}},
        rel(out),
        "module deltas rounded to three decimals; verdicts use DOES NOT PASS",
        output_tex_sha256=sha256(out),
    )


def build_table2_provenance() -> dict[str, object]:
    table_path = ROOT / "paper_versions" / "v5_claim_eligibility_audit" / "source" / "generated" / "table_final_baselines_v4.tex"
    source_path = ARTIFACTS / "audit" / "final_test" / "seed_aggregated_predictions.csv"
    gates_path = ARTIFACTS / "gates" / "figure2_three_comparisons.json"
    gate_a = json.loads(gates_path.read_text(encoding="utf-8"))["comparisons"][0]
    return provenance_entry(
        "table2_locked_same_task_audit",
        "Table II",
        "ExtraTrees",
        "baseline/full/metadata_only/weather_only",
        "extra_trees_leaf_1",
        "locked_2016_2025",
        gate_a["row_id_sha256"],
        "five selected seeds aggregated for learned models; one deterministic baseline seed",
        json.loads((ARTIFACTS / "xai" / "model_hashes.json").read_text(encoding="utf-8"))["seed_aggregated_predictions_csv"],
        gate_a["target_sha256"],
        source_path,
        "existing generated/table_final_baselines_v4.tex from locked audit artifacts",
        "locked RMSE/R2/delta values in generated table",
        rel(table_path),
        "values displayed to three decimals with paired 95% CI brackets",
        output_tex_sha256=sha256(table_path),
    )


def build_fig4_provenance() -> dict[str, object]:
    source_path = ARTIFACTS / "maps" / "state_level_locked_delta_rmse.csv"
    map_pdf = PAPER_GEN / "figure_state_delta_rmse_map.pdf"
    df = pd.read_csv(source_path)
    if int(df["n_locked_rows"].sum()) != 333:
        raise AssertionError("State map locked-row count does not sum to 333")
    return provenance_entry(
        "fig4_state_delta_rmse_map",
        "Fig. 4",
        "ExtraTrees",
        "weather_only_vs_zero",
        "extra_trees_leaf_1",
        "locked_2016_2025",
        sha256(source_path),
        "state-level aggregation of locked rows",
        json.loads((ARTIFACTS / "xai" / "model_hashes.json").read_text(encoding="utf-8"))["seed_aggregated_predictions_csv"],
        "target hash inherited from locked panel audit",
        source_path,
        "existing map-generation artifact",
        df[["state", "n_locked_rows", "delta_rmse_t_ha"]].to_dict(orient="records"),
        rel(map_pdf),
        "state deltas displayed by colorbar; row counts as integer labels",
        output_pdf_sha256=sha256(map_pdf),
    )


def main() -> None:
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    records.append(build_workflow())
    records.append(build_fig2())
    gt_audit, dumbbell_prov, table_prov = build_synthetic_outputs()
    records.extend([dumbbell_prov, build_fig4_provenance(), build_module_table(), build_table2_provenance(), table_prov])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "required_fields": [
            "artifact_id",
            "manuscript_location",
            "model_family",
            "feature_family",
            "configuration_id",
            "split_id",
            "row_id_or_population_hash",
            "seed_or_seed_aggregation",
            "prediction_hash",
            "target_hash",
            "source_metric_file",
            "generation_script",
            "expected_unrounded_value",
            "displayed_value",
            "rounding_rule",
            "assertion_status",
        ],
        "records": records,
        "synthetic_gt_audit": gt_audit,
        "assertion_status": "PASS" if all(r["assertion_status"] == "PASS" for r in records) and gt_audit["assertion_status"] == "PASS" else "FAIL",
    }
    if payload["assertion_status"] != "PASS":
        raise AssertionError("V5.1 provenance assertions failed")
    (REPORT_DIR / "v5_1_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Built V5.1 claim-eligibility assets and provenance")


if __name__ == "__main__":
    main()
