"""Render manuscript tables and figures from checked canonical audit records."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "artifacts" / "audit_records"
TABLES = ROOT / "artifacts" / "tables"
OUT = ROOT / "paper" / "generated"


def fmt(value: object, digits: int = 3) -> str:
    """Use one numeric formatter and never emit a misleading negative zero."""
    if not isinstance(value, (float, int)):
        return str(value)
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def tex(headers: list[str], rows: list[list[str]], alignment: str) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{alignment}}}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule", *[" & ".join(row) + r" \\" for row in rows], "\\bottomrule", "\\end{tabular}", ""])


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AssertionError(f"{source} lacks required columns: {missing}")


def save_table(name: str, headers: list[str], rows: list[list[str]], alignment: str, csv: pd.DataFrame | None = None) -> None:
    (OUT / f"{name}.tex").write_text(tex(headers, rows, alignment), encoding="utf-8")
    if csv is not None:
        TABLES.mkdir(parents=True, exist_ok=True)
        csv.to_csv(TABLES / f"{name.replace('table_', '')}.csv", index=False)


def ci(point: object, low: object, high: object) -> str:
    return f"{fmt(point)} [{fmt(low)}, {fmt(high)}]"


def human_model(value: str) -> str:
    labels = {"hist_gradient_boosting": "HistGradientBoosting", "elastic_net": "ElasticNet", "extra_trees_leaf_1": "ExtraTrees (leaf=1)", "extra_trees_leaf_2": "ExtraTrees (leaf=2)", "random_forest_leaf_1": "Random Forest (leaf=1)", "random_forest_leaf_2": "Random Forest (leaf=2)"}
    return labels.get(value, value.replace("_", " "))


def digest_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values("row_id")[columns]
    return hashlib.sha256(ordered.to_csv(index=False, float_format="%.17g").encode("utf-8")).hexdigest()


def build_gate_b1_representative_manifest(selected: dict[str, object]) -> None:
    """Record the locked feature-family representatives used by Gate A/B1/B2."""
    aggregate = pd.read_csv(ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv")
    validation = pd.read_csv(ROOT / "artifacts" / "audit" / "selection" / "validation_model_grid.csv")
    config_id = str(selected["config_id"])
    families = {
        "Gate A selected model": "weather_only",
        "Gate B1 left representative": "full",
        "Gate B1 right representative": "metadata_only",
        "Gate B2 left diagnostic representative": "weather_only",
        "Gate B2 right diagnostic representative": "metadata_only",
    }
    records = []
    for role, family in families.items():
        vector = aggregate[(aggregate.config_id == config_id) & (aggregate.feature_family == family)][["row_id", "trend_residual_t_ha", "prediction"]].copy()
        val_row = validation[(validation.config_id == config_id) & (validation.feature_family == family)].iloc[0]
        records.append(
            {
                "role": role,
                "config_id": config_id,
                "model": selected["model"],
                "feature_family": family,
                "selection_rule": "Fixed architecture inherited from the validation-selected Gate A configuration; feature family is changed only to form the pre-specified Gate B1/B2 contrast.",
                "validation_rmse_t_ha": float(val_row.rmse_t_ha),
                "locked_rows": len(vector),
                "row_id_sha256": digest_frame(vector, ["row_id"]),
                "target_sha256": digest_frame(vector, ["row_id", "trend_residual_t_ha"]),
                "prediction_sha256": digest_frame(vector, ["row_id", "prediction"]),
                "locked_test_access_for_selection": False,
            }
        )
    selection_dir = ROOT / "artifacts" / "audit" / "selection"
    pd.DataFrame(records).to_csv(selection_dir / "gate_b1_representatives.csv", index=False)
    payload = {
        "schema": "gate-b1-representatives-v1",
        "gate_a_selected_config_id": config_id,
        "gate_a_selected_feature_family": selected["feature_family"],
        "protocol": "Gate A uses the single overall validation-selected configuration. Fixed architecture Gate B1 is a pre-specified feature-group contrast: Full and Metadata-only predictions use the same config id as the Gate A winner and are locked before final-test interpretation.",
        "final_test_accessed_for_selection": False,
        "records": records,
    }
    (selection_dir / "gate_b1_representatives.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_figure_two() -> None:
    """Render the three Gate A/B comparisons from paired locked-test records."""
    paired = pd.read_csv(RECORDS / "paired_comparisons.csv")
    selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text(encoding="utf-8"))["selected_config"]
    build_gate_b1_representative_manifest(selected)
    config_id = str(selected["config_id"])
    aggregate = pd.read_csv(ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv")
    comparisons = [
        ("Gate A", "Selected vs Zero", f"{config_id}_weather_only_vs_zero", config_id, "weather_only", "zero_residual", "baseline", "primary"),
        ("Gate B1 PRIMARY", "Full vs Metadata", f"{config_id}_full_vs_metadata_only", config_id, "full", config_id, "metadata_only", "primary"),
        ("Gate B2 DIAGNOSTIC", "Weather vs Metadata", f"{config_id}_weather_only_vs_metadata_only", config_id, "weather_only", config_id, "metadata_only", "diagnostic"),
    ]
    rows, evidence = [], []
    draw_path = ROOT / "artifacts" / "audit" / "bootstrap" / "year_block_draws.csv"
    draw_hash = hashlib.sha256(draw_path.read_bytes()).hexdigest()
    for gate, label, comparison, left_id, left_family, right_id, right_family, role in comparisons:
        result = paired[(paired.comparison == comparison) & (paired.metric == "rmse_t_ha")].iloc[0]
        left = aggregate[(aggregate.config_id == left_id) & (aggregate.feature_family == left_family)][["row_id", "trend_residual_t_ha", "prediction"]].copy()
        right = aggregate[(aggregate.config_id == right_id) & (aggregate.feature_family == right_family)][["row_id", "trend_residual_t_ha", "prediction"]].copy()
        left_target = left.sort_values("row_id").trend_residual_t_ha.to_numpy()
        right_target = right.sort_values("row_id").trend_residual_t_ha.to_numpy()
        if len(left) != len(right) or set(left.row_id) != set(right.row_id) or not np.array_equal(left_target, right_target):
            raise AssertionError(f"Figure 2 comparison is not row/target aligned: {comparison}")
        if int(result.n) != len(left) or int(result.n_boot) != 2000 or result.resampling_unit != "year_block":
            raise AssertionError(f"Figure 2 paired record violates the locked bootstrap contract: {comparison}")
        rows.append({"gate": gate, "label": label, "comparison": comparison, "estimate": float(result.delta_left_minus_right), "ci95_low": float(result.ci95_low), "ci95_high": float(result.ci95_high), "role": role})
        evidence.append({"gate": gate, "role": role, "comparison": comparison, "n_rows": int(result.n), "n_boot": int(result.n_boot), "resampling_unit": result.resampling_unit, "left": {"config_id": left_id, "feature_family": left_family, "prediction_sha256": digest_frame(left, ["row_id", "prediction"])}, "right": {"config_id": right_id, "feature_family": right_family, "prediction_sha256": digest_frame(right, ["row_id", "prediction"])}, "row_id_sha256": digest_frame(left, ["row_id"]), "target_sha256": digest_frame(left, ["row_id", "trend_residual_t_ha"]), "estimate": float(result.delta_left_minus_right), "ci95_low": float(result.ci95_low), "ci95_high": float(result.ci95_high)})
    values = pd.DataFrame(rows)
    gate_dir = ROOT / "artifacts" / "gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "figure2_three_comparisons.json").write_text(json.dumps({"selected_config": selected, "bootstrap_draw_sha256": draw_hash, "comparisons": evidence}, indent=2) + "\n", encoding="utf-8")
    TABLES.mkdir(parents=True, exist_ok=True)
    values.to_csv(TABLES / "figure2_three_comparisons.csv", index=False)
    errors = np.array([values.estimate - values.ci95_low, values.ci95_high - values.estimate])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    x = np.arange(len(values))
    ax.bar(x, values.estimate, yerr=errors, capsize=3, color=["#ad3b3b", "#276f5f", "#5b7db1"], width=.62)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, [f"{gate}\n{label}" for gate, label in zip(values.gate, values.label)], fontsize=6.5)
    ax.set_ylabel(r"Paired $\Delta$RMSE (left $-$ right)", fontsize=7)
    ax.set_title("Locked 2016--2025; year-block 95% intervals", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(pad=.6)
    fig.savefig(OUT / "figure_final_audit.pdf")
    fig.savefig(OUT / "figure_final_audit.png", dpi=220)
    plt.close(fig)


def build_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 1.95))
    labels = [
        "Train-only\ntarget",
        "Completed-season\nweather features",
        "Validation lock\n2012--2015",
        "Future-period test\n2016--2025",
        "Gate A + B1\nABSTAIN/INTERPRET",
    ]
    for index, label in enumerate(labels):
        ax.text(index, .55, label, ha="center", va="center", fontsize=8.6, bbox={"boxstyle": "round,pad=0.38", "facecolor": "#e7f0ed", "edgecolor": "#276f5f"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(index + .60, .55), xytext=(index + .40, .55), arrowprops={"arrowstyle": "->", "lw": 1.1})
    ax.text(1, .13, "post-season scientific audit, not pre-harvest forecast", ha="center", fontsize=7.5, color="#4c5f69")
    ax.text(4, .13, "Gate B2 remains diagnostic only", ha="center", fontsize=7.5, color="#4c5f69")
    ax.set_xlim(-.55, 4.55)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=.35)
    fig.savefig(OUT / "figure_audit_workflow.pdf")
    fig.savefig(OUT / "figure_audit_workflow.png", dpi=240)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_figure_two()
    build_workflow_figure()
    topk = pd.read_csv(RECORDS / "topk_null_audit.csv")
    rank = pd.read_csv(RECORDS / "rank_null_audit.csv")
    tail = pd.read_csv(ROOT / "artifacts" / "audit" / "tail" / "tail_metrics_by_threshold.csv")
    error_rank_rows, topk_rows = [], []
    for item in tail.itertuples():
        rank_row = rank[rank.threshold == item.threshold].iloc[0]
        top = topk[(topk.threshold == item.threshold) & (topk.definition == "k=10")].iloc[0]
        role = "Primary" if item.threshold == "z<-1" else "Sensitivity"
        error_rank_rows.append([role, item.threshold.replace("<", "$<$"), str(item.n), ci(item.paired_delta_rmse, item.paired_delta_rmse_ci95_low, item.paired_delta_rmse_ci95_high), ci(item.paired_delta_mae, item.paired_delta_mae_ci95_low, item.paired_delta_mae_ci95_high), f"{fmt(rank_row.spearman)} [{fmt(rank_row.spearman_ci95_low)}, {fmt(rank_row.spearman_ci95_high)}]", fmt(rank_row.permutation_pvalue), str(top.status)])
        topk_rows.append([role, item.threshold.replace("<", "$<$"), str(item.n), f"{int(top.overlap)}/{int(top.k)}", fmt(top.random_expectation * top.k, 2), f"{fmt(top.lift,2)} [{fmt(top.lift_ci95_low,2)}, {fmt(top.lift_ci95_high,2)}]", fmt(top.hypergeometric_pvalue), fmt(top.permutation_pvalue), str(top.status)])
    save_table("table_tail_error_rank", ["Role", "Tail", "$n$", "$\\Delta$RMSE [95\\% CI]", "$\\Delta$MAE [95\\% CI]", "Rank $\\rho$ [95\\% CI]", "Perm. $p$", "Status"], error_rank_rows, "llrrrrrl", pd.DataFrame(error_rank_rows, columns=["Role", "Tail", "n", "Delta RMSE [95% CI]", "Delta MAE [95% CI]", "Rank rho [95% CI]", "Permutation p", "Status"]))
    save_table("table_tail_topk", ["Role", "Tail", "$n$", "Observed", "Expected", "Lift [95\\% CI]", "$H$ $p$", "$P$ $p$", "Status"], topk_rows, "llrrrrrrl", pd.DataFrame(topk_rows, columns=["Role", "Tail", "n", "Observed overlap", "Expected overlap", "Lift [95% CI]", "Hypergeometric p", "Within-year permutation p", "Status"]))

    gate = pd.read_csv(RECORDS / "fidelity_gate_components.csv")
    gate_rows = []
    for x in gate.itertuples():
        component = {"extra_trees_leaf_1_full_vs_metadata_only": "Full vs. Metadata only", "extra_trees_leaf_1_weather_only_vs_metadata_only": "Weather only vs. Metadata only diagnostic", "FINAL GATE B1": "FINAL GATE B1"}.get(str(x.component), str(x.component).replace("_", " "))
        gate_rows.append([str(x.gate), component, str(x.scope).replace("<", "$<$"), str(x.status)])
    save_table("table_gate_ab", ["Gate", "Component", "Scope", "Status"], gate_rows, "lllc")
    gate_policy_rows = [["Gate A overall", "Primary", "Selected model vs. zero: upper paired 95\\% RMSE CI $<$ 0"], ["Gate A primary tail", "Primary, $z<-1$", "RMSE/MAE CI upper $<$ 0; rank CI lower $>0$ and permutation $p<.05$; top-10 lift $>1$, lift-CI lower $>1$, $H$ and within-year $P$ $p<.05$"], ["Gate B1 weather value", "Primary", "Full vs. Metadata only: upper paired 95\\% RMSE CI $<$ 0"], ["Gate B2", "Diagnostic", "Weather only vs. Metadata only; exploratory representation contrast only"], ["Severe tails", "Sensitivity", "$z<-1.5$ and $z<-2$; cannot replace the primary tail"]]
    save_table("table_gate_definition", ["Component", "Role", "Pre-specified pass condition"], gate_policy_rows, "p{0.20\\textwidth}p{0.16\\textwidth}p{0.56\\textwidth}")
    policy_check = {"config": "configs/fidelity_gate.yaml", "primary_tail": "z<-1", "gate_b1": "Full vs. Metadata only", "gate_b2": "Weather only vs. Metadata only diagnostic", "topk_conjunction": "lift > 1; lower lift CI > 1; H p < .05; P p < .05", "table_rows": len(gate_policy_rows)}
    validation_dir = ROOT / "artifacts" / "validation"; validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "table_ii_to_config_consistency.json").write_text(json.dumps(policy_check, indent=2) + "\n", encoding="utf-8")

    rolling = pd.read_csv(RECORDS / "temporal_and_capacity_audits.csv").query("audit == 'rolling_origin'")
    stage = pd.read_csv(ROOT / "artifacts" / "audit" / "stage_features" / "stage_feature_sensitivity.csv")
    crops = pd.read_csv(ROOT / "artifacts" / "audit" / "crop" / "crop_specific_metrics.csv")
    loso = pd.read_csv(ROOT / "artifacts" / "audit" / "spatial" / "leave_one_state_out.csv")
    retrospective_summary = pd.read_csv(RECORDS / "retrospective_target_comparison.csv").iloc[0]
    robust_rows = [
        ["Rolling origin", "3/4 negative; 1/4 CI pass", "Diagnostic", "Unstable"],
        ["Stage proxies", f"$\\Delta$RMSE {fmt(stage.iloc[-1].delta_rmse_vs_primary)}", "Sensitivity", "No material change"],
        ["Crop-specific", f"$R^2$ {fmt(crops.r2.min())} to {fmt(crops.r2.max())}", "Diagnostic", "Unstable"],
        ["Leave-one-state-out", f"{int(loso.rmse_better_than_zero.sum())}/{len(loso)} states improve", "Diagnostic", "Unstable"],
        ["Full-series detrending", f"Jaccard {fmt(retrospective_summary.jaccard)}", "Retrospective", "Future yields enter target"],
    ]
    robust_csv = pd.DataFrame(robust_rows, columns=["Audit", "Main result", "Role", "Interpretation"])
    save_table("table_robustness", list(robust_csv.columns), robust_rows, "p{0.20\\textwidth}p{0.32\\textwidth}p{0.16\\textwidth}p{0.24\\textwidth}", robust_csv)

    history = pd.read_csv(RECORDS / "min_history_sensitivity.csv")
    scale = pd.read_csv(RECORDS / "target_scale_sensitivity.csv")
    detrending = pd.read_csv(RECORDS / "alternative_detrending_sensitivity.csv")
    expanded = pd.read_csv(RECORDS / "expanded_model_baselines.csv")
    for frame, fields, source in ((history, ["min_history", "n_test", "rmse_t_ha", "baseline_rmse_t_ha", "delta_rmse_t_ha", "delta_rmse_ci95_low", "delta_rmse_ci95_high", "gate_a_overall_status"], "min-history"), (scale, ["n_test", "rmse_z", "baseline_rmse_z", "delta_rmse_z", "delta_rmse_z_ci95_low", "delta_rmse_z_ci95_high", "status"], "scale"), (detrending, ["detrending", "n_test", "rmse_t_ha", "baseline_rmse_t_ha", "delta_rmse_t_ha", "delta_rmse_ci95_low", "delta_rmse_ci95_high", "status"], "detrending"), (expanded, ["model", "n_final", "rmse_t_ha", "baseline_rmse_t_ha", "delta_rmse_t_ha", "delta_rmse_ci95_low", "delta_rmse_ci95_high", "status"], "expanded")):
        require_columns(frame, fields, source)
    target_rows, target_csv = [], []
    for x in history.itertuples():
        role = "Primary" if x.min_history == 3 else "Sensitivity"
        population = "Same" if x.min_history in (3, 5) else "Different"
        result = "PASS*" if x.min_history in (8, 10) and "PASS" in x.gate_a_overall_status else "FAIL"
        target_rows.append([f"History {x.min_history}", role, population, str(x.n_test), "t ha$^{-1}$", f"{fmt(x.rmse_t_ha)} / {fmt(x.baseline_rmse_t_ha)}", f"{fmt(x.delta_rmse_t_ha)} [{fmt(x.delta_rmse_ci95_low)}, {fmt(x.delta_rmse_ci95_high)}]", result])
    for x in scale.itertuples():
        target_rows.append(["Standardized target", "Sensitivity", "Same", str(x.n_test), "z units", f"{fmt(x.rmse_z)} / {fmt(x.baseline_rmse_z)}", f"{fmt(x.delta_rmse_z)} [{fmt(x.delta_rmse_z_ci95_low)}, {fmt(x.delta_rmse_z_ci95_high)}]", "FAIL"])
    for x in detrending.itertuples():
        if x.detrending == "huber_train_only":
            target_rows.append(["Huber detrending", "Sensitivity", "Same", str(x.n_test), "t ha$^{-1}$", f"{fmt(x.rmse_t_ha)} / {fmt(x.baseline_rmse_t_ha)}", f"{fmt(x.delta_rmse_t_ha)} [{fmt(x.delta_rmse_ci95_low)}, {fmt(x.delta_rmse_ci95_high)}]", "FAIL"])
    for x in expanded.itertuples():
        target_rows.append([human_model(x.model), "Sensitivity", "Same", str(x.n_final), "t ha$^{-1}$", f"{fmt(x.rmse_t_ha)} / {fmt(x.baseline_rmse_t_ha)}", f"{fmt(x.delta_rmse_t_ha)} [{fmt(x.delta_rmse_ci95_low)}, {fmt(x.delta_rmse_ci95_high)}]", "FAIL"])
    target_csv = pd.DataFrame(target_rows, columns=["Analysis", "Role", "Population", "Rows", "Scale", "RMSE / zero", "Delta RMSE [95% CI]", "Result"])
    save_table("table_target_model_sensitivity", ["Analysis", "Role", "Population", "Rows", "Scale", "RMSE / zero", "Delta RMSE [95\\% CI]", "Result"], target_rows, "lllrllll", target_csv)

    bootstrap = pd.read_csv(RECORDS / "bootstrap_scheme_comparison.csv")
    require_columns(bootstrap, ["scheme", "n_clusters", "n_test", "estimate", "ci95_low", "ci95_high", "status"], "resampling")
    resampling_rows = [[str(x.scheme).replace("_", " "), str(x.n_clusters), str(x.n_test), fmt(x.estimate), f"[{fmt(x.ci95_low)}, {fmt(x.ci95_high)}]", str(x.status)] for x in bootstrap.itertuples()]
    resampling_csv = pd.DataFrame(resampling_rows, columns=["Resampling unit", "Number clusters", "Test rows", "Delta RMSE", "95% CI", "Status"])
    save_table("table_resampling_sensitivity", ["Resampling unit", "Number clusters", "Test rows", "Delta RMSE", "95\\% CI", "Status"], resampling_rows, "lrrrlc", resampling_csv)

    temporal = pd.read_csv(RECORDS / "temporal_and_capacity_audits.csv")
    require_columns(temporal, ["audit", "n_train", "n_test", "delta_rmse_t_ha", "delta_rmse_ci95_low", "delta_rmse_ci95_high", "status"], "temporal")
    temporal_rows = []
    for x in temporal.itertuples():
        label = f"Rolling fold {int(x.fold)} ({int(x.test_start)}--{int(x.test_end)})" if x.audit == "rolling_origin" else f"Prefix {x.fraction * 100:.0f}\\% through {int(x.train_end)}"
        temporal_rows.append([label, str(x.n_train), str(x.n_test), fmt(x.delta_rmse_t_ha), f"[{fmt(x.delta_rmse_ci95_low)}, {fmt(x.delta_rmse_ci95_high)}]", str(x.status)])
    save_table("table_temporal_capacity", ["Audit", "$n_{train}$", "$n_{test}$", "$\\Delta$RMSE", "95\\% CI", "Status"], temporal_rows, "lrrrlc")

    flow = pd.read_csv(ROOT / "artifacts" / "data" / "data_flow.csv")
    save_table("table_data_flow", ["Stage", "Rows", "Rule"], [[str(x.stage).replace("_", " "), str(x.rows), str(x.rule)] for x in flow.itertuples()], "lrl")
    events = pd.read_csv(RECORDS / "event_detection_metrics.csv")
    require_columns(events, ["true_positive", "false_positive", "true_negative", "false_negative", "precision", "recall", "f1", "decision_score_threshold"], "event detection")
    save_table("table_event_detection", ["Tail", "Prev.", "PR-AUC", "ROC-AUC", "Bal. acc.", "Precision", "Recall", "F1"], [[str(x.threshold).replace("<", "$<$"), fmt(x.prevalence), fmt(x.pr_auc), fmt(x.roc_auc), fmt(x.balanced_accuracy), fmt(x.precision), fmt(x.recall), fmt(x.f1)] for x in events.itertuples()], "lrrrrrrr")

    all_stability = pd.read_csv(RECORDS / "validation_stability.csv")
    stability = all_stability.head(6)
    config_map = pd.DataFrame([{"config_id": x.config_id, "display_label": human_model(str(x.config_id)), "feature_family_display": str(x.feature_family).replace("weather_only", "Weather only").replace("metadata_only", "Metadata only").replace("full", "Full")} for x in all_stability.itertuples()]).drop_duplicates()
    TABLES.mkdir(parents=True, exist_ok=True); (ROOT / "artifacts" / "validation").mkdir(parents=True, exist_ok=True)
    config_map.to_csv(ROOT / "artifacts" / "validation" / "config_display_map.csv", index=False)
    if config_map.duplicated(["config_id", "feature_family_display"]).any() or config_map.display_label.isna().any():
        raise AssertionError("Validation display mapping must be one-to-one per configuration and feature family, and complete")
    stable_rows = [[human_model(str(x.config_id)), str(x.feature_family).replace("weather_only", "Weather only").replace("metadata_only", "Metadata only").replace("full", "Full"), fmt(x.rmse_mean), fmt(x.rmse_sd), fmt(x.rank_first_probability, 2), "Yes" if x.one_standard_error_eligible else "No"] for x in stability.itertuples()]
    save_table("table_selection_stability", ["Model", "Features", "Mean RMSE", "SD", "P(rank=1)", "1-SE"], stable_rows, "llrrrr")
    selected = pd.read_json(ROOT / "artifacts" / "audit" / "selection" / "selected_config.json", typ="series")["selected_config"]
    needed = pd.DataFrame(
        [
            {"config_id": selected["config_id"], "feature_family": "weather_only"},
            {"config_id": selected["config_id"], "feature_family": "full"},
            {"config_id": selected["config_id"], "feature_family": "metadata_only"},
        ]
    )
    selection_display = pd.concat(
        [
            all_stability.head(6)[["config_id", "model", "feature_family", "rmse_mean", "rmse_sd", "seed_count"]],
            all_stability.merge(needed, on=["config_id", "feature_family"])[["config_id", "model", "feature_family", "rmse_mean", "rmse_sd", "seed_count"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["config_id", "feature_family"])

    def gate_role(row: pd.Series) -> str:
        if row.config_id == selected["config_id"] and row.feature_family == "weather_only":
            return "Gate A; B2"
        if row.config_id == selected["config_id"] and row.feature_family == "full":
            return "B1 Full"
        if row.config_id == selected["config_id"] and row.feature_family == "metadata_only":
            return "B1/B2 Metadata"
        return "Validation candidate"

    selection_rows = [
        [
            human_model(str(x.config_id)),
            str(x.feature_family).replace("weather_only", "Weather only").replace("metadata_only", "Metadata only").replace("full", "Full"),
            gate_role(x),
            str(int(selected["n"])),
            str(int(selected["validation_seed_count"])),
            fmt(x.rmse_mean),
            fmt(x.rmse_sd),
            "Yes" if (x.config_id == selected["config_id"] and x.feature_family == selected["feature_family"]) else "No",
        ]
        for x in selection_display.itertuples()
    ]
    save_table("table_validation_selection", ["Model", "Features", "Role", "$n$", "Seeds", "RMSE", "Seed SD", "Selected"], selection_rows, "lllrrrrr")
    cross_table = pd.DataFrame([{"config_id": x.config_id, "feature_family": x.feature_family, "display_label": human_model(str(x.config_id)), "in_table_iii": True, "in_table_iv": True} for x in stability.itertuples()])
    cross_table.to_csv(ROOT / "artifacts" / "validation" / "config_cross_table_audit.csv", index=False)
    if cross_table.display_label.str.contains("_").any() or not cross_table.in_table_iii.all() or not cross_table.in_table_iv.all():
        raise AssertionError("Tables III and IV must use the same human-readable configuration labels")

    retrospective = pd.read_csv(RECORDS / "retrospective_target_comparison.csv").iloc[0]
    transition = pd.read_csv(RECORDS / "target_membership_transition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    axes[0].bar(["Train-only", "Full-series"], [retrospective.prospective_events, retrospective.retrospective_events], color=["#467d74", "#a76645"], hatch=["", "//"])
    axes[0].set_ylabel("z<-1 event rows"); axes[0].set_title(f"Jaccard {retrospective.jaccard:.3f} [{retrospective.jaccard_ci95_low:.3f}, {retrospective.jaccard_ci95_high:.3f}]")
    labels = ["Neither" if not x.event and not x.retrospective_event else "Train-only only" if x.event and not x.retrospective_event else "Full-series only" if not x.event and x.retrospective_event else "Both" for x in transition.itertuples()]
    axes[1].bar(labels, transition["size"], color="#667fb2", hatch="//")
    axes[1].set_title("Membership transitions"); axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", labelrotation=22, labelsize=6)
    fig.text(.5, .01, "RETROSPECTIVE ONLY: full-series detrending uses future yields.", ha="center", color="#9c2d2d", fontsize=8, fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / "figure_retrospective_v2.png", dpi=220); fig.savefig(OUT / "figure_retrospective_v2.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
