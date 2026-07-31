"""Build v5.8 KSE paper tables and macros from locked audit artifacts.

The script writes only inside the v5.8 paper source tree.  It does not alter
the locked crop, county, PJM, or synthetic experiment artifacts.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "paper_versions"
    / "v5_9_kse_model_first_6pages"
    / "source"
    / "generated"
)

CROP_SUMMARY = ROOT / "artifacts" / "audit" / "final_test" / "naive_baselines.csv"
CROP_PREDICTIONS = (
    ROOT
    / "artifacts"
    / "audit"
    / "final_test"
    / "seed_aggregated_predictions.csv"
)
PAIRED = ROOT / "artifacts" / "audit" / "bootstrap" / "paired_feature_family_ci.csv"
TAIL = ROOT / "artifacts" / "audit" / "tail" / "tail_metrics_by_threshold.csv"
RANK = ROOT / "artifacts" / "audit_records" / "rank_null_audit.csv"
TOPK = ROOT / "artifacts" / "audit_records" / "topk_null_audit.csv"
SYNTHETIC = (
    ROOT
    / "artifacts"
    / "experiments"
    / "synthetic-gate-benchmark"
    / "scenario_level_decisions.csv"
)
SYNTHETIC_SUMMARY = (
    ROOT
    / "artifacts"
    / "experiments"
    / "synthetic-gate-benchmark"
    / "summary.json"
)
PJM = (
    ROOT
    / "artifacts"
    / "experiments"
    / "external-domain-eia"
    / "pjm_predictions.csv"
)
COUNTY_SUMMARY = (
    ROOT
    / "artifacts"
    / "experiments"
    / "county-v2-weather-models"
    / "summary.json"
)
MDE_SUMMARY = (
    ROOT
    / "artifacts"
    / "experiments"
    / "kse-v5-8"
    / "mde"
    / "module_a_mde_summary.json"
)
LIGHTGBM_SUMMARY = (
    ROOT
    / "artifacts"
    / "experiments"
    / "kse-v5-8"
    / "lightgbm"
    / "lightgbm_summary.json"
)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def get_pair(paired: pd.DataFrame, comparison: str) -> pd.Series:
    row = paired[
        (paired["comparison"] == comparison) & (paired["metric"] == "rmse_t_ha")
    ]
    if len(row) != 1:
        raise AssertionError(f"Expected one RMSE row for {comparison}, found {len(row)}")
    return row.iloc[0]


def crop_model_row(
    predictions: pd.DataFrame, config_id: str, family: str
) -> dict[str, float | int | str]:
    group = predictions[
        (predictions["config_id"] == config_id)
        & (predictions["feature_family"] == family)
    ]
    if len(group) != 333:
        raise AssertionError((config_id, family, len(group)))
    y = group["trend_residual_t_ha"].to_numpy(dtype=float)
    p = group["prediction"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - p) ** 2) / denominator)
    return {
        "model": str(group["model"].iloc[0]),
        "family": family,
        "n": len(group),
        "seeds": int(group["n_seeds"].iloc[0]),
        "r2": r2,
        "rmse": rmse,
    }


def write_macros(
    module_a: pd.Series,
    module_b: pd.Series,
    tail: pd.Series,
    rank: pd.Series,
    topk: pd.Series,
    pjm_stats: dict[str, float],
    county: dict,
    mde: dict,
    lightgbm: dict,
    synthetic_summary: dict,
    global_top10_event_overlap: int,
    locked_mean_residual: float,
    baseline_rmse: float,
) -> None:
    macros = {
        "AuditTotalRows": "1,257",
        "AuditValidationRows": "140",
        "AuditValidationExcluded": "4",
        "AuditTestCandidateRows": "343",
        "AuditTestRows": "333",
        "AuditTestExcluded": "10",
        "AuditSelectedModel": "ExtraTrees",
        "AuditSelectedFamily": "Weather-only",
        "AuditSelectedRtwo": "-0.014",
        "AuditSelectedRMSE": "0.669",
        "AuditSelectedDeltaRMSE": f"{module_a.delta_left_minus_right:.3f}",
        "AuditGateACI": (
            f"[{module_a.ci95_low:.3f}, {module_a.ci95_high:.3f}]"
        ),
        "AuditGateBOneDeltaRMSE": f"{module_b.delta_left_minus_right:.3f}",
        "AuditGateBOneCI": (
            f"[{module_b.ci95_low:.3f}, {module_b.ci95_high:.3f}]"
        ),
        "AuditTailRows": f"{int(tail['n'])}",
        "AuditTailDeltaRMSE": f"{tail.paired_delta_rmse:.3f}",
        "AuditTailRankRho": f"{rank.spearman:.3f}",
        "AuditTopKOverlap": f"{int(topk.overlap)}/{int(topk.k)}",
        "AuditGlobalTopKEventOverlap": f"{global_top10_event_overlap}/10",
        "AuditTopKExpectedTailEvents": f"{10 * 73 / 333:.2f}",
        "AuditMDE": f"{mde['mde_absolute_improvement_t_ha']:.3f}",
        "AuditMDEPower": f"{100 * mde['estimated_power_at_mde']:.1f}",
        "AuditMDERelativeBaselinePct": (
            f"{100 * mde['mde_absolute_improvement_t_ha'] / baseline_rmse:.1f}"
        ),
        "AuditObservedRelativeMDEPct": (
            f"{100 * abs(module_a.delta_left_minus_right) / mde['mde_absolute_improvement_t_ha']:.0f}"
        ),
        "AuditLockedMeanResidual": f"{locked_mean_residual:.3f}",
        "AuditLightGBMRMSE": f"{lightgbm['locked_rmse_t_ha']:.3f}",
        "AuditLightGBMRtwo": f"{lightgbm['locked_r2']:.3f}",
        "AuditLightGBMADelta": f"{lightgbm['module_a_delta_rmse_t_ha']:.3f}",
        "AuditLightGBMACI": (
            f"[{lightgbm['module_a_ci95_low']:.3f}, "
            f"{lightgbm['module_a_ci95_high']:.3f}]"
        ),
        "AuditLightGBMBDelta": f"{lightgbm['module_b_delta_rmse_t_ha']:.3f}",
        "AuditLightGBMBCI": (
            f"[{lightgbm['module_b_ci95_low']:.3f}, "
            f"{lightgbm['module_b_ci95_high']:.3f}]"
        ),
        "AuditCountyDelta": f"{county['gate_a_selected_vs_zero']['point_delta_rmse']:.2f}",
        "AuditPJMMeanDemand": f"{pjm_stats['mean_demand_mwh']:,.0f}",
        "AuditPJMFullRelativeRMSE": f"{100 * pjm_stats['full_relative_rmse']:.1f}",
        "AuditPJMCalendarRelativeRMSE": (
            f"{100 * pjm_stats['calendar_relative_rmse']:.1f}"
        ),
        "AuditSyntheticCorrectStopSignal": "68/90",
        "AuditSyntheticCorrectStopSignalPct": "75.6",
        "AuditSyntheticCorrectStopDesign": "1/150",
        "AuditSyntheticCorrectStopDesignPct": "0.7",
        "AuditSyntheticCorrectPermit": (
            f"{synthetic_summary['tp']}/{synthetic_summary['valid_ground_truth_runs']}"
        ),
    }
    text = "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()
    )
    (OUT / "audit_numbers_kse.tex").write_text(text + "\n", encoding="utf-8")


def write_module_table(
    module_a: pd.Series,
    module_b: pd.Series,
    tail: pd.Series,
    rank: pd.Series,
    topk: pd.Series,
) -> None:
    rows = [
        (
            "A",
            "Selected vs. baseline",
            (
                f"{module_a.delta_left_minus_right:.3f} "
                f"[{module_a.ci95_low:.3f}, {module_a.ci95_high:.3f}]"
            ),
            "Not passed",
            "Model-descriptive explanation only",
        ),
        (
            "B",
            "Full vs. metadata",
            (
                f"{module_b.delta_left_minus_right:.3f} "
                f"[{module_b.ci95_low:.3f}, {module_b.ci95_high:.3f}]"
            ),
            "Not passed",
            "Model-descriptive explanation only",
        ),
        (
            "E",
            "Tail, rank, top-$k$",
            (
                f"$\\Delta$RMSE {tail.paired_delta_rmse:.3f}; "
                f"$\\rho$ {rank.spearman:.3f}; "
                f"top-10 {int(topk.overlap)}/{int(topk.k)}"
            ),
            "Not passed",
            "Model-descriptive explanation only",
        ),
    ]
    lines = [
        r"\begin{tabular}{@{}cp{0.17\textwidth}p{0.28\textwidth}cp{0.25\textwidth}@{}}",
        r"\toprule",
        r"Mod. & Locked contrast & Estimate & Verdict & Maximum permitted interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_modules_kse.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_baseline_table(
    baselines: pd.DataFrame,
    predictions: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    def display(value: float) -> str:
        normalized = 0.0 if abs(float(value)) < 0.00005 else float(value)
        return f"{normalized:.4f}"

    rows: list[tuple[str, str, int, int, float, float, str]] = []
    for _, row in baselines.iterrows():
        comparison = f"{row.config_id}_baseline_vs_zero"
        if row.config_id == "zero_residual":
            delta = "0.0000 [0.0000, 0.0000]"
        else:
            pair = get_pair(paired, comparison)
            delta = (
                f"{display(pair.delta_left_minus_right)} "
                f"[{display(pair.ci95_low)}, {display(pair.ci95_high)}]"
            )
        rows.append(
            (
                str(row.model),
                "Baseline",
                int(row.n),
                int(row.n_seeds),
                float(row.r2),
                float(row.rmse_t_ha),
                delta,
            )
        )

    for family, comparison in [
        ("full", "extra_trees_leaf_1_full_vs_zero"),
        ("metadata_only", "extra_trees_leaf_1_metadata_only_vs_zero"),
        ("weather_only", "extra_trees_leaf_1_weather_only_vs_zero"),
    ]:
        item = crop_model_row(predictions, "extra_trees_leaf_1", family)
        pair = get_pair(paired, comparison)
        label = {
            "full": "Full",
            "metadata_only": "Metadata-only",
            "weather_only": "Weather-only",
        }[family]
        rows.append(
            (
                str(item["model"]),
                label,
                int(item["n"]),
                int(item["seeds"]),
                float(item["r2"]),
                float(item["rmse"]),
                (
                    f"{display(pair.delta_left_minus_right)} "
                    f"[{display(pair.ci95_low)}, {display(pair.ci95_high)}]"
                ),
            )
        )

    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Model & Features & $n$ & Seeds & $R^2$ & RMSE & $\Delta$RMSE [95\% CI] \\",
        r"\midrule",
    ]
    for model, family, n, seeds, r2, rmse, delta in rows:
        lines.append(
            f"{model} & {family} & {n:d} & {seeds:d} & "
            f"{r2:.4f} & {rmse:.4f} & {delta} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_baselines_kse.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def count(value: float, denominator: int = 30) -> str:
    numerator = int(round(float(value) * denominator))
    if not np.isclose(numerator / denominator, value):
        raise AssertionError((value, numerator, denominator))
    return f"{numerator}/{denominator}"


def synthetic_rows(
    frame: pd.DataFrame, names: list[str]
) -> list[tuple[str, str, str, str, str]]:
    labels = {
        "correlated_features": "Correlated features",
        "geographic_shift": "Geographic shift",
        "imbalanced_tail": "Imbalanced tail",
        "leakage": "Leakage",
        "measurement_error": "Measurement error",
        "moderate_signal": "Moderate signal",
        "no_signal": "No signal",
        "omitted_confounder": "Omitted confounder",
        "small_sample": "Small sample",
        "spatial_resolution_mismatch": "Spatial mismatch",
        "strong_signal": "Strong signal",
        "temporal_drift": "Temporal drift",
        "train_only_detrending": "Train-only detrending",
        "weak_signal": "Weak signal",
    }
    rows = []
    indexed = frame.set_index("scenario")
    for name in names:
        row = indexed.loc[name]
        rows.append(
            (
                labels[name],
                count(row.module_a_pass_rate),
                count(row.module_b_pass_rate),
                count(row.module_e_pass_rate),
                count(row.policy_permit_rate),
            )
        )
    return rows


def small_synthetic_tabular(
    heading: str,
    rows: list[tuple[str, str, str, str, str]],
    summaries: list[tuple[str, str]],
) -> str:
    lines = [
        rf"\multicolumn{{5}}{{l}}{{\textit{{{heading}}}}} \\",
        r"Scenario & A & B & E & Permissions \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    for label, value in summaries:
        lines.append(rf"\multicolumn{{4}}{{r}}{{\textit{{{label}}}}} & {value} \\")
    return "\n".join(lines)


def write_synthetic_table(frame: pd.DataFrame) -> None:
    signal = ["no_signal", "weak_signal", "measurement_error"]
    design = [
        "temporal_drift",
        "leakage",
        "omitted_confounder",
        "correlated_features",
        "geographic_shift",
    ]
    valid = [
        "imbalanced_tail",
        "moderate_signal",
        "small_sample",
        "spatial_resolution_mismatch",
        "strong_signal",
        "train_only_detrending",
    ]
    signal_text = small_synthetic_tabular(
        "Impermissible: signal absent or degraded",
        synthetic_rows(frame, signal),
        [("correct stops", "68/90")],
    )
    design_text = small_synthetic_tabular(
        "Impermissible: predictive-design violations",
        synthetic_rows(frame, design),
        [("correct stops", "1/150")],
    )
    valid_text = small_synthetic_tabular(
        "Permissible regimes",
        synthetic_rows(frame, valid),
        [("correct permissions", "160/180")],
    )
    text = "\n".join(
        [
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            signal_text,
            r"\midrule",
            design_text,
            r"\midrule",
            valid_text,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUT / "table_synthetic_counts_kse.tex").write_text(
        text + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baselines = pd.read_csv(CROP_SUMMARY).sort_values(
        "config_id", kind="stable"
    )
    predictions = pd.read_csv(CROP_PREDICTIONS)
    paired = pd.read_csv(PAIRED)
    tail = pd.read_csv(TAIL).query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(RANK).query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(TOPK).query(
        "threshold == 'z<-1' and definition == 'k=10'"
    ).iloc[0]
    synthetic = pd.read_csv(SYNTHETIC)
    synthetic_summary = json.loads(SYNTHETIC_SUMMARY.read_text(encoding="utf-8"))
    county = json.loads(COUNTY_SUMMARY.read_text(encoding="utf-8"))
    mde = json.loads(MDE_SUMMARY.read_text(encoding="utf-8"))
    lightgbm = json.loads(LIGHTGBM_SUMMARY.read_text(encoding="utf-8"))

    module_a = get_pair(paired, "extra_trees_leaf_1_weather_only_vs_zero")
    module_b = get_pair(
        paired, "extra_trees_leaf_1_full_vs_metadata_only"
    )
    selected = predictions[
        (predictions["config_id"] == "extra_trees_leaf_1")
        & (predictions["feature_family"] == "weather_only")
    ].copy()
    selected["event"] = selected["trend_residual_z"] < -1.0
    selected["event_score"] = -selected["prediction"]
    global_top10_event_overlap = int(
        selected.nlargest(10, "event_score")["event"].sum()
    )
    if global_top10_event_overlap != 2:
        raise AssertionError(global_top10_event_overlap)

    pjm = pd.read_csv(PJM)
    observed = pjm["observed_demand"].to_numpy(dtype=float)

    def rmse(column: str) -> float:
        prediction = pjm[column].to_numpy(dtype=float)
        return float(np.sqrt(np.mean((observed - prediction) ** 2)))

    mean_demand = float(observed.mean())
    pjm_stats = {
        "n": int(len(pjm)),
        "mean_demand_mwh": mean_demand,
        "naive_rmse_mwh": rmse("naive_train_mean_prediction"),
        "calendar_rmse_mwh": rmse("calendar_prediction"),
        "full_rmse_mwh": rmse("calendar_forecast_prediction"),
    }
    pjm_stats["naive_relative_rmse"] = pjm_stats["naive_rmse_mwh"] / mean_demand
    pjm_stats["calendar_relative_rmse"] = (
        pjm_stats["calendar_rmse_mwh"] / mean_demand
    )
    pjm_stats["full_relative_rmse"] = pjm_stats["full_rmse_mwh"] / mean_demand

    if len(pjm) != 92:
        raise AssertionError(f"Expected 92 locked PJM rows, found {len(pjm)}")
    if synthetic["false_permission_count"].sum() != 171:
        raise AssertionError("Synthetic false-permission count changed")
    if synthetic["false_abstention_count"].sum() != 20:
        raise AssertionError("Synthetic false-abstention count changed")
    max_baseline_difference = pd.read_csv(
        ROOT
        / "artifacts"
        / "audit"
        / "final_test"
        / "baseline_prediction_audit.csv"
    ).query("record_type == 'pairwise_difference'")[
        "max_abs_difference"
    ].max()
    if not 0 < max_baseline_difference < 1e-12:
        raise AssertionError(max_baseline_difference)

    write_macros(
        module_a,
        module_b,
        tail,
        rank,
        topk,
        pjm_stats,
        county,
        mde,
        lightgbm,
        synthetic_summary,
        global_top10_event_overlap,
        float(
            predictions[
                (predictions["config_id"] == "extra_trees_leaf_1")
                & (predictions["feature_family"] == "weather_only")
            ]["trend_residual_t_ha"].mean()
        ),
        float(baselines.loc[baselines["config_id"] == "zero_residual", "rmse_t_ha"].iloc[0]),
    )
    write_module_table(module_a, module_b, tail, rank, topk)
    write_baseline_table(baselines, predictions, paired)
    write_synthetic_table(synthetic)

    report = {
        "status": "PASS",
        "output_directory": str(OUT.relative_to(ROOT)),
        "source_hashes": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                CROP_SUMMARY,
                CROP_PREDICTIONS,
                PAIRED,
                TAIL,
                RANK,
                TOPK,
                SYNTHETIC,
                SYNTHETIC_SUMMARY,
                PJM,
                COUNTY_SUMMARY,
                MDE_SUMMARY,
                LIGHTGBM_SUMMARY,
            ]
        },
        "baseline_max_pairwise_prediction_difference": float(
            max_baseline_difference
        ),
        "pjm": pjm_stats,
        "synthetic_group_checks": {
            "signal_absent_or_degraded_correct_stop": "68/90",
            "predictive_design_violation_correct_stop": "1/150",
            "valid_correct_permission": "160/180",
        },
        "top10_prevalence_expectation": 10 * 73 / 333,
        "top10_all_row_event_overlap": global_top10_event_overlap,
    }
    (OUT / "kse_asset_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
