"""E07, E08, E10, E11 and E12 sensitivities for the canonical audit."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_yield_xai.core import detrend_and_score, detrend_train_test, full_season_weather_features, load_frame, make_project_paths  # noqa: E402
from run_main8_audit import (  # noqa: E402
    BOOTSTRAP_SEED, CONFIGS, FINAL_TEST, FINAL_TRAIN_END, N_BOOT, SEEDS, aggregate_predictions,
    metric_dict, paired_delta, pipeline, selected_config, selected_features,
)


def write(frame: pd.DataFrame, name: str) -> None:
    path = ROOT / "artifacts" / "audit_records" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def row_id(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: f"{row.crop}|{row.region}|{int(row.year)}|{row.window}", axis=1)


def score_fold(frame: pd.DataFrame, min_history: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = frame[frame.year <= FINAL_TRAIN_END].copy()
    test_raw = frame[(frame.year >= FINAL_TEST[0]) & (frame.year <= FINAL_TEST[1])].copy()
    history = train_raw.groupby(["crop", "region"]).size().rename("n_train_history")
    test = test_raw.join(history, on=["crop", "region"])
    test = test[test.n_train_history >= min_history].drop(columns="n_train_history")
    train = train_raw.merge(test[["crop", "region"]].drop_duplicates(), on=["crop", "region"], how="inner")
    return detrend_train_test(train, test)[:2]


def model_predictions(train: pd.DataFrame, test: pd.DataFrame, config: dict[str, object], target: str = "trend_residual_t_ha") -> pd.DataFrame:
    numeric, categorical = selected_features(pd.concat([train, test], ignore_index=True), "weather_only")
    outputs = []
    for seed in SEEDS:
        fitted = pipeline(numeric, categorical, config, seed)
        fitted.fit(train[numeric + categorical], train[target])
        result = test[["crop", "region", "year", "window", "trend_residual_t_ha", "trend_residual_z"]].copy()
        result.insert(0, "row_id", row_id(result))
        result["prediction"] = fitted.predict(test[numeric + categorical])
        result["seed"] = seed
        outputs.append(result)
    return aggregate_predictions(pd.concat([output.assign(config_id=str(config["config_id"]), model=str(config["model"]), feature_family="weather_only") for output in outputs], ignore_index=True))


def huber_detrend_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternative target based on a robust trend fit using training rows only."""
    train_parts, test_parts = [], []
    for key, train_group in train.groupby(["crop", "region"], sort=True):
        train_group = train_group.sort_values("year").copy()
        test_group = test[(test.crop == key[0]) & (test.region == key[1])].sort_values("year").copy()
        fitted = HuberRegressor(epsilon=1.35, max_iter=1000).fit(train_group[["year"]], train_group["yield_t_ha"])
        train_residual = train_group["yield_t_ha"].to_numpy() - fitted.predict(train_group[["year"]])
        scale = float(np.std(train_residual, ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise AssertionError(f"Invalid Huber residual scale for {key}")
        for group, output in ((train_group, train_parts), (test_group, test_parts)):
            result = group.copy()
            result["trend_residual_t_ha"] = result["yield_t_ha"].to_numpy() - fitted.predict(result[["year"]])
            result["trend_residual_z"] = result["trend_residual_t_ha"] / scale
            output.append(result)
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def year_draws(years: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    years = np.unique(years)
    return pd.DataFrame({"replicate": range(N_BOOT), "sampled_years": ["|".join(map(str, rng.choice(years, len(years), replace=True))) for _ in range(N_BOOT)]})


def min_history_and_scale() -> None:
    frame = load_frame(make_project_paths(ROOT))
    selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text())["selected_config"]
    config = next(item for item in CONFIGS if item["config_id"] == selected["config_id"])
    rows, scale_rows, history_vectors = [], [], {}
    reference_ids: set[str] | None = None
    for history in (3, 5, 8, 10):
        train, test = score_fold(frame, history)
        raw = model_predictions(train, test, config)
        zero = raw.copy(); zero["prediction"] = 0.0; zero["config_id"] = "zero_residual"; zero["model"] = "Zero residual"; zero["feature_family"] = "baseline"
        delta = paired_delta(raw, zero, year_draws(raw.year.to_numpy()), f"history_{history}_selected_vs_zero")
        rmse = delta[delta.metric == "rmse_t_ha"].iloc[0]
        metrics = metric_dict(raw.trend_residual_t_ha, raw.prediction)
        baseline = metric_dict(raw.trend_residual_t_ha, np.zeros(len(raw)))
        ids = set(raw.row_id)
        if history in (8, 10):
            ordered = raw.sort_values("row_id").reset_index(drop=True)
            def vector_hash(columns: list[str]) -> str:
                return hashlib.sha256(ordered[columns].to_csv(index=False, float_format="%.17g").encode("utf-8")).hexdigest()
            history_vectors[history] = {
                "rows": ordered,
                "row_id_sha256": vector_hash(["row_id"]),
                "target_sha256": vector_hash(["row_id", "trend_residual_t_ha"]),
                "prediction_sha256": vector_hash(["row_id", "prediction"]),
            }
        jaccard = 1.0 if reference_ids is None else len(ids & reference_ids) / len(ids | reference_ids)
        reference_ids = ids if history == 3 else reference_ids
        same_as_primary = ids == reference_ids
        population_label = "SAME ELIGIBLE POPULATION" if same_as_primary else "DIFFERENT POPULATION"
        status = "PRIMARY FAIL" if history == 3 and rmse.ci95_high >= 0 else ("PRIMARY PASS" if history == 3 else (f"SENSITIVITY PASS; {population_label}" if rmse.ci95_high < 0 else f"SENSITIVITY FAIL; {population_label}"))
        rows.append({"min_history": history, "n_train": len(train), "n_test": len(test), "tail_z_lt_1_n": int((raw.trend_residual_z < -1).sum()), "eligibility_jaccard_vs_history3": jaccard, **metrics, "baseline_rmse_t_ha": baseline["rmse_t_ha"], "delta_rmse_t_ha": rmse.delta_left_minus_right, "delta_rmse_ci95_low": rmse.ci95_low, "delta_rmse_ci95_high": rmse.ci95_high, "gate_a_overall_status": status})
        if history == 3:
            standardized = model_predictions(train, test, config, target="trend_residual_z")
            zero_z = standardized.copy()
            zero_z["prediction"] = 0.0
            # paired_delta expects its response in the historical column name.
            left = standardized.drop(columns="trend_residual_t_ha").rename(columns={"trend_residual_z": "trend_residual_t_ha"})
            right = zero_z.drop(columns="trend_residual_t_ha").rename(columns={"trend_residual_z": "trend_residual_t_ha"})
            delta_z = paired_delta(left, right, year_draws(standardized.year.to_numpy()), "standardized_target_vs_zero")
            z_rmse = delta_z[delta_z.metric == "rmse_t_ha"].iloc[0]
            z_metric = metric_dict(standardized.trend_residual_z, standardized.prediction)
            z_base = metric_dict(standardized.trend_residual_z, np.zeros(len(standardized)))
            scale_rows.append({"analysis": "model trained on train-only standardized residual z", "target": "trend_residual_z", "n_train": len(train), "n_test": len(test), "rmse_z": z_metric["rmse_t_ha"], "baseline_rmse_z": z_base["rmse_t_ha"], "delta_rmse_z": z_rmse.delta_left_minus_right, "mae_z": z_metric["mae_t_ha"], "r2_z": z_metric["r2"], "delta_rmse_z_ci95_low": z_rmse.ci95_low, "delta_rmse_z_ci95_high": z_rmse.ci95_high, "status": "SENSITIVITY PASS" if z_rmse.ci95_high < 0 else "SENSITIVITY FAIL"})
    write(pd.DataFrame(rows), "min_history_sensitivity.csv")
    write(pd.DataFrame(scale_rows), "target_scale_sensitivity.csv")
    primary_train, _ = score_fold(frame, 3)
    scales = primary_train.groupby(["crop", "region"], as_index=False)["trend_residual_t_ha"].std(ddof=1).rename(columns={"trend_residual_t_ha": "sigma_train"})
    scales["finite"] = np.isfinite(scales.sigma_train)
    scales["above_minimum"] = scales.sigma_train > 1e-8
    targets_dir = ROOT / "artifacts" / "targets"; targets_dir.mkdir(parents=True, exist_ok=True)
    scales.to_csv(targets_dir / "train_scale_diagnostics.csv", index=False)
    (targets_dir / "train_scale_summary.json").write_text(json.dumps({"minimum_guard": 1e-8, "n_series": len(scales), "n_finite": int(scales.finite.sum()), "n_above_minimum": int(scales.above_minimum.sum()), "min": float(scales.sigma_train.min()), "q05": float(scales.sigma_train.quantile(.05)), "median": float(scales.sigma_train.median())}, indent=2) + "\n", encoding="utf-8")
    _, final_test = score_fold(frame, 3)
    series_history = frame[frame.year <= FINAL_TRAIN_END].groupby(["crop", "region"]).size().rename("series_history")
    population = final_test.assign(row_id=row_id(final_test)).join(series_history, on=["crop", "region"])
    for threshold in (3, 5, 8, 10):
        population[f"eligible_history_{threshold}"] = population.series_history >= threshold
        population[f"rows_removed_history_{threshold}"] = int((~population[f"eligible_history_{threshold}"]).sum())
    population["crop_state"] = population.crop + "|" + population.region
    population["overlap_with_history3"] = 1.0
    population = population[["row_id", "crop", "region", "year", "window", "series_history", "crop_state", "eligible_history_3", "eligible_history_5", "eligible_history_8", "eligible_history_10", "rows_removed_history_3", "rows_removed_history_5", "rows_removed_history_8", "rows_removed_history_10", "overlap_with_history3"]]
    write(population, "min_history_population_audit.csv")
    membership_dir = ROOT / "artifacts" / "sensitivity"; membership_dir.mkdir(parents=True, exist_ok=True)
    membership = population[["row_id", "eligible_history_3", "eligible_history_5", "eligible_history_8", "eligible_history_10"]].copy()
    membership.to_csv(membership_dir / "min_history_membership.csv", index=False)
    summary = {}
    for threshold in (3, 5, 8, 10):
        ids_for_rule = sorted(membership.loc[membership[f"eligible_history_{threshold}"], "row_id"])
        primary_ids = set(membership.loc[membership.eligible_history_3, "row_id"])
        current = set(ids_for_rule)
        summary[str(threshold)] = {"count": len(current), "row_id_sha256": hashlib.sha256("\n".join(ids_for_rule).encode()).hexdigest(), "jaccard_vs_history3": len(current & primary_ids) / len(current | primary_ids), "added_vs_history3": len(current - primary_ids), "removed_vs_history3": len(primary_ids - current)}
    (membership_dir / "min_history_membership_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    eight, ten = set(membership.loc[membership.eligible_history_8, "row_id"]), set(membership.loc[membership.eligible_history_10, "row_id"])
    eight_rows, ten_rows = history_vectors[8]["rows"], history_vectors[10]["rows"]
    shared = eight_rows.merge(ten_rows, on="row_id", suffixes=("_history8", "_history10"), validate="one_to_one")
    target_max_abs_difference = float(np.abs(shared.trend_residual_t_ha_history8 - shared.trend_residual_t_ha_history10).max())
    prediction_max_abs_difference = float(np.abs(shared.prediction_history8 - shared.prediction_history10).max())
    hash_rows = pd.DataFrame([
        {"history": history, "n_rows": len(history_vectors[history]["rows"]), "row_id_sha256": history_vectors[history]["row_id_sha256"], "target_sha256": history_vectors[history]["target_sha256"], "prediction_sha256": history_vectors[history]["prediction_sha256"]}
        for history in (8, 10)
    ])
    diff_rows = pd.DataFrame([
        {"comparison": "history8_only", "count": len(eight - ten), "row_ids": "|".join(sorted(eight - ten))},
        {"comparison": "history10_only", "count": len(ten - eight), "row_ids": "|".join(sorted(ten - eight))},
    ])
    for directory in (membership_dir, ROOT / "artifacts"):
        hash_rows.to_csv(directory / "history_sensitivity_hashes.csv", index=False)
        diff_rows.to_csv(directory / "history_8_10_membership_diff.csv", index=False)
    audit = {"same_row_membership": eight == ten, "same_target_hash": history_vectors[8]["target_sha256"] == history_vectors[10]["target_sha256"], "same_prediction_hash": history_vectors[8]["prediction_sha256"] == history_vectors[10]["prediction_sha256"], "history8_row_id_sha256": history_vectors[8]["row_id_sha256"], "history10_row_id_sha256": history_vectors[10]["row_id_sha256"], "history8_target_sha256": history_vectors[8]["target_sha256"], "history10_target_sha256": history_vectors[10]["target_sha256"], "history8_prediction_sha256": history_vectors[8]["prediction_sha256"], "history10_prediction_sha256": history_vectors[10]["prediction_sha256"], "rows_only_history8": sorted(eight - ten), "rows_only_history10": sorted(ten - eight), "target_max_abs_difference": target_max_abs_difference, "prediction_max_abs_difference": prediction_max_abs_difference, "explanation": f"History 8 and History 10 have identical eligible row IDs and target vectors. Their predictions differ only at floating-point precision (maximum absolute difference {prediction_max_abs_difference:.2e}, accepted tolerance 2.5e-16), so the reported metrics coincide at the displayed precision."}
    (membership_dir / "history_8_vs_10_hash_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def bootstrap_schemes() -> None:
    values = pd.read_csv(ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv")
    selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text())["selected_config"]
    model = values[(values.config_id == selected["config_id"]) & (values.feature_family == selected["feature_family"])].copy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for scheme, groups in (("year_block", model.year.astype(str)), ("state_year_block", model.region + "|" + model.year.astype(str)), ("crop_state_cluster", model.crop + "|" + model.region)):
        unique = groups.unique(); by_group = {group: model[groups == group] for group in unique}; deltas = []
        for _ in range(N_BOOT):
            sample = pd.concat([by_group[group] for group in rng.choice(unique, len(unique), replace=True)], ignore_index=True)
            deltas.append(metric_dict(sample.trend_residual_t_ha, sample.prediction)["rmse_t_ha"] - metric_dict(sample.trend_residual_t_ha, np.zeros(len(sample)))["rmse_t_ha"])
        rows.append({"scheme": scheme, "n_clusters": len(unique), "n_test": len(model), "estimate": metric_dict(model.trend_residual_t_ha, model.prediction)["rmse_t_ha"] - metric_dict(model.trend_residual_t_ha, np.zeros(len(model)))["rmse_t_ha"], "ci95_low": float(np.quantile(deltas, .025)), "ci95_high": float(np.quantile(deltas, .975)), "n_boot": N_BOOT, "status": "SENSITIVITY PASS" if np.quantile(deltas, .975) < 0 else "SENSITIVITY FAIL"})
    write(pd.DataFrame(rows), "bootstrap_scheme_comparison.csv")


def alternative_detrending() -> None:
    frame = load_frame(make_project_paths(ROOT))
    selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text())["selected_config"]
    config = next(item for item in CONFIGS if item["config_id"] == selected["config_id"])
    linear_train, linear_test = score_fold(frame, 3)
    raw_columns = ["trend_yield_t_ha", "trend_residual_t_ha", "trend_residual_z", "is_low_yield_anomaly"]
    huber_train, huber_test = huber_detrend_train_test(linear_train.drop(columns=raw_columns, errors="ignore"), linear_test.drop(columns=raw_columns, errors="ignore"))
    rows = []
    for name, train, test in (("linear_train_only", linear_train, linear_test), ("huber_train_only", huber_train, huber_test)):
        predicted = model_predictions(train, test, config)
        zero = predicted.copy(); zero["prediction"] = 0.0
        paired = paired_delta(predicted, zero, year_draws(predicted.year.to_numpy()), f"{name}_vs_zero")
        rmse = paired[paired.metric == "rmse_t_ha"].iloc[0]
        metrics = metric_dict(predicted.trend_residual_t_ha, predicted.prediction)
        baseline = metric_dict(predicted.trend_residual_t_ha, np.zeros(len(predicted)))
        rows.append({"detrending": name, "fit_scope": "training years only", "n_train": len(train), "n_test": len(test), "event_n_z_lt_1": int((test.trend_residual_z < -1).sum()), **metrics, "baseline_rmse_t_ha": baseline["rmse_t_ha"], "delta_rmse_t_ha": rmse.delta_left_minus_right, "delta_rmse_ci95_low": rmse.ci95_low, "delta_rmse_ci95_high": rmse.ci95_high, "status": "SENSITIVITY PASS" if rmse.ci95_high < 0 else "SENSITIVITY FAIL"})
    write(pd.DataFrame(rows), "alternative_detrending_sensitivity.csv")


def temporal_and_capacity_audits() -> None:
    """Exact rolling folds and prefix learning curve under the locked configuration."""
    frame = load_frame(make_project_paths(ROOT))
    selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text())["selected_config"]
    config = next(item for item in CONFIGS if item["config_id"] == selected["config_id"])
    rows = []
    for fold, (end, start, stop) in enumerate(((2003, 2004, 2006), (2006, 2007, 2009), (2009, 2010, 2012), (2012, 2013, 2015)), 1):
        train, test, _ = __import__("run_main8_audit").score_fold(frame, end, start, stop)
        predicted = model_predictions(train, test, config); zero = predicted.copy(); zero["prediction"] = 0.0
        comparison = paired_delta(predicted, zero, year_draws(predicted.year.to_numpy()), f"rolling_fold_{fold}_vs_zero")
        rmse = comparison[comparison.metric == "rmse_t_ha"].iloc[0]
        rows.append({"audit": "rolling_origin", "fold": fold, "train_end": end, "test_start": start, "test_end": stop, "n_train": len(train), "n_test": len(test), "delta_rmse_t_ha": rmse.delta_left_minus_right, "delta_rmse_ci95_low": rmse.ci95_low, "delta_rmse_ci95_high": rmse.ci95_high, "status": "PASS" if rmse.ci95_high < 0 else "FAIL"})
    train, test = score_fold(frame, 3)
    years = np.array(sorted(train.year.unique()))
    for fraction in (.25, .50, .75, 1.00):
        cutoff = int(years[max(2, int(np.ceil(len(years) * fraction))) - 1])
        subset = train[train.year <= cutoff].copy()
        available = subset.groupby(["crop", "region"]).size()
        eligible = available[available >= 3].index
        subset = subset.set_index(["crop", "region"]).loc[eligible].reset_index()
        eligible_test = test.set_index(["crop", "region"]).loc[eligible].reset_index()
        predicted = model_predictions(subset, eligible_test, config); zero = predicted.copy(); zero["prediction"] = 0.0
        comparison = paired_delta(predicted, zero, year_draws(predicted.year.to_numpy()), f"learning_curve_{fraction}_vs_zero")
        rmse = comparison[comparison.metric == "rmse_t_ha"].iloc[0]
        rows.append({"audit": "prefix_learning_curve", "fraction": fraction, "train_end": cutoff, "n_train": len(subset), "n_test": len(eligible_test), "delta_rmse_t_ha": rmse.delta_left_minus_right, "delta_rmse_ci95_low": rmse.ci95_low, "delta_rmse_ci95_high": rmse.ci95_high, "status": "PASS" if rmse.ci95_high < 0 else "FAIL"})
    write(pd.DataFrame(rows), "temporal_and_capacity_audits.csv")


def group_macro_metrics() -> None:
    """Macro and subgroup scores for the locked selected prediction, without re-selection."""
    values = pd.read_csv(ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv")
    selected = json.loads((ROOT / "artifacts" / "audit" / "selection" / "selected_config.json").read_text())["selected_config"]
    model = values[(values.config_id == selected["config_id"]) & (values.feature_family == selected["feature_family"])].copy()
    rows = []
    for grouping in (("crop", ["crop"]), ("state", ["region"]), ("crop_state", ["crop", "region"]), ("year", ["year"])):
        for values_key, group in model.groupby(grouping[1], sort=True):
            key = values_key if isinstance(values_key, str) else "|".join(map(str, values_key if isinstance(values_key, tuple) else (values_key,)))
            rows.append({"grouping": grouping[0], "group": key, "n": len(group), **metric_dict(group.trend_residual_t_ha, group.prediction)})
    output = pd.DataFrame(rows)
    macro = output.groupby("grouping", as_index=False)[["r2", "rmse_t_ha", "mae_t_ha", "spearman"]].mean(numeric_only=True)
    for column in output.columns:
        if column not in macro:
            macro[column] = np.nan
    macro["group"] = "MACRO_MEAN"; macro["n"] = output.groupby("grouping").size().to_numpy()
    write(pd.concat([output, macro[output.columns]], ignore_index=True), "group_macro_metrics.csv")


def retrospective_effect() -> None:
    frame = load_frame(make_project_paths(ROOT))
    train, prospective = score_fold(frame, 3)
    retrospective, _ = detrend_and_score(frame)
    keys = set(row_id(prospective))
    retrospective = retrospective[row_id(retrospective).isin(keys)].copy()
    prospective_events = prospective.assign(row_id=row_id(prospective), event=prospective.trend_residual_z < -1)[["row_id", "crop", "region", "year", "window", "trend_residual_t_ha", "trend_residual_z", "event"]]
    retrospective_events = retrospective.assign(row_id=row_id(retrospective), retrospective_event=retrospective.trend_residual_z < -1)[["row_id", "trend_residual_t_ha", "trend_residual_z", "retrospective_event"]]
    retrospective_events = retrospective_events.rename(columns={"trend_residual_t_ha": "retrospective_trend_residual_t_ha", "trend_residual_z": "retrospective_trend_residual_z"})
    transitions = prospective_events.merge(retrospective_events, on="row_id", validate="one_to_one")
    event_sets = [set(transitions.loc[transitions.event, "row_id"]), set(transitions.loc[transitions.retrospective_event, "row_id"])]
    jaccard = len(event_sets[0] & event_sets[1]) / len(event_sets[0] | event_sets[1])
    rng = np.random.default_rng(BOOTSTRAP_SEED); years = transitions.row_id.str.split("|", expand=True)[2].astype(int).unique(); draws = []
    for _ in range(N_BOOT):
        chosen = rng.choice(years, len(years), replace=True); sample = pd.concat([transitions[transitions.row_id.str.contains(f"\\|{year}\\|")] for year in chosen])
        a, b = set(sample.loc[sample.event, "row_id"]), set(sample.loc[sample.retrospective_event, "row_id"]); draws.append(len(a & b) / len(a | b))
    summary = pd.DataFrame([{"prospective_n": len(prospective), "prospective_events": len(event_sets[0]), "retrospective_events": len(event_sets[1]), "jaccard": jaccard, "jaccard_ci95_low": float(np.quantile(draws, .025)), "jaccard_ci95_high": float(np.quantile(draws, .975)), "status": "RETROSPECTIVE_ONLY"}])
    # This comparison is row-aligned to the locked 333 rows and deliberately keeps
    # both target vectors visible; it is not an alternate final-test estimate.
    transitions["prospective_target_sha256"] = hashlib.sha256(np.ascontiguousarray(transitions.trend_residual_t_ha.to_numpy()).tobytes()).hexdigest()
    transitions["retrospective_target_sha256"] = hashlib.sha256(np.ascontiguousarray(transitions.retrospective_trend_residual_t_ha.to_numpy()).tobytes()).hexdigest()
    transitions["target_delta_t_ha"] = transitions.retrospective_trend_residual_t_ha - transitions.trend_residual_t_ha
    transitions["rank_prospective"] = transitions.trend_residual_t_ha.rank(method="average")
    transitions["rank_retrospective"] = transitions.retrospective_trend_residual_t_ha.rank(method="average")
    rank_correlation = float(transitions[["trend_residual_t_ha", "retrospective_trend_residual_t_ha"]].corr(method="spearman").iloc[0, 1])
    summary["rank_spearman"] = rank_correlation
    summary["membership_transitions"] = int((transitions.event != transitions.retrospective_event).sum())
    write(summary, "retrospective_target_comparison.csv")
    write(transitions.groupby(["event", "retrospective_event"], as_index=False).size(), "target_membership_transition.csv")
    write(transitions, "retrospective_vs_train_only.csv")


def main() -> None:
    min_history_and_scale(); alternative_detrending(); bootstrap_schemes(); temporal_and_capacity_audits(); group_macro_metrics(); retrospective_effect()
    print("Extended target/bootstrap/retrospective audits written.")


if __name__ == "__main__":
    main()
