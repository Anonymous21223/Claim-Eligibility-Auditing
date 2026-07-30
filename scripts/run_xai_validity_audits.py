"""Run non-claiming XAI validity audits on the locked temporal split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_yield_xai.core import full_season_weather_features, group_features, load_frame, make_project_paths  # noqa: E402
from run_revision_audit import FINAL_TEST_YEARS, FINAL_TRAIN_END, SEEDS, metrics, model_pipeline, score_fold  # noqa: E402


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def grouped_shap_audits(train: pd.DataFrame, test: pd.DataFrame, features: list[str], groups: dict[str, list[str]], out: Path) -> None:
    numeric, categorical = ["lat", "lon"] + features, ["crop", "region"]
    model = model_pipeline(numeric, categorical, "extra_trees", SEEDS[0])
    model.fit(train[numeric + categorical], train["trend_residual_t_ha"])
    transformed = model.named_steps["preprocess"].transform(test[numeric + categorical])
    names = [str(name).replace("numeric__", "").replace("category__", "") for name in model.named_steps["preprocess"].get_feature_names_out()]
    values = np.asarray(shap.TreeExplainer(model.named_steps["model"]).shap_values(transformed), dtype=float)
    expected = float(np.asarray(shap.TreeExplainer(model.named_steps["model"]).expected_value).reshape(-1)[0])
    index = {name: position for position, name in enumerate(names)}

    rows = []
    rng = np.random.default_rng(20260718)
    k = min(len(items) for items in groups.values())
    for group, items in groups.items():
        positions = [index[item] for item in items]
        total = np.abs(values[:, positions]).sum(axis=1)
        rows.append({"protocol": "sum_abs", "driver_group": group, "n_features": len(items), "importance": float(total.mean())})
        rows.append({"protocol": "mean_abs_per_feature", "driver_group": group, "n_features": len(items), "importance": float((total / len(items)).mean())})
        draws = []
        for _ in range(100):
            selected = rng.choice(positions, size=k, replace=False)
            draws.append(float(np.abs(values[:, selected]).sum(axis=1).mean()))
        rows.append({"protocol": f"balanced_random_k_{k}", "driver_group": group, "n_features": len(items), "importance": float(np.mean(draws)), "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975))})
    sensitivity = pd.DataFrame(rows)
    sensitivity["rank"] = sensitivity.groupby("protocol")["importance"].rank(ascending=False, method="first").astype(int)
    write(sensitivity, out / "group_size_sensitivity.csv")
    write(sensitivity[sensitivity.protocol.str.startswith("balanced")], out / "balanced_group_results.csv")

    family = {
        "weather": [index[name] for name in features],
        "location": [index[name] for name in ["lat", "lon"]],
        "crop": [i for i, name in enumerate(names) if name.startswith("crop_")],
        "state": [i for i, name in enumerate(names) if name.startswith("region_")],
    }
    family_rows = [{"feature_family": name, "mean_abs_shap": float(np.abs(values[:, positions]).sum(axis=1).mean())} for name, positions in family.items()]
    family_frame = pd.DataFrame(family_rows)
    family_frame["share_of_total"] = family_frame.mean_abs_shap / family_frame.mean_abs_shap.sum()
    write(family_frame, out / "shap_family_share.csv")

    prediction = model.predict(test[numeric + categorical])
    group_local = []
    for row_idx, source in test.reset_index(drop=True).iterrows():
        for group, items in groups.items():
            signed = float(values[row_idx, [index[item] for item in items]].sum())
            group_local.append({"row_id": f"{source.crop}|{source.region}|{int(source.year)}|{source.window}", "crop": source.crop, "region": source.region, "year": int(source.year), "driver_group": group, "signed_group_shap": signed, "adverse_contribution": max(0.0, -signed), "predicted_residual": float(prediction[row_idx]), "observed_residual": float(source.trend_residual_t_ha), "base_value": expected, "reconstruction_error": float(expected + values[row_idx].sum() - prediction[row_idx])})
    local = pd.DataFrame(group_local)
    write(local, out / "local_case_decomposition.csv")
    reconstruction = local.groupby("row_id", as_index=False).agg(max_abs_reconstruction_error=("reconstruction_error", lambda x: float(np.max(np.abs(x)))), predicted_residual=("predicted_residual", "first"), observed_residual=("observed_residual", "first"))
    reconstruction["same_sign"] = np.sign(reconstruction.predicted_residual) == np.sign(reconstruction.observed_residual)
    write(reconstruction, out / "shap_reconstruction_checks.csv")
    case_log = reconstruction.assign(status=np.where(reconstruction.same_sign, "not_used_fidelity_gate_failed", "excluded_wrong_prediction_sign"), reason="Final-test residual model did not meet the pre-specified fidelity gate.")
    write(case_log, out / "case_selection_log.csv")
    (out / "shap_config.yaml").write_text("explainer: TreeExplainer\nmodel: ExtraTreesRegressor\nfeature_space: post-preprocessing\nbackground: tree_path_dependent_default\noutput: residual_t_ha\nsplit: locked_2016_2025\n", encoding="utf-8")


def stratified_permutation(train: pd.DataFrame, test: pd.DataFrame, features: list[str], groups: dict[str, list[str]], out: Path) -> None:
    numeric, categorical = ["lat", "lon"] + features, ["crop", "region"]
    model = model_pipeline(numeric, categorical, "extra_trees", SEEDS[0])
    model.fit(train[numeric + categorical], train["trend_residual_t_ha"])
    baseline = model.predict(test[numeric + categorical])
    base_rmse = metrics(test["trend_residual_t_ha"], baseline)["rmse_t_ha"]
    protocols = {"global": [], "within_crop": ["crop"], "within_crop_window": ["crop", "window"], "within_state": ["region"], "within_year": ["year"]}
    rng, rows, diagnostics = np.random.default_rng(20260718), [], []
    for protocol, columns in protocols.items():
        for group, items in groups.items():
            scores = []
            for _ in range(20):
                shuffled = test.copy()
                if columns:
                    for _, position in test.groupby(columns, dropna=False).groups.items():
                        pos = np.asarray(list(position))
                        shuffled.loc[pos, items] = test.loc[pos, items].iloc[rng.permutation(len(pos))].to_numpy()
                else:
                    shuffled.loc[:, items] = test[items].iloc[rng.permutation(len(test))].to_numpy()
                scores.append(metrics(test["trend_residual_t_ha"], model.predict(shuffled[numeric + categorical]))["rmse_t_ha"] - base_rmse)
            rows.append({"protocol": protocol, "driver_group": group, "n_features": len(items), "baseline_rmse_t_ha": base_rmse, "rmse_increase_t_ha": float(np.mean(scores)), "ci95_low": float(np.quantile(scores, .025)), "ci95_high": float(np.quantile(scores, .975)), "repeats": 20})
            diagnostics.append({"protocol": protocol, "driver_group": group, "mean_abs_feature_shift": float(np.mean(np.abs(test[items].to_numpy() - shuffled[items].to_numpy()))), "n_rows": len(test)})
    result = pd.DataFrame(rows)
    result["rank"] = result.groupby("protocol")["rmse_increase_t_ha"].rank(ascending=False, method="first").astype(int)
    write(result, out / "conditional_permutation_results.csv")
    write(result, out / "permutation_protocols.csv")
    write(pd.DataFrame(diagnostics), out / "permutation_ood_diagnostics.csv")


def correlation_audit(train: pd.DataFrame, features: list[str], out: Path) -> None:
    corr = train[features].corr().abs()
    pairs = []
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            if corr.loc[left, right] >= .85:
                pairs.append({"feature_a": left, "feature_b": right, "abs_correlation": float(corr.loc[left, right])})
    write(pd.DataFrame(pairs), out / "correlation_clusters.csv")


def main() -> None:
    frame = load_frame(make_project_paths(ROOT))
    train, test, _ = score_fold(frame, FINAL_TRAIN_END, *FINAL_TEST_YEARS)
    features = full_season_weather_features(frame)
    groups = group_features(features)
    out = ROOT / "artifacts" / "xai"
    grouped_shap_audits(train, test, features, groups, out)
    stratified_permutation(train, test, features, groups, out)
    correlation_audit(train, features, out)
    print(f"XAI validity audits written to {out}")


if __name__ == "__main__":
    main()
