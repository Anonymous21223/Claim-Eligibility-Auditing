"""Run the locked V2 model ladder with validation-only selection and one temporal holdout evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2_county" / "processed" / "county_winter_wheat_weather_panel.csv"
SPEC = ROOT / "configs" / "experiments" / "county_v2_model_spec.json"
OUT = ROOT / "artifacts" / "experiments" / "county-v2-weather-models"
REPORTS = ROOT / "reports" / "experiments"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, prediction)))


def add_residual(reference: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    coefficients: dict[str, tuple[float, float]] = {}
    for county, group in reference.groupby("county_fips"):
        coefficients[str(county)] = tuple(np.polyfit(group["year"], group["yield_bu_acre"], 1))
    output = target.copy()
    trend = output.apply(lambda row: coefficients[str(row.county_fips)][0] * row.year + coefficients[str(row.county_fips)][1], axis=1)
    output["trend_yield_bu_acre"] = trend.astype(float)
    output["target_residual_bu_acre"] = output["yield_bu_acre"] - output["trend_yield_bu_acre"]
    return output


def make_model(candidate: dict[str, object], weather: list[str], metadata: list[str], seed: int) -> Pipeline:
    family = str(candidate["family"])
    numeric = weather if family == "weather_only" else metadata[1:] if family == "metadata_only" else [*weather, *metadata[1:]]
    categorical = [] if family == "weather_only" else ["county_fips"]
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("county", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    if candidate["model"] == "Ridge":
        estimator = Ridge(alpha=float(candidate["alpha"]))
    else:
        estimator = ExtraTreesRegressor(
            n_estimators=int(candidate["n_estimators"]),
            min_samples_leaf=int(candidate["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    return Pipeline([("features", preprocess), ("model", estimator)])


def evaluate(candidates: list[dict[str, object]], train: pd.DataFrame, test: pd.DataFrame, weather: list[str], metadata: list[str], seed: int) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows = []
    fitted: dict[str, Pipeline] = {}
    for candidate in candidates:
        model = make_model(candidate, weather, metadata, seed)
        family = str(candidate["family"])
        features = weather if family == "weather_only" else metadata if family == "metadata_only" else [*weather, *metadata]
        model.fit(train[features], train["target_residual_bu_acre"])
        prediction = model.predict(test[features])
        rows.append({
            "config_id": candidate["id"],
            "family": family,
            "model": candidate["model"],
            "rmse_bu_acre": rmse(test["target_residual_bu_acre"], prediction),
            "r2": float(r2_score(test["target_residual_bu_acre"], prediction)),
            "n": len(test),
        })
        fitted[str(candidate["id"])] = model
    return pd.DataFrame(rows).sort_values("rmse_bu_acre"), fitted


def block_bootstrap(y: np.ndarray, left: np.ndarray, right: np.ndarray, years: np.ndarray, seed: int, draws: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    blocks = {year: np.flatnonzero(years == year) for year in sorted(set(years))}
    values = []
    for _ in range(draws):
        sampled = np.concatenate([blocks[year] for year in rng.choice(list(blocks), size=len(blocks), replace=True)])
        values.append(rmse(y[sampled], left[sampled]) - rmse(y[sampled], right[sampled]))
    return {"point_delta_rmse": rmse(y, left) - rmse(y, right), "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)), "draws": draws}


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA, dtype={"county_fips": str})
    train_raw = frame[frame["split_role"].eq("train")].copy()
    validation_raw = frame[frame["split_role"].eq("validation")].copy()
    holdout_raw = frame[frame["split_role"].eq("locked_holdout")].copy()
    train = add_residual(train_raw, train_raw)
    validation = add_residual(train_raw, validation_raw)
    weather = list(spec["weather_features"])
    metadata = list(spec["metadata_features"])
    validation_metrics, _ = evaluate(spec["candidates"], train, validation, weather, metadata, int(spec["seed"]))
    selected_id = str(validation_metrics.iloc[0]["config_id"])
    selected_spec = next(item for item in spec["candidates"] if item["id"] == selected_id)

    final_reference_raw = pd.concat([train_raw, validation_raw], ignore_index=True)
    final_train = add_residual(final_reference_raw, final_reference_raw)
    holdout = add_residual(final_reference_raw, holdout_raw)
    final_metrics, final_models = evaluate(spec["candidates"], final_train, holdout, weather, metadata, int(spec["seed"]))
    family = str(selected_spec["family"])
    features = weather if family == "weather_only" else metadata if family == "metadata_only" else [*weather, *metadata]
    selected_prediction = final_models[selected_id].predict(holdout[features])
    zero_prediction = np.zeros(len(holdout))
    same_model_metadata = next(item for item in spec["candidates"] if item["model"] == selected_spec["model"] and item["family"] == "metadata_only")
    same_model_full = next(item for item in spec["candidates"] if item["model"] == selected_spec["model"] and item["family"] == "full")
    metadata_prediction = final_models[str(same_model_metadata["id"])].predict(holdout[metadata])
    full_prediction = final_models[str(same_model_full["id"])].predict(holdout[[*weather, *metadata]])
    gate_a = block_bootstrap(holdout["target_residual_bu_acre"].to_numpy(), selected_prediction, zero_prediction, holdout["year"].to_numpy(), int(spec["seed"]))
    gate_b1 = block_bootstrap(holdout["target_residual_bu_acre"].to_numpy(), full_prediction, metadata_prediction, holdout["year"].to_numpy(), int(spec["seed"]))
    actual_abs = np.abs(holdout["target_residual_bu_acre"].to_numpy())
    predicted_abs = np.abs(selected_prediction)
    k = max(1, int(np.ceil(0.1 * len(holdout))))
    true_tail = set(np.argsort(actual_abs)[-k:])
    ranked = set(np.argsort(predicted_abs)[-k:])
    tail_precision = len(true_tail & ranked) / k
    status = "CANDIDATE" if gate_a["ci95_high"] < 0 and tail_precision > 0.10 else "INCONCLUSIVE"
    explanation = "INTERPRET" if family == "full" and gate_a["ci95_high"] < 0 and gate_b1["ci95_high"] < 0 else "ABSTAIN"
    OUT.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_csv(OUT / "validation_metrics.csv", index=False)
    final_metrics.to_csv(OUT / "locked_holdout_all_candidates.csv", index=False)
    predictions = holdout[["county_fips", "year", "yield_bu_acre", "trend_yield_bu_acre", "target_residual_bu_acre"]].copy()
    predictions["selected_prediction_residual"] = selected_prediction
    predictions["zero_prediction_residual"] = zero_prediction
    predictions["metadata_prediction_residual"] = metadata_prediction
    predictions.to_csv(OUT / "locked_holdout_predictions.csv", index=False)
    summary = {
        "status": status,
        "explanation_availability": explanation,
        "data": DATA.relative_to(ROOT).as_posix(),
        "data_sha256": digest(DATA),
        "spec": SPEC.relative_to(ROOT).as_posix(),
        "selected_on_validation": selected_id,
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "holdout_selected_rmse_bu_acre": rmse(holdout["target_residual_bu_acre"], selected_prediction),
        "holdout_zero_rmse_bu_acre": rmse(holdout["target_residual_bu_acre"], zero_prediction),
        "gate_a_selected_vs_zero": gate_a,
        "gate_b1_weather_increment": gate_b1,
        "tail_precision_at_10pct": tail_precision,
        "tail_chance_precision": 0.10,
        "holdout_rows": len(holdout),
        "holdout_years": sorted(holdout["year"].unique().tolist()),
        "prediction_path": (OUT / "locked_holdout_predictions.csv").relative_to(ROOT).as_posix(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "county-v2-weather-models.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# V2 County Weather Model Experiment", "", *[f"- {key}: `{value}`" for key, value in summary.items()], "", "The temporal holdout was not used to select the candidate. Feature attributions are unavailable unless both registered gates pass."]
    (REPORTS / "county-v2-weather-models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["status", "selected_on_validation", "holdout_selected_rmse_bu_acre", "holdout_zero_rmse_bu_acre", "explanation_availability"]}))


if __name__ == "__main__":
    main()
