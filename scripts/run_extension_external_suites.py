"""Run locked EXT-SEMI and EXT-PJM suites after the crop winner is locked."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.extension_factorial import (  # noqa: E402
    estimator,
    feature_columns,
    hierarchy_predictions,
    load_protocol,
    preprocessing,
    representation_frame,
)


OUT = ROOT / "artifacts" / "extensions" / "v1"
SEMI = OUT / "semi"
PJM = OUT / "pjm"
KEYS = ["country", "region", "crop", "year", "window"]


def require_lock() -> dict[str, Any]:
    lock = json.loads((OUT / "core" / "winner_lock.json").read_text())
    if lock["status"] != "LOCKED" or not lock.get("winner"):
        raise AssertionError("External suites require a locked crop winner")
    return lock


def selected_config(arm_id: str, model_id: str, family: str) -> dict[str, Any]:
    selected = pd.read_csv(OUT / "core" / "selected_inner_configs.csv")
    row = selected[
        (selected.arm_id == arm_id)
        & (selected.model_id == model_id)
        & (selected.feature_family == family)
        & (selected.outer_fold == "O6")
    ].iloc[0]
    protocol = load_protocol(str(ROOT))
    for grid in protocol["model_configs"][model_id].values():
        for config in grid:
            if config["config_id"] == row.config_id:
                return config
    raise KeyError(row.config_id)


def candidate_registry() -> list[dict[str, Any]]:
    lock = require_lock()
    protocol = load_protocol(str(ROOT))
    arms = {arm["arm_id"]: arm for arm in protocol["arms"]}
    registry = []
    for role, record in [
        ("winner", lock["winner"]),
        *[(f"runner_up_{index}", item) for index, item in enumerate(lock["runners_up"], 1)],
    ]:
        arm = arms[record["arm_id"]]
        registry.append(
            {
                "candidate_id": role,
                **arm,
                "model_id": record["model_id"],
                "feature_family": record["feature_family"],
                "full_config": selected_config(record["arm_id"], record["model_id"], record["feature_family"]),
                "metadata_config": selected_config(record["arm_id"], record["model_id"], "metadata_only"),
            }
        )
    registry.append(
        {
            "candidate_id": "kse_primary",
            "arm_id": "KSE_PRIMARY",
            "representation": "BASE",
            "hierarchy": "FLAT",
            "grid": "LOCKED_KSE",
            "model_id": "catboost",
            "feature_family": "weather_only",
            "full_config": {
                "config_id": "KSE_CAT",
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3,
                "iterations": 400,
            },
            "metadata_config": {
                "config_id": "KSE_CAT_META",
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3,
                "iterations": 400,
            },
        }
    )
    return registry


def standardized(values: pd.Series, train: pd.Series) -> np.ndarray:
    scale = float(train.std(ddof=1))
    if not np.isfinite(scale) or scale == 0:
        return np.zeros(len(values))
    return (values.to_numpy(dtype=float) - float(train.mean())) / scale


def semi_source() -> pd.DataFrame:
    frame, _ = representation_frame(str(ROOT), "STAGE-HYBRID", 2023)
    train = frame.year <= 2023
    signal = (
        -0.50 * standardized(frame["heat_days_35_cal_mid"], frame.loc[train, "heat_days_35_cal_mid"])
        + 0.30 * standardized(frame["rain_sum_cal_mid"], frame.loc[train, "rain_sum_cal_mid"])
        - 0.20 * standardized(frame["max_3day_rain_gdd_late"], frame.loc[train, "max_3day_rain_gdd_late"])
    )
    frame = frame[KEYS].copy()
    frame["known_weather_component"] = signal
    return frame


def degrade_weather(
    frame: pd.DataFrame,
    weather: list[str],
    measurement_error: float,
    spatial_mismatch: float,
    seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    train = result.year <= 2023
    rng = np.random.default_rng(seed)
    for column in weather:
        scale = float(result.loc[train, column].std(ddof=1))
        if measurement_error and np.isfinite(scale) and scale > 0:
            result[column] = result[column] + rng.normal(
                0,
                measurement_error * scale,
                len(result),
            )
    if spatial_mismatch > 0:
        quantiles = result.loc[train, "lat"].quantile([1 / 3, 2 / 3]).to_numpy()
        result["climate_stratum"] = pd.cut(
            result["lat"],
            [-np.inf, quantiles[0], quantiles[1], np.inf],
            labels=False,
            include_lowest=True,
        )
        count = int(round(len(result) * spatial_mismatch))
        selected = rng.choice(result.index, count, replace=False)
        for index in selected:
            row = result.loc[index]
            donors = result[
                (result.year == row.year)
                & (result.climate_stratum == row.climate_stratum)
                & (result.region != row.region)
            ]
            if donors.empty:
                continue
            donor = donors.iloc[int(rng.integers(len(donors)))]
            result.loc[index, weather] = donor[weather].to_numpy()
        result = result.drop(columns="climate_stratum")
    return result


def fit_semisynthetic(task: dict[str, Any]) -> dict[str, Any]:
    candidate = task["candidate"]
    frame, weather = representation_frame(
        str(ROOT), candidate["representation"], 2023
    )
    truth_source = semi_source()
    frame = frame.merge(truth_source, on=KEYS, validate="one_to_one")
    frame = degrade_weather(
        frame,
        weather,
        task["measurement_error"],
        task["spatial_mismatch"],
        task["seed"],
    )
    train_mask = frame.year <= 2023
    test_mask = frame.year.between(2024, 2025)
    train = frame[train_mask].copy()
    test = frame[test_mask].copy()
    rng = np.random.default_rng(task["seed"] + 100003)
    base_scale = 0.45
    group_effect = {
        key: value
        for key, value in zip(
            train[["crop", "region"]].drop_duplicates().itertuples(index=False, name=None),
            rng.normal(0, 0.12, train[["crop", "region"]].drop_duplicates().shape[0]),
        )
    }
    mde = 0.20 * base_scale
    def target(part: pd.DataFrame) -> np.ndarray:
        group = np.asarray([group_effect.get((r.crop, r.region), 0.0) for r in part.itertuples()])
        local_rng = np.random.default_rng(task["seed"] + int(part.year.min()) * 97)
        return (
            group
            + task["signal_strength"] * mde * part["known_weather_component"].to_numpy()
            + local_rng.normal(0, base_scale, len(part))
        )
    train["trend_residual_t_ha"] = target(train)
    test["trend_residual_t_ha"] = target(test)
    hierarchy_train = np.zeros(len(train))
    hierarchy_test = np.zeros(len(test))
    if candidate["hierarchy"] == "HIER-RESIDUAL":
        hierarchy_train, hierarchy_test, _ = hierarchy_predictions(train, test)
    full_numeric, full_categorical = feature_columns(candidate["feature_family"], weather)
    metadata_numeric, metadata_categorical = feature_columns("metadata_only", weather)

    def predict(numeric, categorical, config, hierarchy=False):
        model = Pipeline(
            [
                ("preprocess", preprocessing(numeric, categorical)),
                ("model", estimator(candidate["model_id"], config, task["seed"])),
            ]
        )
        offset_train = hierarchy_train if hierarchy else 0.0
        offset_test = hierarchy_test if hierarchy else 0.0
        model.fit(
            train[numeric + categorical],
            train["trend_residual_t_ha"].to_numpy() - offset_train,
        )
        return np.asarray(model.predict(test[numeric + categorical])) + offset_test

    prediction = predict(
        full_numeric,
        full_categorical,
        candidate["full_config"],
        candidate["hierarchy"] == "HIER-RESIDUAL",
    )
    metadata = predict(
        metadata_numeric,
        metadata_categorical,
        candidate["metadata_config"],
        candidate["hierarchy"] == "HIER-RESIDUAL",
    )
    y = test["trend_residual_t_ha"].to_numpy()
    zero = np.zeros(len(y))
    rmse = lambda p: float(mean_squared_error(y, p) ** 0.5)
    return {
        "base_id": task["base_id"],
        "candidate_id": candidate["candidate_id"],
        "arm_id": candidate["arm_id"],
        "model_id": candidate["model_id"],
        "signal_strength_mde": task["signal_strength"],
        "measurement_error_sd": task["measurement_error"],
        "spatial_mismatch_fraction": task["spatial_mismatch"],
        "seed": task["seed"],
        "n_test": len(test),
        "rmse_model": rmse(prediction),
        "rmse_metadata": rmse(metadata),
        "rmse_zero": rmse(zero),
        "module_a_point_pass": rmse(prediction) < rmse(zero),
        "module_b_point_pass": rmse(prediction) < rmse(metadata),
        "y_json": json.dumps(y.tolist()),
        "prediction_json": json.dumps(prediction.tolist()),
    }


def run_semi(workers: int) -> None:
    SEMI.mkdir(parents=True, exist_ok=True)
    checkpoint = SEMI / "base_fit_results.parquet"
    candidates = candidate_registry()
    tasks = []
    for signal in [0, 0.25, 0.5, 1.0, 1.5]:
        for error in [0, 0.25, 0.5, 1.0]:
            for mismatch in [0, 0.25, 0.5, 1.0]:
                for seed in range(30):
                    for candidate in candidates:
                        identity = f"{signal}|{error}|{mismatch}|{seed}|{candidate['candidate_id']}"
                        tasks.append(
                            {
                                "base_id": sha256(identity.encode()).hexdigest()[:20],
                                "signal_strength": signal,
                                "measurement_error": error,
                                "spatial_mismatch": mismatch,
                                "seed": 71000 + seed,
                                "candidate": candidate,
                            }
                        )
    existing = pd.read_parquet(checkpoint) if checkpoint.exists() else pd.DataFrame()
    done = set(existing.base_id) if len(existing) else set()
    pending = [task for task in tasks if task["base_id"] not in done]
    records = existing.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(fit_semisynthetic, pending), 1):
            records.append(result)
            if index % 50 == 0 or index == len(pending):
                pd.DataFrame(records).to_parquet(checkpoint, index=False)
                print(f"EXT-SEMI base fits: {len(done)+index}/{len(tasks)}", flush=True)
    base = pd.DataFrame(records)
    rows = []
    prevalence_factors = {"base": 1.0, "half": 0.5, "quarter": 0.25}
    for record in base.to_dict("records"):
        y = np.asarray(json.loads(record.pop("y_json")))
        prediction = np.asarray(json.loads(record.pop("prediction_json")))
        for label, factor in prevalence_factors.items():
            prevalence = max(1 / len(y), 0.16 * factor)
            threshold = float(np.quantile(y, prevalence))
            tail = y <= threshold
            k = max(1, int(tail.sum()))
            observed = set(np.argsort(y)[:k])
            predicted = set(np.argsort(prediction)[:k])
            overlap = len(observed & predicted)
            lift = overlap * len(y) / (k * k)
            tail_rmse_model = mean_squared_error(y[tail], prediction[tail]) ** 0.5
            tail_rmse_zero = mean_squared_error(y[tail], np.zeros(tail.sum())) ** 0.5
            rho = float(spearmanr(y[tail], prediction[tail]).statistic) if tail.sum() > 2 else 0.0
            e_pass = tail_rmse_model < tail_rmse_zero and rho > 0 and lift > 1
            valid = record["signal_strength_mde"] > 0
            permission = bool(record["module_a_point_pass"] and record["module_b_point_pass"] and e_pass)
            rows.append(
                {
                    **record,
                    "event_prevalence": label,
                    "event_n": int(tail.sum()),
                    "event_spearman": rho,
                    "event_topk_lift": float(lift),
                    "module_e_point_pass": e_pass,
                    "policy_permission": permission,
                    "ground_truth_valid": valid,
                    "correct_permission": bool(valid and permission),
                    "correct_stop": bool((not valid) and (not permission)),
                    "ground_truth_used_as_model_input": False,
                }
            )
    results = pd.DataFrame(rows)
    if len(results) != 28800:
        raise AssertionError(f"Expected 28,800 EXT-SEMI rows, found {len(results)}")
    results.to_csv(SEMI / "scenario_results.csv", index=False)


def pjm_degraded(frame: pd.DataFrame, mechanism: str, level: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    train = frame.date < "2024-10-01"
    raw_forecast = frame["forecast"].astype(float).copy()
    train_median = float(raw_forecast.loc[train].median())
    forecast = raw_forecast.fillna(train_median)
    if mechanism == "permutation_mix":
        permuted = forecast.copy()
        strata = frame.date.dt.dayofweek
        for is_train in [True, False]:
            for value in sorted(strata.unique()):
                index = frame.index[(train == is_train) & (strata == value)]
                permuted.loc[index] = rng.permutation(forecast.loc[index].to_numpy())
        return (1 - level) * forecast + level * permuted
    if mechanism == "noise":
        scale = float(forecast.loc[train].std(ddof=1))
        return forecast + rng.normal(0, level * scale, len(frame))
    if mechanism == "dropout":
        output = forecast.copy()
        mask = rng.random(len(frame)) < level
        output.loc[mask] = np.nan
        return output.fillna(train_median)
    if mechanism == "lag_substitution":
        return forecast.shift(int(level)).fillna(train_median)
    raise ValueError(mechanism)


def run_pjm() -> None:
    PJM.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        ROOT / "artifacts" / "experiments" / "external-domain-eia" / "pjm_daily_2024.csv",
        parse_dates=["date"],
    )
    train = frame.date < "2024-10-01"
    test = ~train
    calendar = ["dow", "sin_doy", "cos_doy"]
    levels = {
        "permutation_mix": [0, 0.1, 0.25, 0.5, 0.75, 1],
        "noise": [0, 0.25, 0.5, 1, 2],
        "dropout": [0, 0.1, 0.25, 0.5, 0.75],
        "lag_substitution": [0, 1, 2, 7],
    }
    rows = []
    y = frame.loc[test, "target"].to_numpy()
    naive = np.repeat(float(frame.loc[train, "target"].mean()), len(y))
    for mechanism, values in levels.items():
        for level in values:
            for seed in range(30):
                work = frame.copy()
                work["degraded_forecast"] = pjm_degraded(work, mechanism, level, 81000 + seed)
                calendar_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=23, n_jobs=1)
                full_model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=23, n_jobs=1)
                calendar_model.fit(work.loc[train, calendar], work.loc[train, "target"])
                full_model.fit(work.loc[train, calendar + ["degraded_forecast"]], work.loc[train, "target"])
                calendar_prediction = calendar_model.predict(work.loc[test, calendar])
                full_prediction = full_model.predict(work.loc[test, calendar + ["degraded_forecast"]])
                rng = np.random.default_rng(91000 + seed)
                draws_a, draws_b = [], []
                for _ in range(2000):
                    index = rng.integers(0, len(y), len(y))
                    rmse = lambda p: mean_squared_error(y[index], p[index]) ** 0.5
                    draws_a.append(rmse(full_prediction) - rmse(naive))
                    draws_b.append(rmse(full_prediction) - rmse(calendar_prediction))
                rows.append(
                    {
                        "mechanism": mechanism,
                        "level": level,
                        "seed": seed,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "delta_rmse_a": float(mean_squared_error(y, full_prediction) ** 0.5 - mean_squared_error(y, naive) ** 0.5),
                        "delta_rmse_b": float(mean_squared_error(y, full_prediction) ** 0.5 - mean_squared_error(y, calendar_prediction) ** 0.5),
                        "module_a_ci95_high": float(np.quantile(draws_a, 0.975)),
                        "module_b_ci95_high": float(np.quantile(draws_b, 0.975)),
                        "module_a_pass": float(np.quantile(draws_a, 0.975)) < 0,
                        "module_b_pass": float(np.quantile(draws_b, 0.975)) < 0,
                    }
                )
    scenarios = pd.DataFrame(rows)
    scenarios.to_csv(PJM / "scenario_results.csv", index=False)
    curve = scenarios.groupby(["mechanism", "level"], as_index=False).agg(
        mean_delta_rmse_b=("delta_rmse_b", "mean"),
        sd_delta_rmse_b=("delta_rmse_b", "std"),
        module_a_pass_probability=("module_a_pass", "mean"),
        module_b_pass_probability=("module_b_pass", "mean"),
        seeds=("seed", "nunique"),
    )
    curve.to_csv(PJM / "degradation_curve.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["semi", "pjm"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    require_lock()
    if args.suite == "semi":
        run_semi(args.workers)
    else:
        run_pjm()


if __name__ == "__main__":
    main()
