"""Leakage-safe model utilities for extension-factorial-v1."""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crop_yield_xai.core import (
    detrend_train_test,
    full_season_weather_features,
)


KEYS = ["country", "region", "crop", "year", "window"]
CAT_COLUMNS = ["crop", "region", "window"]
STAGE_KEYS = ["crop", "region", "year", "window"]
STOCHASTIC_MODELS = {"extra_trees", "lightgbm", "catboost"}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_protocol(root_text: str) -> dict[str, Any]:
    root = Path(root_text)
    return json.loads(
        (root / "configs" / "extension_factorial_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


@lru_cache(maxsize=1)
def load_base_frame(root_text: str) -> pd.DataFrame:
    root = Path(root_text)
    protocol = load_protocol(root_text)
    frame = pd.read_csv(root / protocol["data"]["processed_reference"])
    if len(frame) != 1257 or frame[KEYS].duplicated().any():
        raise AssertionError("Unexpected base panel shape or duplicate keys")
    return frame


@lru_cache(maxsize=16)
def load_stage_frame(root_text: str, cutoff: int) -> pd.DataFrame:
    path = (
        Path(root_text)
        / "data"
        / "derived"
        / "stage_features"
        / f"cutoff_{cutoff}"
        / "stage_features.csv"
    )
    frame = pd.read_csv(path)
    if len(frame) != 1257 or frame[STAGE_KEYS].duplicated().any():
        raise AssertionError(f"Invalid stage frame for cutoff {cutoff}")
    if frame.isna().any().any():
        raise AssertionError(f"Missing stage features for cutoff {cutoff}")
    return frame


def representation_frame(
    root_text: str,
    representation: str,
    cutoff: int,
) -> tuple[pd.DataFrame, list[str]]:
    base = load_base_frame(root_text).copy()
    base_weather = full_season_weather_features(base)
    if representation == "BASE":
        return base, base_weather
    stage = load_stage_frame(root_text, cutoff)
    stage_columns = [column for column in stage if "_cal_" in column or "_gdd_" in column]
    if representation == "STAGE-CAL":
        selected = [column for column in stage_columns if "_cal_" in column]
    elif representation == "STAGE-GDD":
        selected = [column for column in stage_columns if "_gdd_" in column]
    elif representation == "STAGE-HYBRID":
        selected = stage_columns
    else:
        raise ValueError(f"Unknown representation: {representation}")
    merged = base.merge(
        stage[STAGE_KEYS + selected],
        on=STAGE_KEYS,
        how="left",
        validate="one_to_one",
    )
    if merged[selected].isna().any().any():
        raise AssertionError(f"Stage merge failed for {representation}, cutoff {cutoff}")
    weather = selected if representation != "STAGE-HYBRID" else base_weather + selected
    return merged, weather


def feature_columns(
    family: str,
    weather: list[str],
) -> tuple[list[str], list[str]]:
    if family == "metadata_only":
        return ["lat", "lon"], CAT_COLUMNS
    if family == "weather_only":
        return weather, []
    if family == "full":
        return ["lat", "lon"] + weather, CAT_COLUMNS
    raise ValueError(f"Unknown feature family: {family}")


def preprocessing(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def estimator(model_id: str, config: dict[str, Any], seed: int):
    params = {key: value for key, value in config.items() if key != "config_id"}
    if model_id == "linear_shrinkage":
        kind = params.pop("kind")
        if kind == "ridge":
            return Ridge(random_state=seed, **params)
        return ElasticNet(random_state=seed, max_iter=10000, **params)
    if model_id == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=1, **params)
    if model_id == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
            **params,
        )
    if model_id == "catboost":
        return CatBoostRegressor(
            loss_function="RMSE",
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=seed,
            early_stopping=False,
            **params,
        )
    raise ValueError(f"Unknown model: {model_id}")


def hierarchy_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str]:
    train_meta = train[["crop", "region", "year", "lat", "lon"]].copy()
    eval_meta = evaluation[["crop", "region", "year", "lat", "lon"]].copy()
    train_meta["crop_state"] = train_meta["crop"] + "|" + train_meta["region"]
    eval_meta["crop_state"] = eval_meta["crop"] + "|" + eval_meta["region"]
    center = float(train_meta["year"].mean())
    train_year = train_meta["year"].to_numpy(dtype=float) - center
    eval_year = eval_meta["year"].to_numpy(dtype=float) - center
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_groups = encoder.fit_transform(
        train_meta[["crop", "region", "crop_state"]]
    )
    eval_groups = encoder.transform(eval_meta[["crop", "region", "crop_state"]])
    state_encoder = OneHotEncoder(handle_unknown="ignore")
    train_state = state_encoder.fit_transform(train_meta[["crop_state"]])
    eval_state = state_encoder.transform(eval_meta[["crop_state"]])
    train_slope = train_state.multiply(train_year[:, None])
    eval_slope = eval_state.multiply(eval_year[:, None])
    train_fixed = np.column_stack(
        [train_year, train_meta["lat"], train_meta["lon"]]
    )
    eval_fixed = np.column_stack(
        [eval_year, eval_meta["lat"], eval_meta["lon"]]
    )
    design_train = sparse.hstack(
        [sparse.csr_matrix(train_fixed), train_groups, train_slope],
        format="csr",
    )
    design_eval = sparse.hstack(
        [sparse.csr_matrix(eval_fixed), eval_groups, eval_slope],
        format="csr",
    )
    mode = "intercept_and_crop_state_slope"
    try:
        hierarchy = Ridge(alpha=10.0, solver="lsqr")
        hierarchy.fit(design_train, train["trend_residual_t_ha"])
        pred_train = np.asarray(hierarchy.predict(design_train), dtype=float)
        pred_eval = np.asarray(hierarchy.predict(design_eval), dtype=float)
        if not np.isfinite(pred_train).all() or not np.isfinite(pred_eval).all():
            raise FloatingPointError("Non-finite hierarchy prediction")
    except (FloatingPointError, ValueError):
        mode = "random_intercept_only_fallback"
        design_train = sparse.hstack(
            [sparse.csr_matrix(train_fixed), train_groups],
            format="csr",
        )
        design_eval = sparse.hstack(
            [sparse.csr_matrix(eval_fixed), eval_groups],
            format="csr",
        )
        hierarchy = Ridge(alpha=10.0, solver="lsqr")
        hierarchy.fit(design_train, train["trend_residual_t_ha"])
        pred_train = np.asarray(hierarchy.predict(design_train), dtype=float)
        pred_eval = np.asarray(hierarchy.predict(design_eval), dtype=float)
    return pred_train, pred_eval, mode


def fit_predict_job(root_text: str, job: dict[str, Any]) -> dict[str, Any]:
    protocol = load_protocol(root_text)
    frame, weather = representation_frame(
        root_text,
        str(job["representation"]),
        int(job["train_end"]),
    )
    train_raw = frame[frame["year"] <= int(job["train_end"])].copy()
    evaluation_raw = frame[
        frame["year"].between(
            int(job["evaluation_start"]),
            int(job["evaluation_end"]),
        )
    ].copy()
    minimum_history = int(protocol["data"]["minimum_training_history"])
    history = (
        train_raw.groupby(["crop", "region"])
        .size()
        .rename("n_history")
        .reset_index()
    )
    eligible = history[history["n_history"] >= minimum_history][
        ["crop", "region"]
    ]
    before_filter = len(evaluation_raw)
    evaluation_raw = evaluation_raw.merge(
        eligible,
        on=["crop", "region"],
        how="inner",
        validate="many_to_one",
    )
    if evaluation_raw.empty:
        raise AssertionError(f"No eligible evaluation rows for {job['job_id']}")
    train, evaluation, trend_audit = detrend_train_test(
        train_raw,
        evaluation_raw,
    )
    hierarchy_train = np.zeros(len(train), dtype=float)
    hierarchy_eval = np.zeros(len(evaluation), dtype=float)
    hierarchy_mode = "flat"
    if job["hierarchy"] == "HIER-RESIDUAL":
        hierarchy_train, hierarchy_eval, hierarchy_mode = hierarchy_predictions(
            train,
            evaluation,
        )
    target = train["trend_residual_t_ha"].to_numpy(dtype=float) - hierarchy_train
    numeric, categorical = feature_columns(str(job["feature_family"]), weather)
    columns = numeric + categorical
    config = json.loads(str(job["config_json"]))
    pipeline = Pipeline(
        [
            ("preprocess", preprocessing(numeric, categorical)),
            (
                "model",
                estimator(str(job["model_id"]), config, int(job["seed"])),
            ),
        ]
    )
    pipeline.fit(train[columns], target)
    prediction = (
        np.asarray(pipeline.predict(evaluation[columns]), dtype=float)
        + hierarchy_eval
    )
    truth = evaluation["trend_residual_t_ha"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "job_id": job["job_id"],
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "n_train": len(train),
        "n_evaluation": len(evaluation),
        "n_evaluation_excluded_history": before_filter - len(evaluation_raw),
        "hierarchy_mode": hierarchy_mode,
        "max_trend_fit_year": int(trend_audit["fit_year_max"].max()),
        "future_access": bool(trend_audit["future_access"].any()),
    }
    if job["phase"] == "outer":
        output = evaluation[KEYS + ["trend_residual_t_ha", "trend_residual_z"]].copy()
        output["arm_id"] = job["arm_id"]
        output["model_id"] = job["model_id"]
        output["feature_family"] = job["feature_family"]
        output["outer_fold"] = job["outer_fold"]
        output["seed"] = int(job["seed"])
        output["config_id"] = job["config_id"]
        output["y_true"] = truth
        output["y_pred"] = prediction
        output["baseline_pred"] = 0.0
        output["row_hash"] = output[KEYS].astype(str).agg("|".join, axis=1).map(
            lambda value: sha256(value.encode()).hexdigest()
        )
        result["predictions"] = output
    return result


def build_inner_jobs(protocol: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in protocol["arms"]:
        for model_id in protocol["models"]:
            configs = protocol["model_configs"][model_id][arm["grid"]]
            for family in protocol["feature_families"]:
                for outer in protocol["outer_folds"]:
                    for inner in outer["inner"]:
                        for config in configs:
                            fields = {
                                "phase": "inner",
                                **arm,
                                "model_id": model_id,
                                "feature_family": family,
                                "outer_fold": outer["outer_fold"],
                                "inner_fold": inner["inner_fold"],
                                "train_end": inner["train_end"],
                                "evaluation_start": inner["validation_start"],
                                "evaluation_end": inner["validation_end"],
                                "seed": protocol["seeds"][0],
                                "config_id": config["config_id"],
                                "config_json": json.dumps(config, sort_keys=True),
                            }
                            key = json.dumps(fields, sort_keys=True)
                            fields["job_id"] = "I-" + sha256(key.encode()).hexdigest()[:16]
                            rows.append(fields)
    return pd.DataFrame(rows)


def select_inner_configs(
    jobs: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    merged = jobs.merge(results[["job_id", "rmse"]], on="job_id", validate="one_to_one")
    group = [
        "arm_id",
        "model_id",
        "feature_family",
        "outer_fold",
        "config_id",
    ]
    summary = (
        merged.groupby(group, as_index=False)
        .agg(mean_inner_rmse=("rmse", "mean"), worst_inner_rmse=("rmse", "max"))
    )
    summary["complexity_rank"] = summary.groupby(
        ["arm_id", "model_id", "feature_family", "outer_fold"]
    )["config_id"].rank(method="dense")
    summary = summary.sort_values(
        [
            "arm_id",
            "model_id",
            "feature_family",
            "outer_fold",
            "mean_inner_rmse",
            "worst_inner_rmse",
            "complexity_rank",
        ],
        kind="stable",
    )
    return summary.groupby(
        ["arm_id", "model_id", "feature_family", "outer_fold"],
        as_index=False,
    ).first()


def build_outer_jobs(
    protocol: dict[str, Any],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    arms = {arm["arm_id"]: arm for arm in protocol["arms"]}
    folds = {fold["outer_fold"]: fold for fold in protocol["outer_folds"]}
    config_lookup = {
        config["config_id"]: config
        for model in protocol["model_configs"].values()
        for grid in model.values()
        for config in grid
    }
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        arm = arms[row.arm_id]
        fold = folds[row.outer_fold]
        seeds = (
            protocol["seeds"]
            if row.model_id in STOCHASTIC_MODELS
            else [protocol["seeds"][0]]
        )
        for seed in seeds:
            fields = {
                "phase": "outer",
                **arm,
                "model_id": row.model_id,
                "feature_family": row.feature_family,
                "outer_fold": row.outer_fold,
                "inner_fold": "",
                "train_end": fold["train_end"],
                "evaluation_start": fold["test_start"],
                "evaluation_end": fold["test_end"],
                "seed": seed,
                "config_id": row.config_id,
                "config_json": json.dumps(
                    config_lookup[row.config_id],
                    sort_keys=True,
                ),
            }
            key = json.dumps(fields, sort_keys=True)
            fields["job_id"] = "O-" + sha256(key.encode()).hexdigest()[:16]
            rows.append(fields)
    return pd.DataFrame(rows)
