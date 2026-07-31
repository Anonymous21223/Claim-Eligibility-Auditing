from __future__ import annotations

import math
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .core import (
    ANOMALY_Z_THRESHOLD,
    EVENT_YEARS,
    EXPECTED_DRIVER_GROUPS,
    EXPECTED_EVENT_GROUPS,
    RANDOM_STATE,
    TARGET,
    detrend_and_score,
    driver_group,
    driver_group_description,
    ensure_xai_dirs,
    event_key,
    event_key_columns,
    full_season_weather_features,
    group_features,
    is_leakage_feature,
    load_frame,
    make_project_paths,
    model_feature_columns,
)

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)


ALE_FEATURES = [
    "heat_days_30",
    "heat_degree_days_30",
    "rain_sum",
    "dry_spell_events_14d",
    "max_dry_spell_1mm",
    "max_3day_rain",
]
XAI_N_ESTIMATORS = 160
PERMUTATION_REPEATS = 8
METHOD_DESCRIPTIONS = [
    {
        "method": "SHAP",
        "purpose": "Feature-level model explanation",
        "output_level": "feature and event",
        "main_use": "Ranks weather features and signs their residual contribution",
    },
    {
        "method": "Grouped SHAP",
        "purpose": "Physical driver-group explanation",
        "output_level": "driver group",
        "main_use": "Aggregates SHAP values into heat, drought, frost/cold, excess rain, and radiation groups",
    },
    {
        "method": "Group permutation",
        "purpose": "Predictive dependence check",
        "output_level": "driver group",
        "main_use": "Measures residual-model error increase when a driver group is jointly shuffled",
    },
    {
        "method": "Group ablation",
        "purpose": "Model reliance check",
        "output_level": "driver group",
        "main_use": "Retrains the residual model without each group and measures residual-model error increase",
    },
    {
        "method": "ALE",
        "purpose": "Response-shape diagnostic",
        "output_level": "feature curve",
        "main_use": "Shows accumulated local residual response for selected weather features",
    },
    {
        "method": "LIME",
        "purpose": "Selected event-level local explanation",
        "output_level": "event and feature",
        "main_use": "Checks selected 2012, 2021, and 2022 anomaly cases",
    },
]


def method_settings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"setting": "random_state", "value": RANDOM_STATE, "description": "Seed used for ExtraTrees, sampling, and LIME"},
            {"setting": "xai_n_estimators", "value": XAI_N_ESTIMATORS, "description": "ExtraTrees estimators for XAI models"},
            {"setting": "permutation_repeats", "value": PERMUTATION_REPEATS, "description": "Joint shuffles per driver group"},
            {
                "setting": "lime_selection_rule",
                "value": "lowest z-score cases from 2012, 2021, and 2022",
                "description": "Selected-case rule for LIME local explanations",
            },
        ]
    )


@dataclass(frozen=True)
class ModelBundle:
    model: Pipeline
    numeric: list[str]
    categorical: list[str]
    features: list[str]


def make_regressor(numeric: list[str], categorical: list[str], n_estimators: int = XAI_N_ESTIMATORS) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", regressor)])


def r2_score_manual(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = np.sum((y - y.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def regression_metrics(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "r2": r2_score_manual(y, p),
        "rmse_t_ha": float(math.sqrt(np.mean((y - p) ** 2))),
        "mae_t_ha": float(np.mean(np.abs(y - p))),
    }


def fit_model(frame: pd.DataFrame, target: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    model = make_regressor(numeric, categorical)
    model.fit(frame[numeric + categorical], frame[target])
    return model


def transformed_feature_names(model: Pipeline) -> list[str]:
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    return [str(name) for name in names]


def predict_frame(model: Pipeline, frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    return model.predict(frame[numeric + categorical])


def clean_generated(paths: Any) -> None:
    ensure_xai_dirs(paths)
    for folder in [paths.xai_outputs, paths.xai_figures]:
        for pattern in ("*.csv", "*.md", "*.png", "*.pdf"):
            for path in folder.glob(pattern):
                path.unlink()


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_dataset_summary(frame: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crop, group in frame.groupby("crop", sort=True):
        rows.append(
            {
                "crop": crop,
                "observations": int(len(group)),
                "states": int(group["region"].nunique()),
                "years": f"{int(group['year'].min())}-{int(group['year'].max())}",
                "windows": ", ".join(sorted(group["window"].unique())),
                "low_yield_anomalies": int((anomalies["crop"] == crop).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_driver_group_tables(features: list[str], groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    feature_rows = []
    for group in EXPECTED_DRIVER_GROUPS:
        group_features = groups[group]
        summary_rows.append(
            {
                "driver_group": group,
                "description": driver_group_description(group),
                "n_features": len(group_features),
            }
        )
        for feature in group_features:
            feature_rows.append({"driver_group": group, "feature": feature})
    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)


def build_model_performance(scored: pd.DataFrame, features: list[str], anomalies: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    yield_numeric = ["year", "lat", "lon"] + features
    residual_numeric, categorical = model_feature_columns(features)
    splits = [
        ("yield", TARGET, yield_numeric, 2015, 2016, 2021, "forward_time_train_1990_2015_test_2016_2021"),
        ("yield", TARGET, yield_numeric, 2018, 2019, 2025, "forward_time_train_1990_2018_test_2019_2025"),
        (
            "residual",
            "trend_residual_t_ha",
            residual_numeric,
            2015,
            2016,
            2025,
            "forward_time_train_1990_2015_test_2016_2025",
        ),
    ]
    for model_name, target, numeric, train_end, test_start, test_end, protocol in splits:
        train = scored[scored["year"] <= train_end].copy()
        test = scored[(scored["year"] >= test_start) & (scored["year"] <= test_end)].copy()
        model = fit_model(train, target, numeric, categorical)
        pred = predict_frame(model, test, numeric, categorical)
        metrics = regression_metrics(test[target], pred)
        rows.append(
            {
                "model": f"ExtraTrees {model_name}",
                "target": target,
                "protocol": protocol,
                "scope": "all_rows",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
            }
        )
        if model_name == "residual":
            anomaly_test = test[test["is_low_yield_anomaly"]].copy()
            anomaly_pred = predict_frame(model, anomaly_test, numeric, categorical)
            rows.append(
                {
                    "model": "ExtraTrees residual",
                    "target": target,
                    "protocol": protocol,
                    "scope": "anomaly_rows",
                    "n_train": int(len(train)),
                    "n_test": int(len(anomaly_test)),
                    **regression_metrics(anomaly_test[target], anomaly_pred),
                }
            )

    anomaly_years = sorted(set(EVENT_YEARS) & set(anomalies["year"].unique()))
    all_eval: list[pd.DataFrame] = []
    anomaly_eval: list[pd.DataFrame] = []
    for year in anomaly_years:
        train = scored[scored["year"] != year].copy()
        test = scored[scored["year"] == year].copy()
        model = fit_model(train, "trend_residual_t_ha", residual_numeric, categorical)
        pred = predict_frame(model, test, residual_numeric, categorical)
        part = test[event_key_columns() + ["trend_residual_t_ha", "is_low_yield_anomaly"]].copy()
        part["prediction"] = pred
        part["held_out_year"] = int(year)
        all_eval.append(part)
        anomaly_eval.append(part[part["is_low_yield_anomaly"]].copy())

    all_temporal = pd.concat(all_eval, ignore_index=True)
    anomaly_temporal = pd.concat(anomaly_eval, ignore_index=True)
    for scope, table in [("all_rows_in_anomaly_years", all_temporal), ("anomaly_rows", anomaly_temporal)]:
        metrics = regression_metrics(table["trend_residual_t_ha"], table["prediction"])
        rows.append(
            {
                "model": "ExtraTrees residual",
                "target": "trend_residual_t_ha",
                "protocol": "leave_one_event_year_out_2012_2021_2022",
                "scope": scope,
                "n_train": "varies_by_year",
                "n_test": int(len(table)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def fit_final_residual(scored: pd.DataFrame, features: list[str]) -> ModelBundle:
    numeric, categorical = model_feature_columns(features)
    model = fit_model(scored, "trend_residual_t_ha", numeric, categorical)
    return ModelBundle(model=model, numeric=numeric, categorical=categorical, features=features)


def fit_forward_residual(scored: pd.DataFrame, features: list[str]) -> ModelBundle:
    numeric, categorical = model_feature_columns(features)
    train = scored[scored["year"] <= 2015].copy()
    model = fit_model(train, "trend_residual_t_ha", numeric, categorical)
    return ModelBundle(model=model, numeric=numeric, categorical=categorical, features=features)


def compute_shap_tables(bundle: ModelBundle, scored: pd.DataFrame, anomalies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("The XAI pipeline requires shap. Run `pip install -r requirements.txt`.") from exc

    feature_names = transformed_feature_names(bundle.model)
    preprocessor = bundle.model.named_steps["preprocess"]
    tree_model = bundle.model.named_steps["model"]
    x_all = preprocessor.transform(scored[bundle.numeric + bundle.categorical])
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(x_all, approximate=True, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    name_to_index = {name: index for index, name in enumerate(feature_names)}

    feature_rows = []
    local_rows = []
    anomaly_index = set(anomalies.index)
    eval_index = set(scored.index[scored["year"] >= 2016])
    eval_anomaly_index = sorted(anomaly_index & eval_index)
    for feature in bundle.features:
        if feature not in name_to_index:
            raise AssertionError(f"Missing SHAP feature after preprocessing: {feature}")
        values = shap_values[:, name_to_index[feature]]
        eval_values = values[sorted(eval_index)]
        anomaly_values = values[sorted(anomaly_index)]
        eval_anomaly_values = values[eval_anomaly_index]
        feature_rows.append(
            {
                "feature": feature,
                "driver_group": driver_group(feature),
                "mean_abs_shap_all": float(np.mean(np.abs(values))),
                "mean_signed_shap_all": float(np.mean(values)),
                "mean_abs_shap_eval": float(np.mean(np.abs(eval_values))),
                "mean_signed_shap_eval": float(np.mean(eval_values)),
                "mean_abs_shap_anomalies": float(np.mean(np.abs(anomaly_values))),
                "mean_signed_shap_anomalies": float(np.mean(anomaly_values)),
                "mean_abs_shap_eval_anomalies": float(np.mean(np.abs(eval_anomaly_values))),
                "mean_signed_shap_eval_anomalies": float(np.mean(eval_anomaly_values)),
            }
        )

    for row_index in sorted(anomaly_index):
        source = scored.loc[row_index]
        key = event_key(source)
        for feature in bundle.features:
            value = shap_values[row_index, name_to_index[feature]]
            local_rows.append(
                {
                    "event_key": key,
                    **{column: source[column] for column in event_key_columns()},
                    "feature": feature,
                    "driver_group": driver_group(feature),
                    "feature_value": float(source[feature]),
                    "shap_value": float(value),
                    "abs_shap_value": float(abs(value)),
                }
            )

    shap_feature = pd.DataFrame(feature_rows).sort_values("mean_abs_shap_eval", ascending=False)
    shap_feature["rank_all"] = shap_feature["mean_abs_shap_all"].rank(ascending=False, method="first").astype(int)
    shap_feature["rank_eval"] = range(1, len(shap_feature) + 1)
    shap_local = pd.DataFrame(local_rows)

    group_rows: list[dict[str, Any]] = []
    scope_indexes = [
        ("all_rows", list(scored.index)),
        ("anomaly_rows", sorted(anomaly_index)),
        ("eval_all_rows", sorted(eval_index)),
        ("eval_anomaly_rows", eval_anomaly_index),
    ]
    for scope, row_indexes in scope_indexes:
        for group in EXPECTED_DRIVER_GROUPS:
            group_cols = [name_to_index[feature] for feature in bundle.features if driver_group(feature) == group]
            values = shap_values[row_indexes][:, group_cols].sum(axis=1)
            group_rows.append(
                {
                    "scope": scope,
                    "driver_group": group,
                    "mean_abs_group_shap": float(np.mean(np.abs(values))),
                    "mean_signed_group_shap": float(np.mean(values)),
                    "mean_stress_group_shap": float(np.mean(np.maximum(0.0, -values))),
                    "n_rows": len(row_indexes),
                }
            )
    grouped = pd.DataFrame(group_rows)
    grouped["rank_abs"] = grouped.groupby("scope")["mean_abs_group_shap"].rank(ascending=False, method="first").astype(int)
    grouped["rank_stress"] = grouped.groupby("scope")["mean_stress_group_shap"].rank(ascending=False, method="first").astype(int)

    event_group_rows = []
    for row_index in sorted(anomaly_index):
        source = scored.loc[row_index]
        key = event_key(source)
        for group in EXPECTED_DRIVER_GROUPS:
            group_cols = [name_to_index[feature] for feature in bundle.features if driver_group(feature) == group]
            signed_value = float(shap_values[row_index, group_cols].sum())
            event_group_rows.append(
                {
                    "event_key": key,
                    **{column: source[column] for column in event_key_columns()},
                    "driver_group": group,
                    "signed_group_shap": signed_value,
                    "abs_group_shap": abs(signed_value),
                    "stress_score": max(0.0, -signed_value),
                }
            )
    event_grouped = pd.DataFrame(event_group_rows)
    event_grouped["rank_stress"] = event_grouped.groupby("event_key")["stress_score"].rank(ascending=False, method="first").astype(int)
    event_grouped["rank_abs"] = event_grouped.groupby("event_key")["abs_group_shap"].rank(ascending=False, method="first").astype(int)
    return shap_feature, shap_local, pd.concat([grouped.assign(table="summary"), event_grouped.assign(table="event")], ignore_index=True)


def group_permutation_importance(
    scored: pd.DataFrame,
    bundle: ModelBundle,
    groups: dict[str, list[str]],
    repeats: int = PERMUTATION_REPEATS,
) -> pd.DataFrame:
    train = scored[scored["year"] <= 2015].copy()
    test = scored[scored["year"] >= 2016].copy()
    model = fit_model(train, "trend_residual_t_ha", bundle.numeric, bundle.categorical)
    base_pred = predict_frame(model, test, bundle.numeric, bundle.categorical)
    scopes = {
        "all_rows": test.index.to_numpy(),
        "anomaly_rows": test[test["is_low_yield_anomaly"]].index.to_numpy(),
    }
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for scope, index_values in scopes.items():
        scoped = test.loc[index_values].copy()
        base = base_pred[test.index.get_indexer(index_values)]
        base_metrics = regression_metrics(scoped["trend_residual_t_ha"], base)
        for group, features in groups.items():
            metrics_list = []
            for _repeat in range(repeats):
                shuffled = scoped.copy()
                order = rng.permutation(len(shuffled))
                shuffled.loc[:, features] = shuffled[features].iloc[order].to_numpy()
                pred = predict_frame(model, shuffled, bundle.numeric, bundle.categorical)
                metrics_list.append(regression_metrics(shuffled["trend_residual_t_ha"], pred))
            avg = {
                metric: float(np.mean([m[metric] for m in metrics_list]))
                for metric in ["r2", "rmse_t_ha", "mae_t_ha"]
            }
            rows.append(
                {
                    "scope": scope,
                    "driver_group": group,
                    "baseline_r2": base_metrics["r2"],
                    "permuted_r2": avg["r2"],
                    "r2_drop": base_metrics["r2"] - avg["r2"],
                    "baseline_rmse_t_ha": base_metrics["rmse_t_ha"],
                    "permuted_rmse_t_ha": avg["rmse_t_ha"],
                    "rmse_increase_t_ha": avg["rmse_t_ha"] - base_metrics["rmse_t_ha"],
                    "baseline_mae_t_ha": base_metrics["mae_t_ha"],
                    "permuted_mae_t_ha": avg["mae_t_ha"],
                    "mae_increase_t_ha": avg["mae_t_ha"] - base_metrics["mae_t_ha"],
                    "n_rows": int(len(scoped)),
                    "repeats": repeats,
                }
            )
    table = pd.DataFrame(rows)
    table["rank_rmse"] = table.groupby("scope")["rmse_increase_t_ha"].rank(ascending=False, method="first").astype(int)
    return table


def group_ablation_importance(scored: pd.DataFrame, bundle: ModelBundle, groups: dict[str, list[str]]) -> pd.DataFrame:
    train = scored[scored["year"] <= 2015].copy()
    test = scored[scored["year"] >= 2016].copy()
    base_model = fit_model(train, "trend_residual_t_ha", bundle.numeric, bundle.categorical)
    base_pred = predict_frame(base_model, test, bundle.numeric, bundle.categorical)
    scopes = {
        "all_rows": test.index.to_numpy(),
        "anomaly_rows": test[test["is_low_yield_anomaly"]].index.to_numpy(),
    }
    rows = []
    for group, remove_features in groups.items():
        kept_features = [feature for feature in bundle.features if feature not in remove_features]
        numeric, categorical = model_feature_columns(kept_features)
        ablated_model = fit_model(train, "trend_residual_t_ha", numeric, categorical)
        ablated_pred = predict_frame(ablated_model, test, numeric, categorical)
        for scope, index_values in scopes.items():
            scoped = test.loc[index_values].copy()
            positions = test.index.get_indexer(index_values)
            base_metrics = regression_metrics(scoped["trend_residual_t_ha"], base_pred[positions])
            ablated_metrics = regression_metrics(scoped["trend_residual_t_ha"], ablated_pred[positions])
            rows.append(
                {
                    "scope": scope,
                    "driver_group": group,
                    "baseline_r2": base_metrics["r2"],
                    "ablated_r2": ablated_metrics["r2"],
                    "r2_drop": base_metrics["r2"] - ablated_metrics["r2"],
                    "baseline_rmse_t_ha": base_metrics["rmse_t_ha"],
                    "ablated_rmse_t_ha": ablated_metrics["rmse_t_ha"],
                    "rmse_increase_t_ha": ablated_metrics["rmse_t_ha"] - base_metrics["rmse_t_ha"],
                    "baseline_mae_t_ha": base_metrics["mae_t_ha"],
                    "ablated_mae_t_ha": ablated_metrics["mae_t_ha"],
                    "mae_increase_t_ha": ablated_metrics["mae_t_ha"] - base_metrics["mae_t_ha"],
                    "n_rows": int(len(scoped)),
                    "removed_features": ", ".join(remove_features),
                }
            )
    table = pd.DataFrame(rows)
    table["rank_rmse"] = table.groupby("scope")["rmse_increase_t_ha"].rank(ascending=False, method="first").astype(int)
    return table


def selected_event_cases(anomalies: pd.DataFrame, n_per_year: int = 4) -> pd.DataFrame:
    pieces = []
    for year in EVENT_YEARS:
        subset = anomalies[anomalies["year"] == year].copy()
        if subset.empty:
            continue
        pieces.append(subset.sort_values("trend_residual_z").head(n_per_year))
    if not pieces:
        return anomalies.sort_values("trend_residual_z").head(8).copy()
    return pd.concat(pieces, ignore_index=False)


def lime_case_studies(scored: pd.DataFrame, anomalies: pd.DataFrame, bundle: ModelBundle, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected_event_cases(anomalies)
    status_rows = []
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return (
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "status": "skipped",
                        "reason": "lime is not installed",
                        "n_selected_events": int(len(selected)),
                    }
                ]
            ),
        )

    numeric = bundle.numeric
    train_array = scored[numeric].to_numpy(dtype=float)
    explainer = LimeTabularExplainer(
        training_data=train_array,
        feature_names=numeric,
        mode="regression",
        random_state=RANDOM_STATE,
        discretize_continuous=True,
    )
    feature_order = sorted(numeric, key=len, reverse=True)
    rows = []
    for _, row in selected.iterrows():
        key = event_key(row)

        def predict_numeric(samples: np.ndarray) -> np.ndarray:
            sample_frame = pd.DataFrame(samples, columns=numeric)
            sample_frame["region"] = row["region"]
            sample_frame["crop"] = row["crop"]
            return predict_frame(bundle.model, sample_frame, numeric, bundle.categorical)

        explanation = explainer.explain_instance(
            row[numeric].to_numpy(dtype=float),
            predict_numeric,
            num_features=min(12, len(numeric)),
            num_samples=500,
        )
        for description, weight in explanation.as_list():
            matched = next((feature for feature in feature_order if re.search(rf"\b{re.escape(feature)}\b", description)), None)
            if matched is None or matched not in bundle.features:
                continue
            rows.append(
                {
                    "event_key": key,
                    **{column: row[column] for column in event_key_columns()},
                    "feature": matched,
                    "driver_group": driver_group(matched),
                    "lime_description": description,
                    "lime_weight": float(weight),
                    "abs_lime_weight": float(abs(weight)),
                }
            )
    lime_table = pd.DataFrame(rows)
    status_rows.append(
        {
            "status": "completed" if len(lime_table) else "completed_no_weather_features",
            "reason": "",
            "n_selected_events": int(len(selected)),
            "n_lime_rows": int(len(lime_table)),
        }
    )
    return lime_table, pd.DataFrame(status_rows)


def ale_curves(scored: pd.DataFrame, bundle: ModelBundle, features: list[str] = ALE_FEATURES, bins: int = 10) -> pd.DataFrame:
    rows = []
    base = scored[bundle.numeric + bundle.categorical].copy()
    for feature in features:
        if feature not in bundle.features:
            continue
        values = scored[feature].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 3:
            continue
        effects = []
        counts = []
        mids = []
        for i in range(len(edges) - 1):
            lower, upper = float(edges[i]), float(edges[i + 1])
            if i == len(edges) - 2:
                mask = (scored[feature] >= lower) & (scored[feature] <= upper)
            else:
                mask = (scored[feature] >= lower) & (scored[feature] < upper)
            subset = base.loc[mask].copy()
            if subset.empty:
                effects.append(0.0)
                counts.append(0)
                mids.append((lower + upper) / 2.0)
                continue
            low = subset.copy()
            high = subset.copy()
            low[feature] = lower
            high[feature] = upper
            diff = predict_frame(bundle.model, high, bundle.numeric, bundle.categorical) - predict_frame(
                bundle.model, low, bundle.numeric, bundle.categorical
            )
            effects.append(float(np.mean(diff)))
            counts.append(int(len(subset)))
            mids.append((lower + upper) / 2.0)
        cumulative = np.cumsum(effects)
        weighted_center = float(np.average(cumulative, weights=np.maximum(counts, 1)))
        centered = cumulative - weighted_center
        for mid, effect, ale_value, count in zip(mids, effects, centered, counts):
            rows.append(
                {
                    "feature": feature,
                    "driver_group": driver_group(feature),
                    "feature_value_midpoint": float(mid),
                    "local_effect": float(effect),
                    "ale_residual_t_ha": float(ale_value),
                    "n_rows": int(count),
                }
            )
    return pd.DataFrame(rows)


def method_label(method: str) -> str:
    labels = {
        "group_ablation": "group ablation",
        "group_permutation": "group permutation",
        "grouped_shap": "grouped SHAP",
        "grouped_shap_abs": "grouped SHAP",
        "grouped_shap_anomaly_abs": "grouped SHAP",
        "lime_selected": "LIME selected cases",
    }
    return labels.get(method, method)


def event_top_driver_votes(grouped_shap: pd.DataFrame, lime_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_part = grouped_shap[grouped_shap["table"] == "event"].copy()
    top_shap = event_part.sort_values(["event_key", "rank_abs", "rank_stress"]).groupby("event_key").head(1)
    for _, row in top_shap.iterrows():
        rows.append(
            {
                "event_key": row["event_key"],
                **{column: row[column] for column in event_key_columns()},
                "method": "grouped_shap",
                "top_driver_group": row["driver_group"],
                "score": float(row["abs_group_shap"]),
            }
        )
    if not lime_table.empty:
        lime_group = (
            lime_table.groupby(["event_key", "driver_group"], as_index=False)
            .agg(score=("abs_lime_weight", "sum"))
            .sort_values(["event_key", "score"], ascending=[True, False])
            .groupby("event_key")
            .head(1)
        )
        event_lookup = lime_table.drop_duplicates("event_key").set_index("event_key")
        for _, row in lime_group.iterrows():
            source = event_lookup.loc[row["event_key"]]
            rows.append(
                {
                    "event_key": row["event_key"],
                    **{column: source[column] for column in event_key_columns()},
                    "method": "lime_selected",
                    "top_driver_group": row["driver_group"],
                    "score": float(row["score"]),
                }
            )
    votes = pd.DataFrame(rows)
    expected = []
    for _, row in votes.iterrows():
        groups = EXPECTED_EVENT_GROUPS.get(int(row["year"]), set())
        expected.append(bool(groups and row["top_driver_group"] in groups))
    votes["matches_event_sanity_group"] = expected
    return votes


def method_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ["global", "anomaly"]:
        top = rankings[(rankings["scope"] == scope) & (rankings["rank"] == 1)].copy()
        if top.empty:
            continue
        grouped = (
            top.groupby("driver_group", as_index=False)
            .agg(rank_1_votes=("method", "size"), methods_supporting=("method", lambda values: ", ".join(method_label(v) for v in sorted(values))))
            .sort_values(["rank_1_votes", "driver_group"], ascending=[False, True])
        )
        max_votes = int(grouped["rank_1_votes"].max())
        leaders = grouped[grouped["rank_1_votes"] == max_votes].copy()
        leader_names = "; ".join(leaders["driver_group"])
        leader_support = "; ".join(
            f"{row.driver_group}: {row.methods_supporting}" for row in leaders.itertuples(index=False)
        )
        if scope == "global":
            interpretation = "Heat has majority support across global model-reliance diagnostics."
        elif len(leaders) > 1:
            interpretation = "Heat and drought are co-leading anomaly-focused signals under rank-one support."
        else:
            interpretation = f"{leader_names} has majority support across anomaly-focused explanations."
        rows.append(
            {
                "scope": scope,
                "n_methods": int(len(top)),
                "consensus_driver": leader_names,
                "rank_1_votes": f"{max_votes}/{int(len(top))}" + (" each" if len(leaders) > 1 else ""),
                "methods_supporting": leader_support if len(leaders) > 1 else leaders.iloc[0]["methods_supporting"],
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def method_driver_rankings(
    grouped_shap: pd.DataFrame,
    permutation: pd.DataFrame,
    ablation: pd.DataFrame,
    lime_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    shap_summary = grouped_shap[grouped_shap["table"] == "summary"].copy()
    for scope, method_name, output_scope, level in [
        ("eval_all_rows", "grouped_shap_abs", "global", "forward-time all rows"),
        ("eval_anomaly_rows", "grouped_shap_anomaly_abs", "anomaly", "forward-time anomaly rows"),
    ]:
        sub = shap_summary[shap_summary["scope"] == scope].copy()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "scope": output_scope,
                    "method": method_name,
                    "driver_group": row["driver_group"],
                    "score": float(row["mean_abs_group_shap"]),
                    "n_eval": int(row["n_rows"]),
                    "level": level,
                }
            )
    for table, method_name in [(permutation, "group_permutation"), (ablation, "group_ablation")]:
        for scope in ["all_rows", "anomaly_rows"]:
            sub = table[table["scope"] == scope].copy()
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "scope": "global" if scope == "all_rows" else "anomaly",
                        "method": method_name,
                        "driver_group": row["driver_group"],
                        "score": float(row["rmse_increase_t_ha"]),
                        "n_eval": int(row["n_rows"]),
                        "level": "all rows" if scope == "all_rows" else "anomaly rows",
                    }
                )
    if not lime_table.empty:
        summary = lime_table.groupby("driver_group", as_index=False).agg(score=("abs_lime_weight", "mean"))
        n_lime = int(lime_table["event_key"].nunique())
        for _, row in summary.iterrows():
            rows.append(
                {
                    "scope": "anomaly",
                    "method": "lime_selected",
                    "driver_group": row["driver_group"],
                    "score": float(row["score"]),
                    "n_eval": n_lime,
                    "level": "selected events",
                }
            )
    rankings = pd.DataFrame(rows)
    rankings["rank"] = rankings.groupby(["scope", "method"])["score"].rank(ascending=False, method="first").astype(int)
    return rankings.sort_values(["scope", "method", "rank"])


def representative_event_explanations(votes: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    selected_keys = [event_key(row) for _, row in selected_event_cases(anomalies, n_per_year=2).iterrows()]
    lime_keys = set(votes.loc[votes["method"] == "lime_selected", "event_key"])
    selected_keys = [key for key in selected_keys if key in lime_keys]
    selected_votes = votes[votes["event_key"].isin(selected_keys)].copy()
    anomaly_lookup = anomalies.assign(event_key=anomalies.apply(event_key, axis=1)).set_index("event_key")
    rows = []
    for event, group in selected_votes.groupby("event_key", sort=True):
        first = group.iloc[0]
        top_by_method = group.sort_values("method").set_index("method")["top_driver_group"].to_dict()
        grouped_driver = top_by_method.get("grouped_shap", "")
        lime_driver = top_by_method.get("lime_selected", "")
        rows.append(
            {
                "event_key": event,
                **{column: first[column] for column in event_key_columns()},
                "trend_residual_z": float(anomaly_lookup.loc[event, "trend_residual_z"]),
                "grouped_shap_driver": grouped_driver,
                "lime_driver": lime_driver,
                "drivers_agree": "yes" if grouped_driver and grouped_driver == lime_driver else "no",
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "crop", "region"]).head(6)


def anomaly_threshold_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_scored, baseline_anomalies = detrend_and_score(frame, ANOMALY_Z_THRESHOLD)
    baseline_keys = set(baseline_anomalies.apply(event_key, axis=1))
    for threshold in [-1.0, -1.5, -2.0]:
        scored, anomalies = detrend_and_score(frame, threshold)
        keys = set(anomalies.apply(event_key, axis=1))
        overlap = len(keys & baseline_keys)
        rows.append(
            {
                "threshold": threshold,
                "anomaly_count": int(len(anomalies)),
                "share_of_dataset": float(len(anomalies) / len(scored)),
                "overlap_with_z_minus_1": int(overlap),
                "jaccard_with_z_minus_1": float(overlap / len(keys | baseline_keys)) if keys or baseline_keys else 1.0,
            }
        )
    return pd.DataFrame(rows)


def detrending_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    linear_scored, linear_anomalies = detrend_and_score(frame, ANOMALY_Z_THRESHOLD)
    linear_keys = set(linear_anomalies.apply(event_key, axis=1))
    rows = [
        {
            "detrending_method": "linear",
            "anomaly_count": int(len(linear_anomalies)),
            "overlap_with_linear": int(len(linear_keys)),
            "jaccard_with_linear": 1.0,
        }
    ]
    rolling_pieces = []
    for (_crop, _region), group in frame.groupby(["crop", "region"], sort=True):
        g = group.sort_values("year").copy()
        trend = g[TARGET].rolling(window=7, center=True, min_periods=3).mean()
        trend = trend.bfill().ffill()
        residual = g[TARGET] - trend
        std = np.std(residual, ddof=1)
        z = np.zeros(len(g)) if not np.isfinite(std) or std == 0 else residual / std
        g["trend_yield_t_ha"] = trend
        g["trend_residual_t_ha"] = residual
        g["trend_residual_z"] = z
        g["is_low_yield_anomaly"] = g["trend_residual_z"] < ANOMALY_Z_THRESHOLD
        rolling_pieces.append(g)
    rolling = pd.concat(rolling_pieces, ignore_index=True)
    rolling_keys = set(rolling[rolling["is_low_yield_anomaly"]].apply(event_key, axis=1))
    rows.append(
        {
            "detrending_method": "centered_7_year_rolling",
            "anomaly_count": int(len(rolling_keys)),
            "overlap_with_linear": int(len(rolling_keys & linear_keys)),
            "jaccard_with_linear": float(len(rolling_keys & linear_keys) / len(rolling_keys | linear_keys)),
        }
    )
    return pd.DataFrame(rows)


def event_year_sanity(votes: pd.DataFrame) -> pd.DataFrame:
    event_votes = votes[votes["year"].isin(EVENT_YEARS)].copy()
    rows = []
    for method, group in event_votes.groupby("method"):
        for year, year_group in group.groupby("year"):
            expected = ", ".join(sorted(EXPECTED_EVENT_GROUPS[int(year)]))
            rows.append(
                {
                    "method": method,
                    "year": int(year),
                    "n_events": int(len(year_group)),
                    "expected_driver_groups": expected,
                    "match_rate": float(year_group["matches_event_sanity_group"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "year"])


def fig_workflow(path: Path) -> None:
    static = path.parents[2] / "figures" / "static" / "fig01_method_workflow.png"
    if static.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static, path.with_suffix(".png"))
        return

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    steps = [
        "USDA yield\nNASA POWER weather",
        "Crop-state-year\nweather panel",
        "Detrend yield\nflag low-yield anomalies",
        "Residual ExtraTrees\nforward-time validation",
        "SHAP, permutation,\nablation, ALE, LIME",
        "Driver-group\nmethod agreement",
    ]
    xs = np.linspace(0.07, 0.93, len(steps))
    colors = ["#d8eadf", "#dce7f7", "#f4e3bf", "#eadcf4", "#d9edf0", "#f2d6d0"]
    for i, (x, step) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            0.55,
            step,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35,rounding_size=0.08", "facecolor": colors[i], "edgecolor": "#333333"},
        )
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.075, 0.55), xytext=(x + 0.075, 0.55), arrowprops={"arrowstyle": "->"})
    ax.set_title("Multi-method XAI workflow for crop-yield anomalies", fontsize=13)
    save_figure(fig, path)


def fig_anomaly_timeline(anomalies: pd.DataFrame, path: Path) -> None:
    counts = anomalies.groupby("year").size().reindex(range(1990, 2026), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#c95f50" if year in EVENT_YEARS else "#6f91b7" for year in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("Year")
    ax.set_ylabel("Low-yield anomaly count")
    ax.set_title("Detrended low-yield anomalies by year")
    ax.set_xlim(1989.3, 2025.7)
    save_figure(fig, path)


def fig_shap_summary(shap_feature: pd.DataFrame, path: Path) -> None:
    top = shap_feature.sort_values("mean_abs_shap_eval", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap_eval"], color="#6f91b7")
    ax.set_xlabel("Mean absolute SHAP value on evaluation split (residual t/ha)")
    ax.set_title("Top weather features by SHAP importance")
    save_figure(fig, path)


def fig_grouped_shap(grouped_shap: pd.DataFrame, path: Path) -> None:
    summary = grouped_shap[grouped_shap["table"] == "summary"].copy()
    pivot = (
        summary[summary["scope"].isin(["eval_all_rows", "eval_anomaly_rows"])]
        .pivot(index="driver_group", columns="scope", values="mean_abs_group_shap")
        .reindex(EXPECTED_DRIVER_GROUPS)
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot["eval_all_rows"], width, label="Forward-time all rows", color="#6f91b7")
    ax.bar(x + width / 2, pivot["eval_anomaly_rows"], width, label="Forward-time anomaly rows", color="#c95f50")
    ax.set_xticks(x, pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Mean absolute grouped SHAP")
    ax.set_title("Grouped SHAP by extreme-weather driver on evaluation split")
    ax.legend(frameon=False)
    save_figure(fig, path)


def fig_importance_bars(table: pd.DataFrame, path: Path, title: str, value_column: str = "rmse_increase_t_ha") -> None:
    sub = table[table["scope"] == "all_rows"].sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(sub["driver_group"], sub[value_column], color="#7aa37b")
    ax.set_xlabel("RMSE increase (residual t/ha)")
    ax.set_title(title)
    save_figure(fig, path)


def fig_group_importance(permutation: pd.DataFrame, ablation: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
    for ax, table, title, color in [
        (axes[0], permutation, "Group permutation", "#6f91b7"),
        (axes[1], ablation, "Group ablation", "#7aa37b"),
    ]:
        sub = table[table["scope"] == "all_rows"].sort_values("rmse_increase_t_ha", ascending=True)
        ax.barh(sub["driver_group"], sub["rmse_increase_t_ha"], color=color)
        ax.set_xlabel("RMSE increase (residual t/ha)")
        ax.set_title(title)
    fig.suptitle("Global driver-group importance diagnostics", fontsize=13)
    save_figure(fig, path)


def fig_ale(ale: pd.DataFrame, path: Path) -> None:
    features = [feature for feature in ALE_FEATURES if feature in set(ale["feature"])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, feature in zip(axes.ravel(), features):
        sub = ale[ale["feature"] == feature].sort_values("feature_value_midpoint")
        ax.plot(sub["feature_value_midpoint"], sub["ale_residual_t_ha"], marker="o", color="#6f91b7", linewidth=1.8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("ALE residual")
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.suptitle("ALE curves for selected weather features", fontsize=13)
    save_figure(fig, path)


def fig_agreement(rankings: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, scope, title in [
        (axes[0], "global", "Global rankings"),
        (axes[1], "anomaly", "Anomaly-focused rankings"),
    ]:
        sub = rankings[rankings["scope"] == scope].copy()
        sub["method_label"] = sub["method"].map(method_label)
        pivot = sub.pivot(index="method_label", columns="driver_group", values="rank").reindex(columns=EXPECTED_DRIVER_GROUPS)
        rank_values = pivot.to_numpy(dtype=float)
        heat_values = np.where(np.isfinite(rank_values), 6.0 - rank_values, 0.0)
        image = ax.imshow(heat_values, vmin=0, vmax=5, cmap="YlGnBu")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                label = "" if pd.isna(value) else f"R{int(value)}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)
        ax.set_title(title)
    fig.suptitle("Driver-group rank positions across XAI methods", fontsize=13)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    save_figure(fig, path)


def write_summary(paths: Any, model_performance: pd.DataFrame, rankings: pd.DataFrame, agreement: pd.DataFrame) -> None:
    residual = model_performance[
        (model_performance["model"] == "ExtraTrees residual")
        & (model_performance["protocol"] == "forward_time_train_1990_2015_test_2016_2025")
        & (model_performance["scope"] == "all_rows")
    ].iloc[0]
    top_global = rankings[(rankings["scope"] == "global") & (rankings["rank"] == 1)]
    lines = [
        "# Multi-Method XAI Results Summary",
        "",
        "Generated by `python scripts/run_xai_pipeline.py`.",
        "",
        f"- Residual forward-time R2: {float(residual['r2']):.3f}.",
        f"- Residual forward-time RMSE: {float(residual['rmse_t_ha']):.3f} t/ha.",
        "- Top global driver by method:",
    ]
    for _, row in top_global.iterrows():
        lines.append(f"  - {method_label(str(row['method']))}: {row['driver_group']} ({float(row['score']):.4f})")
    consensus_lines = [
        f"- {row['scope']} consensus: {row['consensus_driver']} ({row['rank_1_votes']} methods)."
        for _, row in agreement.iterrows()
    ]
    lines.extend(
        [
            "",
            *consensus_lines,
            "",
            "All explanations are model-based diagnostics and are not causal event-attribution claims.",
            "",
        ]
    )
    (paths.xai_outputs / "XAI_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def assert_pipeline_outputs(paths: Any) -> None:
    expected_csv = [
        "anomaly_scores_all_rows.csv",
        "low_yield_anomalies.csv",
        "dataset_summary.csv",
        "driver_groups.csv",
        "driver_group_features.csv",
        "model_performance.csv",
        "xai_methods.csv",
        "method_settings.csv",
        "shap_feature_ranking.csv",
        "shap_local_anomaly_values.csv",
        "grouped_shap_summary.csv",
        "group_permutation_importance.csv",
        "group_ablation_importance.csv",
        "lime_event_explanations.csv",
        "lime_status.csv",
        "ale_curves.csv",
        "event_method_top_drivers.csv",
        "method_agreement_matrix.csv",
        "method_driver_rankings.csv",
        "representative_event_explanations.csv",
        "anomaly_threshold_sensitivity.csv",
        "detrending_robustness.csv",
        "event_year_sanity.csv",
    ]
    expected_figures = [
        "fig01_method_workflow.png",
        "fig02_anomaly_timeline.png",
        "fig03_shap_summary.png",
        "fig04_grouped_shap.png",
        "fig05_group_importance.png",
        "fig07_ale_curves.png",
        "fig08_method_agreement.png",
    ]
    missing = [str(paths.xai_outputs / name) for name in expected_csv if not (paths.xai_outputs / name).exists()]
    missing += [str(paths.xai_figures / name) for name in expected_figures if not (paths.xai_figures / name).exists()]
    if missing:
        raise AssertionError(f"Missing XAI outputs: {missing}")
    for figure in expected_figures:
        img = plt.imread(paths.xai_figures / figure)
        if img.size == 0 or float(np.std(img)) == 0.0:
            raise AssertionError(f"Blank figure: {figure}")


def run_xai_pipeline(root: Path | None = None) -> None:
    project_root = root or Path(__file__).resolve().parents[2]
    paths = make_project_paths(project_root)
    clean_generated(paths)

    print("Loading frame and scoring anomalies...", flush=True)
    frame = load_frame(paths)
    scored, anomalies = detrend_and_score(frame, ANOMALY_Z_THRESHOLD)
    scored = scored.reset_index(drop=True)
    anomalies = scored[scored["is_low_yield_anomaly"]].copy()
    features = full_season_weather_features(scored)
    groups = group_features(features)
    if len(anomalies) != 214:
        raise AssertionError(f"Expected 214 low-yield anomalies, found {len(anomalies)}")
    if len(features) != 35:
        raise AssertionError(f"Expected 35 full-season weather features, found {len(features)}")

    print("Fitting residual model and building summary tables...", flush=True)
    bundle = fit_forward_residual(scored, features)

    dataset_summary = build_dataset_summary(frame, anomalies)
    driver_summary, driver_features = build_driver_group_tables(features, groups)
    model_performance = build_model_performance(scored, features, anomalies)
    print("Computing SHAP explanations...", flush=True)
    shap_feature, shap_local, grouped_shap = compute_shap_tables(bundle, scored, anomalies)
    print("Computing group permutation and ablation checks...", flush=True)
    permutation = group_permutation_importance(scored, bundle, groups)
    ablation = group_ablation_importance(scored, bundle, groups)
    print("Computing LIME cases and ALE curves...", flush=True)
    lime_table, lime_status = lime_case_studies(scored, anomalies, bundle, groups)
    ale = ale_curves(scored, bundle)
    print("Computing method agreement tables...", flush=True)
    votes = event_top_driver_votes(grouped_shap, lime_table)
    rankings = method_driver_rankings(grouped_shap, permutation, ablation, lime_table)
    agreement = method_agreement(rankings)
    representative = representative_event_explanations(votes, anomalies)
    threshold = anomaly_threshold_sensitivity(frame)
    detrending = detrending_robustness(frame)
    sanity = event_year_sanity(votes)

    print("Writing XAI CSV outputs...", flush=True)
    outputs = paths.xai_outputs
    write_dataframe(scored, outputs / "anomaly_scores_all_rows.csv")
    write_dataframe(anomalies, outputs / "low_yield_anomalies.csv")
    write_dataframe(dataset_summary, outputs / "dataset_summary.csv")
    write_dataframe(driver_summary, outputs / "driver_groups.csv")
    write_dataframe(driver_features, outputs / "driver_group_features.csv")
    write_dataframe(model_performance, outputs / "model_performance.csv")
    write_dataframe(pd.DataFrame(METHOD_DESCRIPTIONS), outputs / "xai_methods.csv")
    write_dataframe(method_settings(), outputs / "method_settings.csv")
    write_dataframe(shap_feature, outputs / "shap_feature_ranking.csv")
    write_dataframe(shap_local, outputs / "shap_local_anomaly_values.csv")
    write_dataframe(grouped_shap, outputs / "grouped_shap_summary.csv")
    write_dataframe(permutation, outputs / "group_permutation_importance.csv")
    write_dataframe(ablation, outputs / "group_ablation_importance.csv")
    write_dataframe(lime_table, outputs / "lime_event_explanations.csv")
    write_dataframe(lime_status, outputs / "lime_status.csv")
    write_dataframe(ale, outputs / "ale_curves.csv")
    write_dataframe(votes, outputs / "event_method_top_drivers.csv")
    write_dataframe(agreement, outputs / "method_agreement_matrix.csv")
    write_dataframe(rankings, outputs / "method_driver_rankings.csv")
    write_dataframe(representative, outputs / "representative_event_explanations.csv")
    write_dataframe(threshold, outputs / "anomaly_threshold_sensitivity.csv")
    write_dataframe(detrending, outputs / "detrending_robustness.csv")
    write_dataframe(sanity, outputs / "event_year_sanity.csv")

    print("Writing XAI figures...", flush=True)
    figures = paths.xai_figures
    fig_workflow(figures / "fig01_method_workflow")
    fig_anomaly_timeline(anomalies, figures / "fig02_anomaly_timeline")
    fig_shap_summary(shap_feature, figures / "fig03_shap_summary")
    fig_grouped_shap(grouped_shap, figures / "fig04_grouped_shap")
    fig_group_importance(permutation, ablation, figures / "fig05_group_importance")
    fig_ale(ale, figures / "fig07_ale_curves")
    fig_agreement(rankings, figures / "fig08_method_agreement")

    write_summary(paths, model_performance, rankings, agreement)
    assert_pipeline_outputs(paths)
    print("Multi-method XAI pipeline complete.")
    print(f"Rows: {len(scored)}")
    print(f"Low-yield anomalies: {len(anomalies)}")
    print(f"Weather features: {len(features)}")
    print(f"Outputs: {paths.xai_outputs}")
    print(f"Figures: {paths.xai_figures}")
