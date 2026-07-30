"""E09: pre-specified additional tabular baselines under the locked protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from crop_yield_xai.core import full_season_weather_features, load_frame, make_project_paths  # noqa: E402
from run_main8_audit import FINAL_TEST, FINAL_TRAIN_END, SELECTION_END, VALIDATION, metric_dict, paired_delta, row_id, score_fold  # noqa: E402


def fitted(kind: str) -> Pipeline:
    estimator = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=20260718) if kind == "hist_gradient_boosting" else ElasticNet(alpha=.05, l1_ratio=.5, max_iter=10000, random_state=20260718)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])


def main() -> None:
    frame = load_frame(make_project_paths(ROOT)); features = full_season_weather_features(frame)
    val_train, val_test, _ = score_fold(frame, SELECTION_END, *VALIDATION)
    final_train, final_test, _ = score_fold(frame, FINAL_TRAIN_END, *FINAL_TEST)
    draws = pd.read_csv(ROOT / "artifacts" / "audit" / "bootstrap" / "year_block_draws.csv")
    rows, predictions = [], []
    for kind in ("hist_gradient_boosting", "elastic_net"):
        validation = fitted(kind).fit(val_train[features], val_train.trend_residual_t_ha)
        val_prediction = validation.predict(val_test[features])
        final = fitted(kind).fit(final_train[features], final_train.trend_residual_t_ha)
        prediction = final.predict(final_test[features])
        out = final_test[["crop", "region", "year", "trend_residual_t_ha"]].copy()
        out.insert(0, "row_id", row_id(final_test))
        out["prediction"] = prediction
        zero = out.copy(); zero["prediction"] = 0.0
        paired = paired_delta(out, zero, draws, f"{kind}_weather_only_vs_zero")
        rmse = paired[paired.metric == "rmse_t_ha"].iloc[0]
        baseline = metric_dict(final_test.trend_residual_t_ha, np.zeros(len(final_test)))
        row = {"model": kind, "feature_family": "weather_only", "n_features": len(features), "validation_rmse_t_ha": metric_dict(val_test.trend_residual_t_ha, val_prediction)["rmse_t_ha"], "n_final": len(final_test), "baseline_rmse_t_ha": baseline["rmse_t_ha"], "delta_rmse_t_ha": rmse.delta_left_minus_right, "delta_rmse_ci95_low": rmse.ci95_low, "delta_rmse_ci95_high": rmse.ci95_high, "status": "SENSITIVITY PASS" if rmse.ci95_high < 0 else "SENSITIVITY FAIL"}
        row.update(metric_dict(final_test.trend_residual_t_ha, prediction)); rows.append(row)
        out["model"] = kind; predictions.append(out)
    output = ROOT / "artifacts" / "audit_records"; output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "expanded_model_baselines.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "expanded_model_predictions.csv", index=False)
    print("Expanded model baselines written.")


if __name__ == "__main__":
    main()
