"""Temporal, crop, spatial, and phenology robustness checks on locked targets."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_yield_xai.core import full_season_weather_features, load_frame, make_project_paths  # noqa: E402
from run_revision_audit import SEEDS, metrics, model_pipeline, score_fold  # noqa: E402


def evaluate(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str = "extra_trees") -> dict[str, float]:
    numeric, categorical = ["lat", "lon"] + features, ["crop", "region"]
    model = model_pipeline(numeric, categorical, model_name, SEEDS[0])
    model.fit(train[numeric + categorical], train["trend_residual_t_ha"])
    return metrics(test["trend_residual_t_ha"], model.predict(test[numeric + categorical]))


def write(frame: pd.DataFrame, name: str) -> None:
    path = ROOT / "artifacts" / "metrics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    frame = load_frame(make_project_paths(ROOT))
    full_features = full_season_weather_features(frame)
    folds = [(2003, 2004, 2006), (2006, 2007, 2009), (2009, 2010, 2012), (2012, 2013, 2015), (2015, 2016, 2025)]
    rolling = []
    yearly = []
    for number, (train_end, test_start, test_end) in enumerate(folds, 1):
        train, test, audit = score_fold(frame, train_end, test_start, test_end)
        result = {"fold": number, "train_end": train_end, "test_start": test_start, "test_end": test_end, "n_train": len(train), "n_test": len(test)}
        result.update(evaluate(train, test, full_features))
        rolling.append(result)
        for year, subset in test.groupby("year"):
            row = {"fold": number, "year": int(year), "n_test": len(subset)}
            row.update(evaluate(train, subset, full_features))
            yearly.append(row)
    write(pd.DataFrame(rolling), "rolling_origin_results.csv")
    write(pd.DataFrame(yearly), "yearwise_metrics.csv")

    train, test, _ = score_fold(frame, 2015, 2016, 2025)
    pooling = []
    pooled = {"model_scope": "pooled", "crop": "all", "n_train": len(train), "n_test": len(test)}
    pooled.update(evaluate(train, test, full_features))
    pooling.append(pooled)
    for crop in sorted(test.crop.unique()):
        train_crop, test_crop = train[train.crop == crop], test[test.crop == crop]
        row = {"model_scope": "crop_specific", "crop": crop, "n_train": len(train_crop), "n_test": len(test_crop)}
        row.update(evaluate(train_crop, test_crop, full_features))
        pooling.append(row)
    write(pd.DataFrame(pooling), "pooling_comparison.csv")
    write(pd.DataFrame(pooling), "crop_specific_results.csv")

    spatial = []
    for region in sorted(test.region.unique()):
        state_train, state_test = train[train.region != region], test[test.region == region]
        row = {"held_out_region": region, "n_train": len(state_train), "n_test": len(state_test)}
        row.update(evaluate(state_train, state_test, full_features))
        spatial.append(row)
    write(pd.DataFrame(spatial), "leave_one_state_out.csv")
    write(pd.DataFrame(spatial), "spatial_cv_results.csv")

    stage_features = [column for column in frame.columns if column.endswith(("_early", "_mid", "_late"))]
    phenology = []
    for name, features in [("full_season_only", full_features), ("full_season_plus_stage_proxies", full_features + stage_features)]:
        row = {"feature_design": name, "n_weather_features": len(features)}
        row.update(evaluate(train, test, features))
        phenology.append(row)
    write(pd.DataFrame(phenology), "phenology_feature_results.csv")
    calendar = pd.DataFrame([{"crop": crop, "window": ", ".join(sorted(group.window.unique())), "calendar_status": "exact crop calendar / stage boundaries are not documented in repository"} for crop, group in frame.groupby("crop")])
    write(calendar, "calendar_definition.csv")
    print("Robustness audits written.")


if __name__ == "__main__":
    main()
