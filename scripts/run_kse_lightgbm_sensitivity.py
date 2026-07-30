"""Single-specification LightGBM sensitivity under the existing locked protocol.

The model class is changed while the target construction, temporal split,
feature families, preprocessing, evaluation rows, and year-block uncertainty
design remain fixed.  One deterministic LightGBM specification is evaluated.
The best feature family is chosen on 2012--2015 validation rows only.

Because the 2016--2025 test was already reported in earlier manuscript
versions, this result is explicitly a post-hoc model-family sensitivity and
cannot replace the primary validation-selected ExtraTrees verdict.
"""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_yield_xai.core import load_frame, make_project_paths  # noqa: E402
from run_main8_audit import (  # noqa: E402
    FINAL_TEST,
    FINAL_TRAIN_END,
    SELECTION_END,
    VALIDATION,
    feature_sets,
    metric_dict,
    paired_delta,
    row_id,
    score_fold,
)


OUT_DIR = ROOT / "artifacts" / "experiments" / "kse-v5-8" / "lightgbm"
BOOTSTRAP_DRAWS = ROOT / "artifacts" / "audit" / "bootstrap" / "year_block_draws.csv"
MODEL_SPEC = {
    "objective": "regression",
    "n_estimators": 160,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "random_state": 20260730,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbosity": -1,
}
STABLE_DECIMALS = 12


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_float(value: object) -> float:
    return round(float(value), STABLE_DECIMALS)


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def fitted_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = LGBMRegressor(**MODEL_SPEC)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_family(
    train: pd.DataFrame,
    test: pd.DataFrame,
    family: str,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    columns = numeric + categorical
    model = fitted_pipeline(numeric, categorical)
    model.fit(train[columns], train["trend_residual_t_ha"])
    result = test[
        [
            "country",
            "region",
            "crop",
            "year",
            "window",
            "yield_t_ha",
            "trend_residual_t_ha",
            "trend_residual_z",
            "is_low_yield_anomaly",
        ]
    ].copy()
    result.insert(0, "row_id", row_id(result))
    result["prediction"] = model.predict(test[columns])
    result["model"] = "LightGBM"
    result["feature_family"] = family
    result["config_id"] = "lightgbm_kse_fixed"
    return result


def summary_row(frame: pd.DataFrame, split: str) -> dict[str, object]:
    metrics = metric_dict(frame["trend_residual_t_ha"], frame["prediction"])
    return {
        "split": split,
        "model": "LightGBM",
        "feature_family": str(frame["feature_family"].iloc[0]),
        "n": int(len(frame)),
        **metrics,
    }


def main() -> None:
    frame = load_frame(make_project_paths(ROOT))
    sets = feature_sets(frame)
    validation_train, validation_test, _ = score_fold(
        frame, SELECTION_END, *VALIDATION
    )
    final_train, final_test, _ = score_fold(
        frame, FINAL_TRAIN_END, *FINAL_TEST
    )

    validation_outputs = []
    for family, (numeric, categorical) in sets.items():
        validation_outputs.append(
            fit_family(
                validation_train,
                validation_test,
                family,
                numeric,
                categorical,
            )
        )
    validation_predictions = pd.concat(validation_outputs, ignore_index=True)
    validation_summary = pd.DataFrame(
        [
            summary_row(group, "validation_2012_2015")
            for _, group in validation_predictions.groupby(
                "feature_family", sort=True
            )
        ]
    ).sort_values(["rmse_t_ha", "feature_family"], kind="stable")
    selected_family = str(validation_summary.iloc[0]["feature_family"])
    validation_summary["selected_on_validation"] = (
        validation_summary["feature_family"] == selected_family
    )

    locked_outputs = []
    for family, (numeric, categorical) in sets.items():
        locked_outputs.append(
            fit_family(final_train, final_test, family, numeric, categorical)
        )
    locked_predictions = pd.concat(locked_outputs, ignore_index=True)
    locked_summary = pd.DataFrame(
        [
            summary_row(group, "locked_2016_2025")
            for _, group in locked_predictions.groupby(
                "feature_family", sort=True
            )
        ]
    )

    zero = final_test[
        ["crop", "region", "year", "window", "trend_residual_t_ha"]
    ].copy()
    zero.insert(0, "row_id", row_id(final_test))
    zero["prediction"] = 0.0
    draws = pd.read_csv(BOOTSTRAP_DRAWS)
    selected = locked_predictions[
        locked_predictions["feature_family"] == selected_family
    ].copy()
    full = locked_predictions[
        locked_predictions["feature_family"] == "full"
    ].copy()
    metadata = locked_predictions[
        locked_predictions["feature_family"] == "metadata_only"
    ].copy()

    module_a = paired_delta(
        selected,
        zero,
        draws,
        "lightgbm_validation_selected_vs_zero",
    )
    module_b = paired_delta(
        full,
        metadata,
        draws,
        "lightgbm_full_vs_metadata_only",
    )
    comparisons = pd.concat([module_a, module_b], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_summary.round(STABLE_DECIMALS).to_csv(
        OUT_DIR / "lightgbm_validation_summary.csv",
        index=False,
        lineterminator="\n",
    )
    locked_summary.round(STABLE_DECIMALS).to_csv(
        OUT_DIR / "lightgbm_locked_summary.csv",
        index=False,
        lineterminator="\n",
    )
    comparisons.round(STABLE_DECIMALS).to_csv(
        OUT_DIR / "lightgbm_paired_comparisons.csv",
        index=False,
        lineterminator="\n",
    )

    module_a_rmse = module_a[module_a["metric"] == "rmse_t_ha"].iloc[0]
    module_b_rmse = module_b[module_b["metric"] == "rmse_t_ha"].iloc[0]
    locked_selected = locked_summary[
        locked_summary["feature_family"] == selected_family
    ].iloc[0]
    summary = {
        "analysis": "Single-specification LightGBM model-family sensitivity",
        "status": "post-hoc sensitivity; does not replace the primary model",
        "lightgbm_version": __import__("lightgbm").__version__,
        "model_spec": MODEL_SPEC,
        "selection_period": "2012-2015",
        "locked_period": "2016-2025",
        "selected_feature_family": selected_family,
        "validation_rows": int(len(validation_test)),
        "locked_rows": int(len(final_test)),
        "locked_rmse_t_ha": stable_float(locked_selected["rmse_t_ha"]),
        "locked_r2": stable_float(locked_selected["r2"]),
        "module_a_delta_rmse_t_ha": stable_float(
            module_a_rmse["delta_left_minus_right"]
        ),
        "module_a_ci95_low": stable_float(module_a_rmse["ci95_low"]),
        "module_a_ci95_high": stable_float(module_a_rmse["ci95_high"]),
        "module_a_pass": bool(module_a_rmse["ci95_high"] < 0.0),
        "module_b_delta_rmse_t_ha": stable_float(
            module_b_rmse["delta_left_minus_right"]
        ),
        "module_b_ci95_low": stable_float(module_b_rmse["ci95_low"]),
        "module_b_ci95_high": stable_float(module_b_rmse["ci95_high"]),
        "module_b_pass": bool(module_b_rmse["ci95_high"] < 0.0),
        "bootstrap_draws": BOOTSTRAP_DRAWS.relative_to(ROOT).as_posix(),
        "bootstrap_draws_sha256": file_sha256(BOOTSTRAP_DRAWS),
    }
    write_text_lf(
        OUT_DIR / "lightgbm_summary.json",
        json.dumps(summary, indent=2),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
