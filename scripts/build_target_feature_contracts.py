"""Build target, feature-availability, and no-shortcut contract artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.core import (  # noqa: E402
    CAT_FEATURES,
    LEAKAGE_TERMS,
    NON_WEATHER_COLUMNS,
    TARGET,
    driver_group,
    full_season_weather_features,
    load_frame,
    make_project_paths,
    model_feature_columns,
    weather_columns,
)


OUT_DATA = ROOT / "artifacts" / "data"
OUT_TARGETS = ROOT / "artifacts" / "targets"
OUT_AUDIT = ROOT / "artifacts" / "audit_records"
OUT_PAPER = ROOT / "paper" / "generated"


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_feature_availability(frame: pd.DataFrame) -> pd.DataFrame:
    full_weather = set(full_season_weather_features(frame))
    stage_weather = [column for column in weather_columns(frame) if column.endswith(("_early", "_mid", "_late"))]
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            {
                "feature_group": "metadata",
                "feature": feature,
                "available_at_prediction_time": "before season/audit run",
                "window": "static",
                "used_in_primary": True,
                "used_in_sensitivity": False,
                "role": "metadata-only and full models",
            }
            for feature in ["lat", "lon", *CAT_FEATURES]
        ]
    )
    for feature in sorted(full_weather):
        rows.append(
            {
                "feature_group": "weather_full_season",
                "feature": feature,
                "available_at_prediction_time": "after the complete crop-season weather window",
                "window": "full growing season or winter-wheat crop window",
                "used_in_primary": True,
                "used_in_sensitivity": False,
                "role": f"{driver_group(feature)} weather feature",
            }
        )
    for feature in sorted(stage_weather):
        suffix = feature.rsplit("_", 1)[-1]
        rows.append(
            {
                "feature_group": "weather_stage_sensitivity",
                "feature": feature,
                "available_at_prediction_time": f"after {suffix} season segment",
                "window": f"{suffix} third of the crop-season window",
                "used_in_primary": False,
                "used_in_sensitivity": True,
                "role": "stage-proxy sensitivity only",
            }
        )
    return pd.DataFrame(rows)


def build_feature_schema(frame: pd.DataFrame) -> dict[str, object]:
    full_weather = full_season_weather_features(frame)
    numeric_weather, categorical_weather = model_feature_columns(full_weather)
    metadata_numeric = ["lat", "lon"]
    metadata_categorical = list(CAT_FEATURES)
    forbidden_columns = sorted(
        column
        for column in frame.columns
        if column == TARGET
        or column in NON_WEATHER_COLUMNS - {"country", "region", "crop", "year", "window", "lat", "lon"}
        or any(term in column for term in LEAKAGE_TERMS)
    )
    schema = {
        "raw_yield_column": TARGET,
        "target": "trend_residual_t_ha",
        "prediction_task": "post-season scientific audit of train-only detrended yield residuals",
        "primary_weather_feature_count": len(full_weather),
        "metadata_only": {"numeric": metadata_numeric, "categorical": metadata_categorical},
        "weather_only": {"numeric": full_weather, "categorical": []},
        "full": {"numeric": numeric_weather, "categorical": categorical_weather},
        "forbidden_target_derived_columns": forbidden_columns,
        "forbidden_terms": list(LEAKAGE_TERMS),
        "model_matrices_exclude_year": True,
        "model_matrices_exclude_target": True,
    }
    schema["schema_sha256"] = digest_text(json.dumps({key: value for key, value in schema.items() if key != "schema_sha256"}, sort_keys=True))
    return schema


def build_overlap_audit(frame: pd.DataFrame, schema: dict[str, object]) -> pd.DataFrame:
    model_columns = set(schema["metadata_only"]["numeric"] + schema["metadata_only"]["categorical"] + schema["weather_only"]["numeric"])
    rows = []
    for column in frame.columns:
        target_derived = column in schema["forbidden_target_derived_columns"] or any(term in column for term in LEAKAGE_TERMS)
        calendar_or_history = column in {"year"} or column.startswith("n_history") or "history" in column or "scale" in column
        rows.append(
            {
                "column": column,
                "in_model_matrix": column in model_columns,
                "semantic_role": "target_derived" if target_derived else "calendar_or_history" if calendar_or_history else "allowed_feature_or_identifier",
                "forbidden_reason": "target/residual/event/prediction derived" if target_derived else "calendar/history shortcut not used by model matrix" if calendar_or_history else "",
                "status": "FAIL" if (column in model_columns and (target_derived or calendar_or_history)) else "PASS",
            }
        )
    return pd.DataFrame(rows)


def build_no_shortcut_ablation(overlap: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("target_derived_columns", "Remove yield/trend/residual/z/event/prediction columns", "absent_from_model_matrix"),
        ("calendar_year", "Remove year from model features", "absent_from_model_matrix"),
        ("history_or_scale_columns", "Remove history length and residual scale columns", "absent_from_model_matrix"),
    ]
    rows = []
    for group, action, evidence in checks:
        failed = overlap[(overlap.status == "FAIL") & (overlap.semantic_role.isin(["target_derived", "calendar_or_history"]))]
        rows.append(
            {
                "ablation_group": group,
                "required_action": action,
                "protocol": "same primary target/split/population; no retraining needed because columns are absent by construction",
                "evidence": evidence,
                "violating_columns": ";".join(failed.column.astype(str).tolist()),
                "status": "PASS" if failed.empty else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def write_target_spec(frame: pd.DataFrame, schema: dict[str, object]) -> None:
    text = f"""# Target and Feature Contract

## Intended Use

The V3 state-level experiment is a post-season scientific audit. It predicts the
raw train-only detrended yield residual after the full crop-season weather window
is available. It is not a pre-harvest forecast and does not claim causal
attribution.

## Target

For crop-state series `(c,s)` and year `t`, fit a linear trend only on training
years:

```text
y_hat_{{c,s,t}} = a_{{c,s}}^{{train}} + b_{{c,s}}^{{train}} t
r_{{c,s,t}} = y_{{c,s,t}} - y_hat_{{c,s,t}}
z_{{c,s,t}} = r_{{c,s,t}} / max(sigma_{{c,s,train}}, epsilon)
Event_{{c,s,t}}(tau) = 1[z_{{c,s,t}} < tau]
```

- Prediction target: `trend_residual_t_ha` (`r`), not the event indicator.
- Event threshold: `z<-1` primary; `z<-1.5` and `z<-2` sensitivity.
- Scale: training residual standard deviation with `ddof=1`; current training
  scales are audited in `artifacts/targets/train_scale_diagnostics.csv`.
- Minimum history: three prior training rows, a technical minimum for trend/scale
  fitting rather than a stable-variance guarantee.
- Evaluation years are never used to fit trend slope/intercept or residual scale.

## Feature Families

- Metadata-only: `lat`, `lon`, `region`, `crop`.
- Weather-only: {schema["primary_weather_feature_count"]} full-season weather features.
- Full: metadata plus weather.

Feature availability and window details are machine-readable in
`artifacts/data/feature_availability.csv`. Forbidden target-derived columns are
listed in `artifacts/data/feature_matrix_schema.json` and audited in
`artifacts/audit_records/target_feature_overlap.csv`.

## Current Frame

- Rows: {len(frame)}
- Years: {int(frame.year.min())}-{int(frame.year.max())}
- Crops: {", ".join(sorted(frame.crop.unique()))}
"""
    OUT_TARGETS.mkdir(parents=True, exist_ok=True)
    (OUT_TARGETS / "target_spec.md").write_text(text, encoding="utf-8")


def write_feature_availability_table(availability: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        [
            {
                "Feature family": "Metadata",
                "Count": int((availability.feature_group == "metadata").sum()),
                "Availability": "Static / before audit",
                "Role": "Metadata-only and Full",
            },
            {
                "Feature family": "Full-season weather",
                "Count": int((availability.feature_group == "weather_full_season").sum()),
                "Availability": "After complete crop-season window",
                "Role": "Weather-only and Full",
            },
            {
                "Feature family": "Stage weather proxies",
                "Count": int((availability.feature_group == "weather_stage_sensitivity").sum()),
                "Availability": "After early/mid/late segment",
                "Role": "Sensitivity only",
            },
        ]
    )
    write_csv(summary, OUT_DATA / "feature_availability_summary.csv")
    lines = [
        r"\begin{tabular}{lcll}",
        r"\toprule",
        r"Feature family & Count & Availability & Role \\",
        r"\midrule",
    ]
    for item in summary.itertuples(index=False):
        lines.append(f"{item[0]} & {item[1]} & {item[2]} & {item[3]} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    (OUT_PAPER / "table_feature_availability.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = load_frame(make_project_paths(ROOT))
    availability = build_feature_availability(frame)
    schema = build_feature_schema(frame)
    overlap = build_overlap_audit(frame, schema)
    no_shortcut = build_no_shortcut_ablation(overlap)

    write_csv(availability, OUT_DATA / "feature_availability.csv")
    write_csv(overlap, OUT_AUDIT / "target_feature_overlap.csv")
    write_csv(no_shortcut, OUT_AUDIT / "no_shortcut_ablation.csv")
    (OUT_DATA / "feature_matrix_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    write_target_spec(frame, schema)
    write_feature_availability_table(availability)
    if (overlap.status == "FAIL").any() or (no_shortcut.status == "FAIL").any():
        raise AssertionError("Target/feature no-shortcut audit failed")
    print("Target, feature availability, and no-shortcut contracts written.")


if __name__ == "__main__":
    main()
