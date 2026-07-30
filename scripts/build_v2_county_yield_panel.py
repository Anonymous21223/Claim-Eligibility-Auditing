"""Build the pre-specified, conflict-free V2 county yield panel from immutable NASS raw files."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "v2_county" / "raw" / "nass"
DATA = ROOT / "data" / "v2_county"
SPEC = ROOT / "configs" / "experiments" / "county_v2_panel_spec.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_variant(frame: pd.DataFrame) -> pd.Series:
    classes = frame["class_desc"].astype(str).str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
    return frame["commodity_desc"].astype(str).str.upper() + "__" + classes


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    files = sorted(RAW.glob("*.json"))
    if not files:
        raise SystemExit("BLOCKED_NO_NASS_RAW")

    rows: list[dict[str, object]] = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        for row in payload.get("data", []):
            row["raw_source_file"] = file.name
        rows.extend(payload.get("data", []))
    source = pd.DataFrame(rows)
    if source.empty:
        raise SystemExit("BLOCKED_EMPTY_NASS_RAW")

    source["year"] = pd.to_numeric(source["year"], errors="coerce")
    source["yield_bu_acre"] = pd.to_numeric(source["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    county = source["county_ansi"].astype(str).str.strip()
    canonical = (
        source["source_desc"].eq("SURVEY")
        & source["agg_level_desc"].eq("COUNTY")
        & source["statisticcat_desc"].eq("YIELD")
        & source["freq_desc"].eq("ANNUAL")
        & source["domain_desc"].eq("TOTAL")
        & source["prodn_practice_desc"].eq("ALL PRODUCTION PRACTICES")
        & source["util_practice_desc"].eq("ALL UTILIZATION PRACTICES")
        & source["unit_desc"].eq("BU / ACRE")
        & county.str.fullmatch(r"\d{3}")
        & county.ne("000")
        & source["year"].notna()
        & source["yield_bu_acre"].notna()
    )
    frame = source.loc[canonical].copy()
    frame["county_ansi"] = county.loc[canonical]
    frame["state_ansi"] = frame["state_ansi"].astype(str).str.zfill(2)
    frame["county_fips"] = frame["state_ansi"] + frame["county_ansi"]
    frame["crop_variant"] = normalized_variant(frame)
    frame["year"] = frame["year"].astype(int)
    keys = ["crop_variant", "county_fips", "year"]
    conflicts = frame.groupby(keys)["yield_bu_acre"].nunique()
    if conflicts.gt(1).any():
        raise SystemExit(f"BLOCKED_CONFLICTING_CANONICAL_VALUES_{int(conflicts.gt(1).sum())}")
    panel = frame.sort_values([*keys, "raw_source_file"]).drop_duplicates(keys, keep="first").copy()

    coverage_rows = []
    for variant, group in panel.groupby("crop_variant", sort=True):
        history = group.groupby("county_fips")["year"].nunique()
        train_history = group[group["year"] <= 2018].groupby("county_fips")["year"].nunique()
        coverage_rows.append(
            {
                "crop_variant": variant,
                "county_year_rows": int(len(group)),
                "counties": int(group["county_fips"].nunique()),
                "first_year": int(group["year"].min()),
                "last_year": int(group["year"].max()),
                "county_series_ge_10_years": int((history >= 10).sum()),
                "county_series_ge_10_train_years": int((train_history >= 10).sum()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    eligible = coverage[coverage["county_series_ge_10_train_years"] >= 500].sort_values("county_year_rows", ascending=False)
    if len(eligible) != 1:
        raise SystemExit(f"BLOCKED_AMBIGUOUS_POPULATION_{len(eligible)}")
    selected = str(eligible.iloc[0]["crop_variant"])
    if selected != spec["expected_selected_variant"]:
        raise SystemExit(f"BLOCKED_UNEXPECTED_POPULATION_{selected}")
    selected_panel = panel[panel["crop_variant"].eq(selected)].copy()
    training_history = selected_panel[selected_panel["year"] <= 2018].groupby("county_fips")["year"].nunique()
    retained_counties = training_history[training_history >= 10].index
    exclusions = pd.DataFrame({"county_fips": training_history.index, "train_year_count": training_history.values})
    exclusions["retained"] = exclusions["county_fips"].isin(retained_counties)
    exclusions_path = ROOT / "reports" / "v2" / "canonical_yield_population_exclusions.csv"
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions.sort_values("county_fips").to_csv(exclusions_path, index=False)
    selected_panel = selected_panel[selected_panel["county_fips"].isin(retained_counties)].copy()
    selected_panel["split_role"] = "train"
    selected_panel.loc[selected_panel["year"].between(2019, 2021), "split_role"] = "validation"
    selected_panel.loc[selected_panel["year"] >= 2022, "split_role"] = "locked_holdout"
    selected_panel["weather_feature_date_max"] = pd.to_datetime(selected_panel["year"].astype(str) + "-12-31")
    selected_panel["target_available_date"] = selected_panel["weather_feature_date_max"]
    selected_panel = selected_panel.rename(columns={"commodity_desc": "crop", "state_ansi": "state_fips"})
    columns = [
        "county_fips", "state_fips", "crop", "crop_variant", "year", "yield_bu_acre", "unit_desc",
        "weather_feature_date_max", "target_available_date", "split_role", "county_name", "state_name", "raw_source_file",
    ]
    output = DATA / "interim" / "county_yield_v2_canonical.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_panel[columns].sort_values(["county_fips", "year"]).to_csv(output, index=False)
    coverage_path = ROOT / "reports" / "v2" / "canonical_yield_population_coverage.csv"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False)
    summary = {
        "status": "PASS_PRE_MODEL_PANEL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec": SPEC.relative_to(ROOT).as_posix(),
        "raw_file_count": len(files),
        "raw_sha256": {file.relative_to(ROOT).as_posix(): sha256(file) for file in files},
        "canonical_rows_before_exact_deduplication": int(len(frame)),
        "canonical_conflicting_groups": 0,
        "selected_crop_variant": selected,
        "selected_rows": int(len(selected_panel)),
        "selected_counties": int(selected_panel["county_fips"].nunique()),
        "excluded_short_training_history_counties": int((~exclusions["retained"]).sum()),
        "year_range": [int(selected_panel["year"].min()), int(selected_panel["year"].max())],
        "split_counts": {key: int(value) for key, value in selected_panel["split_role"].value_counts().sort_index().items()},
        "panel_path": output.relative_to(ROOT).as_posix(),
        "panel_sha256": sha256(output),
        "coverage_path": coverage_path.relative_to(ROOT).as_posix(),
        "exclusions_path": exclusions_path.relative_to(ROOT).as_posix(),
    }
    report_dir = ROOT / "reports" / "experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "county-panel-v2.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Dataset V2 County Yield Panel", "", *[f"- {key}: `{value}`" for key, value in summary.items()], "", "The panel is locked before weather/model fitting. Its selection uses only source-coverage rules declared in the spec."]
    (report_dir / "county-panel-v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
