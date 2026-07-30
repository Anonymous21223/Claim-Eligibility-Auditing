"""Aggregate validated NASA POWER daily weather and enforce the V2 temporal join contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "v2_county" / "interim" / "county_yield_v2_canonical.csv"
CENTROIDS = ROOT / "data" / "v2_county" / "interim" / "county_centroids.csv"
RAW = ROOT / "data" / "v2_county" / "raw" / "nasa_power"
OUTPUT = ROOT / "data" / "v2_county" / "processed" / "county_winter_wheat_weather_panel.csv"
REPORT = ROOT / "reports" / "v2" / "weather_feature_coverage.json"
PARAMETERS = ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values <= -900)


def main() -> None:
    panel = pd.read_csv(PANEL, dtype={"county_fips": str})
    centroids = pd.read_csv(CENTROIDS, dtype={"county_fips": str})
    panel = panel.merge(centroids[["county_fips", "county_lat", "county_lon"]], on="county_fips", how="left", validate="many_to_one")
    if panel[["county_lat", "county_lon"]].isna().any(axis=1).any():
        raise SystemExit("BLOCKED_MISSING_CENTROID")
    records: list[dict[str, object]] = []
    missing_files: list[str] = []
    for county in sorted(panel["county_fips"].unique()):
        path = RAW / f"{county}.json"
        if not path.exists():
            missing_files.append(county)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("properties", {}).get("parameter", {})
        if not all(parameter in values for parameter in PARAMETERS):
            missing_files.append(county)
            continue
        daily = pd.DataFrame({parameter: pd.Series(values[parameter]) for parameter in PARAMETERS})
        daily.index = pd.to_datetime(daily.index, format="%Y%m%d", errors="coerce")
        daily = daily[~daily.index.isna()].copy()
        daily["year"] = daily.index.year
        for year, group in daily.groupby("year"):
            count = int(group["T2M"].replace(-999, pd.NA).notna().sum())
            records.append({
                "county_fips": county,
                "year": int(year),
                "weather_day_count": count,
                "t2m_mean_c": float(valid(group["T2M"]).mean()),
                "t2m_max_c": float(valid(group["T2M_MAX"]).max()),
                "t2m_min_c": float(valid(group["T2M_MIN"]).min()),
                "precip_total_mm": float(valid(group["PRECTOTCORR"]).sum()),
                "weather_feature_date_max": f"{year}-12-31",
            })
    if missing_files:
        raise SystemExit(f"BLOCKED_MISSING_NASA_RAW_{len(missing_files)}")
    weather = pd.DataFrame(records)
    joined = panel.merge(weather, on=["county_fips", "year"], how="left", validate="one_to_one", suffixes=("", "_weather"))
    feature_columns = ["t2m_mean_c", "t2m_max_c", "t2m_min_c", "precip_total_mm"]
    missing_features = int(joined[feature_columns].isna().any(axis=1).sum())
    temporal_ok = pd.to_datetime(joined["weather_feature_date_max_weather"]) <= pd.to_datetime(joined["target_available_date"])
    if missing_features or not temporal_ok.all():
        raise SystemExit(f"BLOCKED_WEATHER_JOIN_missing={missing_features}_temporal_fail={int((~temporal_ok).sum())}")
    joined = joined.drop(columns=["weather_feature_date_max"]).rename(columns={"weather_feature_date_max_weather": "weather_feature_date_max"})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joined.sort_values(["county_fips", "year"]).to_csv(OUTPUT, index=False)
    report = {
        "status": "PASS",
        "panel_sha256": digest(PANEL),
        "centroid_sha256": digest(CENTROIDS),
        "weather_counties": int(weather["county_fips"].nunique()),
        "weather_rows": int(len(weather)),
        "joined_rows": int(len(joined)),
        "missing_feature_rows": missing_features,
        "temporal_contract_failures": int((~temporal_ok).sum()),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "output_sha256": digest(OUTPUT),
        "feature_columns": feature_columns,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
