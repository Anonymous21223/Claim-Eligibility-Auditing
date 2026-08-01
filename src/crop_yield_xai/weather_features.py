"""Rebuild the model weather features from the bundled NASA POWER daily files."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


DAILY_FIELDS = ("T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN")
SPRING_START = (4, 1)
SPRING_END = (9, 30)
WINTER_START = (9, 1)
WINTER_END = (6, 30)
WINTER_WHEAT_REGIONS = frozenset({"Colorado", "Illinois", "Iowa", "Kansas", "Nebraska", "Oklahoma", "Texas", "Washington"})


def _spec(feature: str, field: str, formula: str, unit: str, window: str = "full growing season") -> dict[str, str]:
    return {"feature": feature, "nasa_field": field, "formula": formula, "unit": unit, "window": window}


FEATURE_SPECS = (
    _spec("rain_sum", "PRECTOTCORR", "sum(precipitation)", "mm"),
    _spec("rain_mean", "PRECTOTCORR", "mean(precipitation)", "mm day-1"),
    _spec("dry_days_1mm", "PRECTOTCORR", "count(precipitation < 1 mm)", "days"),
    _spec("dry_days_2mm", "PRECTOTCORR", "count(precipitation < 2 mm)", "days"),
    _spec("max_dry_spell_1mm", "PRECTOTCORR", "longest consecutive run with precipitation < 1 mm", "days"),
    _spec("dry_spell_events_7d", "PRECTOTCORR", "count(consecutive dry runs >= 7 days; precipitation < 1 mm)", "events"),
    _spec("dry_spell_events_14d", "PRECTOTCORR", "count(consecutive dry runs >= 14 days; precipitation < 1 mm)", "events"),
    _spec("heavy_rain_days_10", "PRECTOTCORR", "count(precipitation >= 10 mm)", "days"),
    _spec("heavy_rain_days_20", "PRECTOTCORR", "count(precipitation >= 20 mm)", "days"),
    _spec("heavy_rain_days_25", "PRECTOTCORR", "count(precipitation >= 25 mm)", "days"),
    _spec("heavy_rain_days_50", "PRECTOTCORR", "count(precipitation >= 50 mm)", "days"),
    _spec("max_1day_rain", "PRECTOTCORR", "max(1-day precipitation)", "mm"),
    _spec("max_3day_rain", "PRECTOTCORR", "max trailing 3-day precipitation sum", "mm"),
    _spec("max_7day_rain", "PRECTOTCORR", "max trailing 7-day precipitation sum", "mm"),
    _spec("wet_days_1mm", "PRECTOTCORR", "count(precipitation >= 1 mm)", "days"),
    _spec("season_tmax_mean", "T2M_MAX", "mean(maximum daily temperature)", "degC"),
    _spec("season_tmin_mean", "T2M_MIN", "mean(minimum daily temperature)", "degC"),
    _spec("season_tmean_mean", "T2M_MAX,T2M_MIN", "mean((T2M_MAX + T2M_MIN) / 2)", "degC"),
    _spec("growing_degree_days_base5", "T2M_MAX,T2M_MIN", "sum(max(((T2M_MAX + T2M_MIN) / 2) - 5, 0))", "degC day"),
    _spec("heat_days_30", "T2M_MAX", "count(T2M_MAX >= 30 degC)", "days"),
    _spec("heat_days_35", "T2M_MAX", "count(T2M_MAX >= 35 degC)", "days"),
    _spec("heat_days_40", "T2M_MAX", "count(T2M_MAX >= 40 degC)", "days"),
    _spec("max_consecutive_heat_days_30", "T2M_MAX", "longest consecutive run with T2M_MAX >= 30 degC", "days"),
    _spec("max_consecutive_heat_days_35", "T2M_MAX", "longest consecutive run with T2M_MAX >= 35 degC", "days"),
    _spec("heatwave_events_3d_30", "T2M_MAX", "count(consecutive heat runs >= 3 days; T2M_MAX >= 30 degC)", "events"),
    _spec("heatwave_events_3d_35", "T2M_MAX", "count(consecutive heat runs >= 3 days; T2M_MAX >= 35 degC)", "events"),
    _spec("heat_degree_days_30", "T2M_MAX", "sum(max(T2M_MAX - 30, 0))", "degC day"),
    _spec("heat_degree_days_35", "T2M_MAX", "sum(max(T2M_MAX - 35, 0))", "degC day"),
    _spec("hot_dry_days_30_1mm", "T2M_MAX,PRECTOTCORR", "count(T2M_MAX >= 30 degC and precipitation < 1 mm)", "days"),
    _spec("frost_days_0", "T2M_MIN", "count(T2M_MIN <= 0 degC)", "days"),
    _spec("cold_days_5", "T2M_MIN", "count(T2M_MIN <= 5 degC)", "days"),
    _spec("min_tmin", "T2M_MIN", "min(T2M_MIN)", "degC"),
    _spec("frost_events_2d", "T2M_MIN", "count(consecutive frost runs >= 2 days; T2M_MIN <= 0 degC)", "events"),
    _spec("radiation_sum", "ALLSKY_SFC_SW_DWN", "sum(daily surface shortwave radiation)", "kWh m-2"),
    _spec("radiation_mean", "ALLSKY_SFC_SW_DWN", "mean(daily surface shortwave radiation)", "kWh m-2 day-1"),
)


def crop_window(crop: str, region: str) -> str:
    return "winter" if crop == "Wheat" and region in WINTER_WHEAT_REGIONS else "spring"


def season_bounds(year: int, window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window == "spring":
        return pd.Timestamp(year, *SPRING_START), pd.Timestamp(year, *SPRING_END)
    if window == "winter":
        return pd.Timestamp(year - 1, *WINTER_START), pd.Timestamp(year, *WINTER_END)
    raise ValueError(f"Unknown crop window: {window}")


def run_lengths(mask: pd.Series | np.ndarray) -> list[int]:
    lengths: list[int] = []
    run = 0
    for value in mask:
        if value:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _round(value: float) -> float:
    return round(float(value), 3)


def _count_runs(mask: pd.Series, minimum_length: int) -> int:
    return sum(length >= minimum_length for length in run_lengths(mask.to_numpy()))


def _full_features(daily: pd.DataFrame) -> dict[str, float | int]:
    rain = daily["PRECTOTCORR"]
    tmax = daily["T2M_MAX"]
    tmin = daily["T2M_MIN"]
    tmean = (tmax + tmin) / 2
    dry_1 = rain < 1
    heat_30 = tmax >= 30
    heat_35 = tmax >= 35
    frost = tmin <= 0
    return {
        "rain_sum": _round(rain.sum()),
        "rain_mean": _round(rain.mean()),
        "dry_days_1mm": int(dry_1.sum()),
        "dry_days_2mm": int((rain < 2).sum()),
        "max_dry_spell_1mm": max(run_lengths(dry_1.to_numpy()), default=0),
        "dry_spell_events_7d": _count_runs(dry_1, 7),
        "dry_spell_events_14d": _count_runs(dry_1, 14),
        "heavy_rain_days_10": int((rain >= 10).sum()),
        "heavy_rain_days_20": int((rain >= 20).sum()),
        "heavy_rain_days_25": int((rain >= 25).sum()),
        "heavy_rain_days_50": int((rain >= 50).sum()),
        "max_1day_rain": _round(rain.max()),
        "max_3day_rain": _round(rain.rolling(3).sum().max()),
        "max_7day_rain": _round(rain.rolling(7).sum().max()),
        "wet_days_1mm": int((rain >= 1).sum()),
        "season_tmax_mean": _round(tmax.mean()),
        "season_tmin_mean": _round(tmin.mean()),
        "season_tmean_mean": _round(tmean.mean()),
        "growing_degree_days_base5": _round(np.maximum(tmean - 5, 0).sum()),
        "heat_days_30": int(heat_30.sum()),
        "heat_days_35": int(heat_35.sum()),
        "heat_days_40": int((tmax >= 40).sum()),
        "max_consecutive_heat_days_30": max(run_lengths(heat_30.to_numpy()), default=0),
        "max_consecutive_heat_days_35": max(run_lengths(heat_35.to_numpy()), default=0),
        "heatwave_events_3d_30": _count_runs(heat_30, 3),
        "heatwave_events_3d_35": _count_runs(heat_35, 3),
        "heat_degree_days_30": _round(np.maximum(tmax - 30, 0).sum()),
        "heat_degree_days_35": _round(np.maximum(tmax - 35, 0).sum()),
        "hot_dry_days_30_1mm": int((heat_30 & dry_1).sum()),
        "frost_days_0": int(frost.sum()),
        "cold_days_5": int((tmin <= 5).sum()),
        "min_tmin": _round(tmin.min()),
        "frost_events_2d": _count_runs(frost, 2),
        "radiation_sum": _round(daily["ALLSKY_SFC_SW_DWN"].sum()),
        "radiation_mean": _round(daily["ALLSKY_SFC_SW_DWN"].mean()),
    }


def _stage_features(daily: pd.DataFrame) -> dict[str, float | int]:
    n = len(daily) // 3
    stages = {"early": daily.iloc[:n], "mid": daily.iloc[n:2 * n], "late": daily.iloc[2 * n:]}
    features: dict[str, float | int] = {}
    for stage, part in stages.items():
        rain = part["PRECTOTCORR"]
        features.update({
            f"rain_sum_{stage}": _round(rain.sum()),
            f"heat_days_35_{stage}": int((part["T2M_MAX"] >= 35).sum()),
            f"frost_days_0_{stage}": int((part["T2M_MIN"] <= 0).sum()),
            f"radiation_sum_{stage}": _round(part["ALLSKY_SFC_SW_DWN"].sum()),
            f"max_3day_rain_{stage}": _round(rain.rolling(3).sum().max()),
        })
    return features


def load_nasa_power_daily(archive: Path) -> dict[str, pd.DataFrame]:
    daily_by_region: dict[str, pd.DataFrame] = {}
    with ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not name.endswith(".csv"):
                continue
            daily = pd.read_csv(zipped.open(name))
            missing = set(DAILY_FIELDS).difference(daily.columns)
            if missing:
                raise ValueError(f"{name} is missing daily fields: {sorted(missing)}")
            daily["date"] = pd.to_datetime(daily["YEAR"].astype(str) + daily["DOY"].astype(str).str.zfill(3), format="%Y%j")
            region = str(daily["region"].iloc[0])
            if region in daily_by_region:
                raise ValueError(f"Duplicate daily archive input for {region}")
            daily_by_region[region] = daily.sort_values("date").reset_index(drop=True)
    return daily_by_region


def build_model_frame(raw_yield: pd.DataFrame, daily_by_region: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for yield_row in raw_yield[raw_yield["year"] >= 1990].itertuples(index=False):
        window = crop_window(yield_row.crop, yield_row.region)
        start, end = season_bounds(int(yield_row.year), window)
        daily = daily_by_region[yield_row.region]
        season = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy().reset_index(drop=True)
        expected_days = pd.date_range(start, end, freq="D")
        if len(season) != len(expected_days) or not season["date"].equals(pd.Series(expected_days, name="date")):
            raise ValueError(f"Incomplete or non-daily NASA POWER input for {yield_row.region}, {yield_row.crop}, {yield_row.year}")
        row: dict[str, object] = {
            "country": yield_row.country,
            "region": yield_row.region,
            "crop": yield_row.crop,
            "year": int(yield_row.year),
            "window": window,
            "season_days": float(len(season)),
        }
        row.update(_full_features(season))
        row.update(_stage_features(season))
        row["lat"] = float(season["lat"].iloc[0])
        row["lon"] = float(season["lon"].iloc[0])
        row["yield_t_ha"] = float(yield_row.yield_t_ha)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_frames(expected: pd.DataFrame, reconstructed: pd.DataFrame, atol: float = 5e-4) -> pd.DataFrame:
    if list(expected.columns) != list(reconstructed.columns):
        raise AssertionError("Reconstructed frame columns do not match the processed frame")
    keys = ["country", "region", "crop", "year"]
    expected = expected.sort_values(keys).reset_index(drop=True)
    reconstructed = reconstructed.sort_values(keys).reset_index(drop=True)
    if not expected[keys].equals(reconstructed[keys]):
        raise AssertionError("Reconstructed crop-state-year keys do not match the processed frame")
    records = []
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            delta = (expected[column].astype(float) - reconstructed[column].astype(float)).abs()
            records.append({"column": column, "max_abs_difference": float(delta.max()), "mismatch_count": int((delta > atol).sum())})
        else:
            mismatch = expected[column].astype(str) != reconstructed[column].astype(str)
            records.append({"column": column, "max_abs_difference": 0.0, "mismatch_count": int(mismatch.sum())})
    report = pd.DataFrame(records)
    if report["mismatch_count"].sum():
        failures = report[report["mismatch_count"] > 0].to_dict("records")
        raise AssertionError(f"Raw-weather reconstruction differs from processed frame: {failures}")
    return report
