"""Fold-safe calendar and GDD stage features for extension-factorial-v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crop_yield_xai.weather_features import (
    load_nasa_power_daily,
    season_bounds,
)


STAGE_FEATURE_NAMES = (
    "rain_sum",
    "heat_days_35",
    "frost_days_0",
    "radiation_sum",
    "max_3day_rain",
)
BOUNDARY_STAGES = {
    "Barley": ("JOINTING", "HEADED"),
    "Canola": ("BLOOMING", "COLORING"),
    "Oats": ("JOINTING", "HEADED"),
    "Wheat": ("JOINTING", "HEADED"),
}


@dataclass(frozen=True)
class StageDefinitions:
    cutoff_year: int
    calendar: pd.DataFrame
    gdd: pd.DataFrame
    coverage: pd.DataFrame


def _value_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def load_progress(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["progress_pct"] = _value_number(frame["Value"])
    frame["date"] = pd.to_datetime(frame["week_ending"], errors="coerce")
    frame["year_number"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["stage"] = frame["short_desc"].str.extract(r"PCT ([A-Z]+)")[0]
    return frame[
        frame["progress_pct"].notna()
        & frame["date"].notna()
        & frame["stage"].notna()
    ].copy()


def _progress_scope(
    progress: pd.DataFrame,
    crop: str,
    window: str,
    cutoff_year: int,
    canola_all_years: bool = False,
) -> pd.DataFrame:
    if crop == "Wheat":
        prefix = (
            "WHEAT, WINTER"
            if window == "winter"
            else "WHEAT, SPRING, (EXCL DURUM)"
        )
        scope = progress[progress["short_desc"].str.startswith(prefix)].copy()
    else:
        scope = progress[progress["commodity_desc"].eq(crop.upper())].copy()
    if not (crop == "Canola" and canola_all_years):
        scope = scope[scope["year_number"] <= cutoff_year]
    return scope


def fifty_percent_dates(scope: pd.DataFrame) -> pd.DataFrame:
    reached = scope[scope["progress_pct"] >= 50].copy()
    keys = ["state_name", "year_number", "stage"]
    return (
        reached.sort_values("date", kind="stable")
        .groupby(keys, as_index=False)
        .first()[keys + ["date"]]
    )


def _median_month_day(dates: pd.Series) -> tuple[int, int]:
    doy = dates.dt.dayofyear.astype(int)
    median_doy = int(np.rint(np.median(doy)))
    stamp = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=median_doy - 1)
    return int(stamp.month), int(stamp.day)


def _state_calendar_rows(
    panel_keys: pd.DataFrame,
    progress: pd.DataFrame,
    cutoff_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for crop, region, window in panel_keys.itertuples(index=False):
        stages = BOUNDARY_STAGES[crop]
        scope = _progress_scope(
            progress,
            crop,
            window,
            cutoff_year,
            canola_all_years=crop == "Canola",
        )
        dates = fifty_percent_dates(scope)
        state = dates[dates["state_name"].eq(region.upper())]
        complete = state.pivot_table(
            index="year_number",
            columns="stage",
            values="date",
            aggfunc="first",
        )
        source_scope = "state"
        if not set(stages).issubset(complete.columns):
            complete = pd.DataFrame()
        else:
            complete = complete.dropna(subset=list(stages))
        if len(complete) < 3:
            complete = dates.pivot_table(
                index=["state_name", "year_number"],
                columns="stage",
                values="date",
                aggfunc="first",
            )
            if not set(stages).issubset(complete.columns):
                complete = pd.DataFrame()
            else:
                complete = complete.dropna(subset=list(stages))
            source_scope = "crop-season fallback"
        if len(complete) < 3:
            frozen_scope = _progress_scope(
                progress,
                crop,
                window,
                int(progress["year_number"].max()),
                canola_all_years=True,
            )
            frozen_dates = fifty_percent_dates(frozen_scope)
            complete = frozen_dates[
                frozen_dates["state_name"].eq(region.upper())
            ].pivot_table(
                index="year_number",
                columns="stage",
                values="date",
                aggfunc="first",
            )
            if set(stages).issubset(complete.columns):
                complete = complete.dropna(subset=list(stages))
            else:
                complete = pd.DataFrame()
            source_scope = "frozen external state fallback"
        if len(complete) < 3:
            complete = frozen_dates.pivot_table(
                index=["state_name", "year_number"],
                columns="stage",
                values="date",
                aggfunc="first",
            )
            if set(stages).issubset(complete.columns):
                complete = complete.dropna(subset=list(stages))
            else:
                complete = pd.DataFrame()
            source_scope = "frozen external crop-season fallback"
        if len(complete) < 3:
            raise AssertionError(
                f"Fewer than three calendar boundary pairs for "
                f"{crop}, {region}, {window}, cutoff {cutoff_year}"
            )
        b1 = _median_month_day(pd.to_datetime(complete[stages[0]]))
        b2 = _median_month_day(pd.to_datetime(complete[stages[1]]))
        b1_stamp = pd.Timestamp(2001, *b1)
        b2_stamp = pd.Timestamp(2001, *b2)
        if b1_stamp >= b2_stamp:
            raise AssertionError(f"Unordered calendar boundaries for {crop}, {region}")
        rows.append(
            {
                "crop": crop,
                "region": region,
                "window": window,
                "boundary1_stage": stages[0],
                "boundary1_month": b1[0],
                "boundary1_day": b1[1],
                "boundary2_stage": stages[1],
                "boundary2_month": b2[0],
                "boundary2_day": b2[1],
                "source_scope": source_scope,
                "source_complete_years": len(complete),
            }
        )
        coverage.append(
            {
                "cutoff_year": cutoff_year,
                "crop": crop,
                "region": region,
                "window": window,
                "calendar_status": "PASS",
                "calendar_source_scope": source_scope,
                "calendar_complete_years": len(complete),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def _daily_gdd(daily: pd.DataFrame, base: float) -> pd.Series:
    mean_temperature = (
        daily["T2M_MAX"].astype(float) + daily["T2M_MIN"].astype(float)
    ) / 2
    return pd.Series(np.maximum(mean_temperature - base, 0.0), index=daily.index)


def _gdd_rows(
    panel_keys: pd.DataFrame,
    progress: pd.DataFrame,
    daily_by_region: dict[str, pd.DataFrame],
    cutoff_year: int,
    base_by_crop: dict[str, float],
    canola_fallback: tuple[float, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for crop, window in (
        panel_keys[["crop", "window"]].drop_duplicates().itertuples(index=False)
    ):
        stages = BOUNDARY_STAGES[crop]
        if crop == "Canola":
            rows.append(
                {
                    "crop": crop,
                    "window": window,
                    "boundary1_stage": stages[0],
                    "boundary1_gdd": canola_fallback[0],
                    "boundary2_stage": stages[1],
                    "boundary2_gdd": canola_fallback[1],
                    "base_celsius": base_by_crop[crop],
                    "source_scope": "fixed external extension fallback",
                    "source_complete_stage_dates": 0,
                }
            )
            continue
        scope = _progress_scope(progress, crop, window, cutoff_year)
        dates = fifty_percent_dates(scope)
        complete = dates.pivot_table(
            index=["state_name", "year_number"],
            columns="stage",
            values="date",
            aggfunc="first",
        )
        if set(stages).issubset(complete.columns):
            complete = complete.dropna(subset=list(stages))
        else:
            complete = pd.DataFrame()
        values: list[tuple[float, float]] = []
        for (state_name, year_number), record in complete.iterrows():
            region = str(state_name).title()
            if region not in daily_by_region:
                continue
            year = int(year_number)
            start, end = season_bounds(year, window)
            daily = daily_by_region[region]
            season = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
            if season.empty:
                continue
            cumulative = _daily_gdd(season, base_by_crop[crop]).cumsum()
            stage_values = []
            for stage in stages:
                stage_date = pd.Timestamp(record[stage])
                positions = season.index[season["date"] <= stage_date]
                if len(positions) == 0:
                    break
                stage_values.append(float(cumulative.loc[positions[-1]]))
            if len(stage_values) == 2 and 0 < stage_values[0] < stage_values[1]:
                values.append((stage_values[0], stage_values[1]))
        source_scope = "cutoff-safe crop-season progress"
        if len(values) < 3:
            frozen_scope = _progress_scope(
                progress,
                crop,
                window,
                int(progress["year_number"].max()),
                canola_all_years=True,
            )
            frozen_dates = fifty_percent_dates(frozen_scope)
            complete = frozen_dates.pivot_table(
                index=["state_name", "year_number"],
                columns="stage",
                values="date",
                aggfunc="first",
            )
            if set(stages).issubset(complete.columns):
                complete = complete.dropna(subset=list(stages))
            else:
                complete = pd.DataFrame()
            values = []
            for (state_name, year_number), record in complete.iterrows():
                region = str(state_name).title()
                if region not in daily_by_region:
                    continue
                year = int(year_number)
                start, end = season_bounds(year, window)
                daily = daily_by_region[region]
                season = daily[
                    (daily["date"] >= start) & (daily["date"] <= end)
                ].copy()
                if season.empty:
                    continue
                cumulative = _daily_gdd(season, base_by_crop[crop]).cumsum()
                stage_values = []
                for stage in stages:
                    stage_date = pd.Timestamp(record[stage])
                    positions = season.index[season["date"] <= stage_date]
                    if len(positions) == 0:
                        break
                    stage_values.append(float(cumulative.loc[positions[-1]]))
                if (
                    len(stage_values) == 2
                    and 0 < stage_values[0] < stage_values[1]
                ):
                    values.append((stage_values[0], stage_values[1]))
            source_scope = "frozen external crop-season fallback"
        if len(values) < 3:
            raise AssertionError(
                f"Fewer than three GDD boundary pairs for "
                f"{crop}, {window}, cutoff {cutoff_year}"
            )
        array = np.asarray(values)
        rows.append(
            {
                "crop": crop,
                "window": window,
                "boundary1_stage": stages[0],
                "boundary1_gdd": float(np.median(array[:, 0])),
                "boundary2_stage": stages[1],
                "boundary2_gdd": float(np.median(array[:, 1])),
                "base_celsius": base_by_crop[crop],
                "source_scope": source_scope,
                "source_complete_stage_dates": len(values),
            }
        )
    return pd.DataFrame(rows)


def derive_stage_definitions(
    root: Path,
    cutoff_year: int,
) -> StageDefinitions:
    protocol = json.loads(
        (root / "configs" / "extension_factorial_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    panel = pd.read_csv(root / protocol["data"]["processed_reference"])
    panel_keys = panel[["crop", "region", "window"]].drop_duplicates()
    progress = load_progress(root / protocol["data"]["crop_progress_snapshot"])
    daily = load_nasa_power_daily(root / protocol["data"]["daily_weather"])
    calendar, coverage = _state_calendar_rows(
        panel_keys,
        progress,
        cutoff_year,
    )
    base_by_crop = {
        key: float(value)
        for key, value in protocol["stage_source_rules"]["gdd_base_celsius"].items()
    }
    canola = tuple(
        float(value)
        for value in protocol["stage_source_rules"][
            "canola_fixed_external_fallback"
        ]["gdd_boundaries_celsius_days"]
    )
    gdd = _gdd_rows(
        panel_keys,
        progress,
        daily,
        cutoff_year,
        base_by_crop,
        canola,
    )
    return StageDefinitions(cutoff_year, calendar, gdd, coverage)


def _aggregate_stage(part: pd.DataFrame, suffix: str) -> dict[str, float]:
    if part.empty:
        return {f"{name}_{suffix}": 0.0 for name in STAGE_FEATURE_NAMES}
    rain = part["PRECTOTCORR"].astype(float)
    return {
        f"rain_sum_{suffix}": float(rain.sum()),
        f"heat_days_35_{suffix}": float((part["T2M_MAX"].astype(float) >= 35).sum()),
        f"frost_days_0_{suffix}": float((part["T2M_MIN"].astype(float) <= 0).sum()),
        f"radiation_sum_{suffix}": float(
            part["ALLSKY_SFC_SW_DWN"].astype(float).sum()
        ),
        f"max_3day_rain_{suffix}": float(rain.rolling(3, min_periods=1).sum().max()),
    }


def build_stage_frame(
    root: Path,
    definitions: StageDefinitions,
) -> pd.DataFrame:
    protocol = json.loads(
        (root / "configs" / "extension_factorial_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    panel = pd.read_csv(root / protocol["data"]["processed_reference"])
    daily_by_region = load_nasa_power_daily(root / protocol["data"]["daily_weather"])
    calendar_lookup = definitions.calendar.set_index(["crop", "region", "window"])
    gdd_lookup = definitions.gdd.set_index(["crop", "window"])
    rows: list[dict[str, object]] = []
    for row in panel.itertuples(index=False):
        start, end = season_bounds(int(row.year), row.window)
        daily = daily_by_region[row.region]
        season = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
        calendar = calendar_lookup.loc[(row.crop, row.region, row.window)]
        b1 = pd.Timestamp(int(row.year), int(calendar.boundary1_month), int(calendar.boundary1_day))
        b2 = pd.Timestamp(int(row.year), int(calendar.boundary2_month), int(calendar.boundary2_day))
        if row.window == "winter" and b1 < start:
            b1 = pd.Timestamp(int(row.year), int(calendar.boundary1_month), int(calendar.boundary1_day))
        cal_parts = (
            season[season["date"] < b1],
            season[(season["date"] >= b1) & (season["date"] < b2)],
            season[season["date"] >= b2],
        )
        gdd = gdd_lookup.loc[(row.crop, row.window)]
        cumulative = _daily_gdd(season, float(gdd.base_celsius)).cumsum()
        gdd_parts = (
            season[cumulative < float(gdd.boundary1_gdd)],
            season[
                (cumulative >= float(gdd.boundary1_gdd))
                & (cumulative < float(gdd.boundary2_gdd))
            ],
            season[cumulative >= float(gdd.boundary2_gdd)],
        )
        output: dict[str, object] = {
            "country": row.country,
            "region": row.region,
            "crop": row.crop,
            "year": int(row.year),
            "window": row.window,
        }
        for label, part in zip(("early", "mid", "late"), cal_parts):
            output.update(_aggregate_stage(part, f"cal_{label}"))
        for label, part in zip(("early", "mid", "late"), gdd_parts):
            output.update(_aggregate_stage(part, f"gdd_{label}"))
        rows.append(output)
    return pd.DataFrame(rows)
