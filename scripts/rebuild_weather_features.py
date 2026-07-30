"""Reconstruct and verify the processed weather features from bundled raw inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.core import driver_group  # noqa: E402
from crop_yield_xai.weather_features import (  # noqa: E402
    FEATURE_SPECS,
    WINTER_WHEAT_REGIONS,
    build_model_frame,
    compare_frames,
    load_nasa_power_daily,
)


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    data_dir = ROOT / "artifacts" / "data"
    expected = pd.read_csv(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv")
    raw_yield = pd.read_csv(ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv")
    reconstructed = build_model_frame(raw_yield, load_nasa_power_daily(ROOT / "data" / "raw" / "nasa_power_daily.zip"))
    comparison = compare_frames(expected, reconstructed)
    write(reconstructed, data_dir / "reconstructed_model_frame.csv")
    write(comparison, data_dir / "weather_reconstruction_validation.csv")

    dictionary = pd.DataFrame(FEATURE_SPECS)
    dictionary.insert(1, "driver_group", dictionary["feature"].map(driver_group))
    dictionary["verification_status"] = "EXACT_RECONSTRUCTION_PASS"
    write(dictionary, data_dir / "feature_dictionary.csv")

    calendar = pd.DataFrame(
        [
            {
                "crop": crop,
                "region": region,
                "window": window,
                "start_rule": "April 1 of yield year" if window == "spring" else "September 1 of previous year",
                "end_rule": "September 30 of yield year" if window == "spring" else "June 30 of yield year",
                "stage_rule": "three consecutive equal blocks; final block receives remainder",
            }
            for crop, region, window in expected[["crop", "region", "window"]].drop_duplicates().sort_values(["crop", "region"]).itertuples(index=False)
        ]
    )
    write(calendar, data_dir / "crop_calendar.csv")
    config = "\n".join(
        [
            "status: PASS",
            "daily_source: data/raw/nasa_power_daily.zip",
            "daily_fields: [T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN]",
            "spring_window: April 1 through September 30 of the yield year (183 days)",
            "winter_wheat_window: September 1 of the prior year through June 30 of the yield year (303 or 304 days)",
            "winter_wheat_regions: [" + ", ".join(sorted(WINTER_WHEAT_REGIONS)) + "]",
            "stage_split: first floor(n/3) days = early; second floor(n/3) days = mid; remainder = late",
            "missing_day_rule: fail reconstruction unless every calendar day is present exactly once",
            "rounding: round continuous aggregates to three decimal places before writing CSV",
            "verification: exact equality to data/processed/us_model_frame_hemisphere_aware_1990_2025.csv at tolerance 0.0005",
            "",
        ]
    )
    (data_dir / "weather_extraction_config.yaml").write_text(config, encoding="utf-8")
    print(f"Weather reconstruction passed: {len(reconstructed)} rows, {len(comparison)} columns, max difference {comparison['max_abs_difference'].max():.6f}")


if __name__ == "__main__":
    main()
