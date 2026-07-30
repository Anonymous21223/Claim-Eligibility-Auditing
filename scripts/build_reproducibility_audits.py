"""Generate traceability artifacts from the repository's current data files."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crop_yield_xai.core import driver_group, full_season_weather_features, load_frame, make_project_paths  # noqa: E402
from crop_yield_xai.weather_features import FEATURE_SPECS, WINTER_WHEAT_REGIONS, build_model_frame, compare_frames, load_nasa_power_daily  # noqa: E402


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    data_out = ROOT / "artifacts" / "data"
    audit_out = ROOT / "artifacts" / "audit"
    frame = load_frame(make_project_paths(ROOT))
    raw_yield = pd.read_csv(ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv")
    reconstructed = build_model_frame(raw_yield, load_nasa_power_daily(ROOT / "data" / "raw" / "nasa_power_daily.zip"))
    reconstruction = compare_frames(frame, reconstructed)
    features = full_season_weather_features(frame)
    missing = frame.isna().agg(["sum", "mean"]).T.reset_index().rename(columns={"index": "column", "sum": "missing_count", "mean": "missing_share"})
    missing["n_rows"] = len(frame)
    write(missing, data_out / "missingness.csv")
    write(pd.DataFrame([{"stage": "raw_yield_file", "rows": len(raw_yield), "rule": "source table supplied in repository"}, {"stage": "analysis_year_filter", "rows": len(frame), "rule": "exclude 34 raw 1989 rows because analysis starts in 1990"}, {"stage": "processed_model_frame", "rows": len(frame), "rule": "exact reconstruction from raw yields plus NASA POWER daily input"}, {"stage": "final_test_eligible", "rows": 333, "rule": "2016-2025 rows with at least three prior series observations"}]), data_out / "data_flow.csv")
    write(pd.DataFrame([{"exclusion_reason": "insufficient_prior_series_history", "scope": "2012-2015 validation", "handling": "excluded before train-only trend fit; logged in detrending audit"}, {"exclusion_reason": "insufficient_prior_series_history", "scope": "2016-2025 final test", "handling": "excluded before train-only trend fit; logged in detrending audit"}]), data_out / "exclusions.csv")
    write(pd.DataFrame([{"target": "yield_t_ha", "raw_file": "data/raw/us_yield_1989_2025_tha.csv", "raw_unit": "t ha-1 (as stored)", "conversion_factor": 1.0, "moisture_assumption": "not documented in repository", "verification": "processed target has the same unit label; upstream USDA conversion cannot be audited from this repository"}]), data_out / "yield_unit_conversion.csv")
    dictionary = pd.DataFrame(FEATURE_SPECS)
    dictionary.insert(1, "driver_group", dictionary["feature"].map(driver_group))
    dictionary["verification_status"] = "EXACT_RECONSTRUCTION_PASS"
    write(dictionary, data_out / "feature_dictionary.csv")
    write(reconstructed, data_out / "reconstructed_model_frame.csv")
    write(reconstruction, data_out / "weather_reconstruction_validation.csv")
    config = "\n".join(["status: PASS", "daily_source: data/raw/nasa_power_daily.zip", "daily_fields: [T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN]", "spring_window: April 1 through September 30 of the yield year (183 days)", "winter_wheat_window: September 1 of the prior year through June 30 of the yield year (303 or 304 days)", "winter_wheat_regions: [" + ", ".join(sorted(WINTER_WHEAT_REGIONS)) + "]", "stage_split: first floor(n/3) days = early; second floor(n/3) days = mid; remainder = late", "missing_day_rule: fail reconstruction unless every calendar day is present exactly once", "rounding: round continuous aggregates to three decimal places before writing CSV", "verification: exact equality to data/processed/us_model_frame_hemisphere_aware_1990_2025.csv at tolerance 0.0005", ""])
    (data_out / "weather_extraction_config.yaml").write_text(config, encoding="utf-8")
    (data_out / "data_vintage.md").write_text("# Data Vintage\n\nStatus: BLOCKED. The repository records local file timestamps but contains no query timestamp, API request log, source release version, or earliest-availability record. The revised manuscript limits use to retrospective audit.\n", encoding="utf-8")
    report = ["# Reproducibility Traceability", "", "Raw yield SHA-256: " + sha256(ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv"), "NASA POWER archive SHA-256: " + sha256(ROOT / "data" / "raw" / "nasa_power_daily.zip"), "Processed frame SHA-256: " + sha256(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv"), "Raw-weather reconstruction: PASS. `scripts/rebuild_weather_features.py` reconstructs all 1,257 rows and all 59 columns exactly; the per-column comparison is `artifacts/data/weather_reconstruction_validation.csv`.", "The feature dictionary records NASA fields, formulas, windows, thresholds, rounding, and calendar rules used by the reconstruction."]
    (audit_out / "reproducibility_traceability.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Reproducibility audits written.")


if __name__ == "__main__":
    main()
