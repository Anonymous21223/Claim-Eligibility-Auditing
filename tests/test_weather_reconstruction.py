from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.weather_features import build_model_frame, compare_frames, load_nasa_power_daily  # noqa: E402


class WeatherReconstructionTests(unittest.TestCase):
    def test_raw_weather_reconstructs_the_processed_frame_exactly(self) -> None:
        expected = pd.read_csv(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv")
        raw_yield = pd.read_csv(ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv")
        reconstructed = build_model_frame(raw_yield, load_nasa_power_daily(ROOT / "data" / "raw" / "nasa_power_daily.zip"))
        report = compare_frames(expected, reconstructed)
        self.assertEqual(len(reconstructed), 1257)
        self.assertEqual(int(report["mismatch_count"].sum()), 0)
        self.assertEqual(float(report["max_abs_difference"].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
