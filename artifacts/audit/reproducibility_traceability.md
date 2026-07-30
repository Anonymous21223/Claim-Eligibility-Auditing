# Reproducibility Traceability

Raw yield SHA-256: 9bdb5c39fa55e3951df5da41a738742f51c684776cba09fcf6c1a6e6b366d47b
NASA POWER archive SHA-256: dfc29822b4c7dc27f3dc20f2af7402c3fdaa522d3c7b7ef4e271d9f1a48ad782
Processed frame SHA-256: 6c0920c8f0a0d2447b8f752f5f69efd650045d57013de4a9dda100d23d28a81b
Raw-weather reconstruction: PASS. `scripts/rebuild_weather_features.py` reconstructs all 1,257 rows and all 59 columns exactly; the per-column comparison is `artifacts/data/weather_reconstruction_validation.csv`.
The feature dictionary records NASA fields, formulas, windows, thresholds, rounding, and calendar rules used by the reconstruction.
