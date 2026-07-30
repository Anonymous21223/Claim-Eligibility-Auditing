# Dataset V2 Data Manifest

Dataset V2 is a separate County-Level Agricultural Risk Panel. Dataset V1 remains frozen and is not modified by this pipeline.

## Current sources

| Source | Scope | Access | Redistribution | Reconstruction |
|---|---|---|---|---|
| USDA NASS Quick Stats | County annual crop yield | `NASS_API_KEY` in runtime environment only | Raw API payloads retained as research artifacts; verify terms before public redistribution | `python scripts/download_nass_v2_county.py` then `python scripts/build_v2_county_yield_panel.py` |
| U.S. Census TIGERweb | Static county interior-point coordinates | Public REST API | Verify Census terms before release | `python scripts/download_v2_county_centroids.py` |
| NASA POWER | County-centroid daily meteorology | Public API | Verify NASA POWER terms before release | Weather downloader pending county-centroid manifest |

All raw NASS requests have a secret-redacted manifest and an SHA-256 checksum. The canonical yield panel is created only from the explicit pre-model rule in `configs/experiments/county_v2_panel_spec.json`.
