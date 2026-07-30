# Data Manifest

| Dataset | Source identifier | Local path | Redistribution status | Reproduction use | SHA-256 |
|---|---|---|---|---|---|
| USDA NASS yield snapshot | USDA NASS public data; see citation request | `data/raw/us_yield_1989_2025_tha.csv` | Public-domain information may be reproduced with acknowledgment | Input to the model frame | Generated during preflight |
| NASA POWER daily weather snapshot | NASA POWER daily service | `data/raw/nasa_power_daily.zip` | Attribution required; NASA requests redistribution notification | Input to weather reconstruction | Generated during preflight |
| Processed model frame | Deterministic reconstruction from the two inputs | `data/processed/us_model_frame_hemisphere_aware_1990_2025.csv` | Derived research data | Exact reconstruction target | Generated during preflight |

The project uses 1,257 crop-state-year rows for 1990--2025 across Barley,
Canola, Oats, and Wheat. The reconstruction command is:

```powershell
python scripts/rebuild_weather_features.py
```

It checks the reconstructed frame against the processed frame at the documented
tolerance and writes its evidence under `artifacts/data/`.

Before any public distribution, retain the cited USDA acknowledgment and NASA
POWER attribution/notification requested by their documentation. Source access
dates and file checksums are recorded by `scripts/repository_inventory.py`.
