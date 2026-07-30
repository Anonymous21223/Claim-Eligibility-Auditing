"""Acquire static county interior points from the official Census TIGERweb service."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "v2_county" / "interim" / "county_yield_v2_canonical.csv"
RAW = ROOT / "data" / "v2_county" / "raw" / "census_tigerweb"
MANIFESTS = ROOT / "data" / "v2_county" / "manifests"
OUTPUT = ROOT / "data" / "v2_county" / "interim" / "county_centroids.csv"
REPORT = ROOT / "reports" / "v2" / "county_centroid_coverage.json"
URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query"


def request_state(state_fips: str) -> requests.Response:
    params = {
        "where": f"STATE='{state_fips}'",
        "outFields": "GEOID,INTPTLAT,INTPTLON,NAME",
        "returnGeometry": "false",
        "f": "json",
    }
    for attempt in range(4):
        try:
            response = requests.get(URL, params=params, timeout=60)
            if response.status_code == 429 and attempt < 3:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    panel = pd.read_csv(PANEL, dtype={"county_fips": str, "state_fips": str})
    required = set(panel["county_fips"].drop_duplicates())
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    raw_paths: list[Path] = []
    for state_fips in sorted(panel["state_fips"].drop_duplicates()):
        response = request_state(state_fips)
        payload = response.json()
        if "error" in payload:
            raise SystemExit(f"BLOCKED_TIGERWEB_ERROR_STATE_{state_fips}")
        target = RAW / f"county_centroids_state_{state_fips}.json"
        target.write_bytes(response.content)
        raw_paths.append(target)
        for feature in payload.get("features", []):
            attrs = feature.get("attributes", {})
            records.append(
                {
                    "county_fips": str(attrs.get("GEOID", "")).zfill(5),
                    "county_lat": attrs.get("INTPTLAT"),
                    "county_lon": attrs.get("INTPTLON"),
                    "census_county_name": attrs.get("NAME"),
                    "source_state_fips": state_fips,
                }
            )
        time.sleep(0.1)
    centroids = pd.DataFrame(records)
    centroids["county_lat"] = pd.to_numeric(centroids["county_lat"], errors="coerce")
    centroids["county_lon"] = pd.to_numeric(centroids["county_lon"], errors="coerce")
    centroids = centroids[centroids["county_fips"].isin(required)].drop_duplicates("county_fips")
    missing = sorted(required - set(centroids["county_fips"]))
    invalid = centroids[centroids[["county_lat", "county_lon"]].isna().any(axis=1)]
    if missing or not invalid.empty:
        raise SystemExit(f"BLOCKED_CENTROID_COVERAGE_missing={len(missing)}_invalid={len(invalid)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    centroids.sort_values("county_fips").to_csv(OUTPUT, index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "U.S. Census Bureau TIGERweb State_County MapServer layer 1",
        "endpoint": "TIGERweb/State_County/MapServer/1/query",
        "spatial_reference": "Static current county interior points; not a historical boundary reconstruction.",
        "panel_sha256": digest(PANEL),
        "raw_files": {path.relative_to(ROOT).as_posix(): digest(path) for path in raw_paths},
        "centroid_path": OUTPUT.relative_to(ROOT).as_posix(),
        "centroid_sha256": digest(OUTPUT),
        "requested_counties": len(required),
        "matched_counties": len(centroids),
        "missing_counties": missing,
    }
    manifest_path = MANIFESTS / "county_centroids_tigerweb.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"status": "PASS", **manifest, "manifest_path": manifest_path.relative_to(ROOT).as_posix()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "requested_counties": len(required), "matched_counties": len(centroids), "centroid_path": manifest["centroid_path"]}))


if __name__ == "__main__":
    main()
