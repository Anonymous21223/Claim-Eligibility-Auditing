"""Resumably acquire NASA POWER daily weather for the locked V2 county population."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "v2_county" / "interim" / "county_yield_v2_canonical.csv"
CENTROIDS = ROOT / "data" / "v2_county" / "interim" / "county_centroids.csv"
RAW = ROOT / "data" / "v2_county" / "raw" / "nasa_power"
MANIFESTS = ROOT / "data" / "v2_county" / "manifests"
REPORT = ROOT / "reports" / "v2" / "nasa_power_download.json"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_valid_cached(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("properties", {}).get("parameter", {})
        return all(parameter in values and len(values[parameter]) == 9497 for parameter in PARAMETERS.split(","))
    except (OSError, json.JSONDecodeError):
        return False


def download(row: tuple[str, float, float]) -> tuple[str, str, str | None]:
    county_fips, lat, lon = row
    target = RAW / f"{county_fips}.json"
    if target.exists() and load_valid_cached(target):
        return county_fips, "cached", digest(target)
    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": f"{lon:.6f}",
        "latitude": f"{lat:.6f}",
        "start": "20000101",
        "end": "20251231",
        "format": "JSON",
    }
    for attempt in range(5):
        try:
            response = requests.get(URL, params=params, timeout=180)
            if response.status_code == 429 and attempt < 4:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            values = payload.get("properties", {}).get("parameter", {})
            if not all(parameter in values and len(values[parameter]) == 9497 for parameter in PARAMETERS.split(",")):
                return county_fips, "invalid_payload", None
            tmp = target.with_suffix(".json.tmp")
            tmp.write_bytes(response.content)
            tmp.replace(target)
            return county_fips, "downloaded", digest(target)
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == 4:
                return county_fips, "request_failed", None
            time.sleep(2**attempt)
    return county_fips, "request_failed", None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("BLOCKED_WORKERS_OUT_OF_RANGE")
    panel = pd.read_csv(PANEL, dtype={"county_fips": str})
    centroids = pd.read_csv(CENTROIDS, dtype={"county_fips": str})
    counties = sorted(set(panel["county_fips"]))
    subset = centroids[centroids["county_fips"].isin(counties)].sort_values("county_fips")
    if len(subset) != len(counties):
        raise SystemExit("BLOCKED_CENTROID_COVERAGE")
    if args.limit is not None:
        subset = subset.head(args.limit)
    RAW.mkdir(parents=True, exist_ok=True)
    rows = [(str(row.county_fips), float(row.county_lat), float(row.county_lon)) for row in subset.itertuples(index=False)]
    results: list[tuple[str, str, str | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(download, rows):
            results.append(result)
    status_counts = pd.Series([status for _, status, _ in results]).value_counts().to_dict()
    failures = sorted(county for county, status, _ in results if status in {"invalid_payload", "request_failed"})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures and args.limit is None else ("PASS_PILOT" if not failures else "BLOCKED_DOWNLOAD_FAILURE"),
        "source": "NASA POWER daily point API",
        "endpoint": "temporal/daily/point",
        "parameters": PARAMETERS.split(","),
        "community": "AG",
        "date_range": ["2000-01-01", "2025-12-31"],
        "population_count": len(counties),
        "requested_counties": len(rows),
        "workers": args.workers,
        "status_counts": status_counts,
        "failed_counties": failures,
        "raw_file_hashes": {f"data/v2_county/raw/nasa_power/{county}.json": hash_value for county, status, hash_value in results if hash_value},
        "panel_sha256": digest(PANEL),
        "centroid_sha256": digest(CENTROIDS),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.limit is not None else "full"
    manifest_path = MANIFESTS / f"nasa_power_{suffix}_download.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in manifest.items() if key != "raw_file_hashes"}
    report["manifest_path"] = manifest_path.relative_to(ROOT).as_posix()
    report["raw_file_hash_count"] = len(manifest["raw_file_hashes"])
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ["status", "requested_counties", "status_counts", "failed_counties"]}))
    if failures:
        raise SystemExit("BLOCKED_NASA_POWER_DOWNLOAD")


if __name__ == "__main__":
    main()
