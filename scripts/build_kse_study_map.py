"""Build the code-generated KSE crop-panel study-coverage map."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv"
BOUNDARY = ROOT / "data" / "reference" / "census_tigerweb_2025_states" / "us_states_contiguous.geojson"
OUT_DIR = ROOT / "paper_versions" / "v5_9_kse_model_first_6pages" / "source" / "figures"
REPORT = ROOT / "paper_versions" / "v5_9_kse_model_first_6pages" / "source" / "generated" / "study_map_provenance_kse.json"

EXPECTED_STATES = {
    "Colorado",
    "Illinois",
    "Iowa",
    "Kansas",
    "Minnesota",
    "Montana",
    "Nebraska",
    "North Dakota",
    "Oklahoma",
    "South Dakota",
    "Texas",
    "Washington",
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    frame = pd.read_csv(FRAME, usecols=["region", "year"])
    states = set(frame["region"].dropna().unique())
    if len(frame) != 1257 or states != EXPECTED_STATES:
        raise AssertionError({"rows": len(frame), "states": sorted(states)})
    if (int(frame["year"].min()), int(frame["year"].max())) != (1990, 2025):
        raise AssertionError("Unexpected panel years")

    boundaries = gpd.read_file(BOUNDARY)
    if len(boundaries) != 49 or not EXPECTED_STATES.issubset(set(boundaries["NAME"])):
        raise AssertionError("Boundary file does not contain the expected contiguous states")
    boundaries = boundaries.to_crs("EPSG:5070")
    highlighted = boundaries[boundaries["NAME"].isin(EXPECTED_STATES)].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.50, 2.48), facecolor="white")
    boundaries.plot(ax=ax, facecolor="#F2F4F5", edgecolor="#7A838A", linewidth=0.35)
    highlighted.plot(ax=ax, facecolor="#2A9D8F", edgecolor="#1F2933", linewidth=0.55)

    for _, row in highlighted.iterrows():
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            row["STUSAB"],
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#0B2430",
        )

    ax.text(
        0.025,
        0.035,
        "12 crop-panel states\n1,257 crop-state-year rows\n1990-2025",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        linespacing=1.18,
        bbox={
            "boxstyle": "round,pad=0.32,rounding_size=0.15",
            "facecolor": "white",
            "edgecolor": "#56616A",
            "linewidth": 0.65,
        },
    )
    ax.set_axis_off()
    ax.set_xlim(-2.45e6, 2.35e6)
    ax.set_ylim(2.20e5, 3.23e6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "study_coverage_map_kse.pdf"
    png_path = OUT_DIR / "study_coverage_map_kse.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        metadata={
            "Creator": "build_kse_study_map.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    report = {
        "status": "PASS",
        "purpose": "descriptive crop-panel study coverage; not an effects or inference map",
        "rows": len(frame),
        "years": [1990, 2025],
        "states": sorted(states),
        "input_sha256": {str(FRAME.relative_to(ROOT)): digest(FRAME), str(BOUNDARY.relative_to(ROOT)): digest(BOUNDARY)},
        "boundary_source": {
            "provider": "U.S. Census Bureau TIGERweb",
            "vintage": "January 1, 2025",
            "metadata_url": "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/10",
            "accessed": "2026-07-31",
            "copyright_text": "Source: U.S. Census Bureau",
        },
        "versions": {"python": platform.python_version(), "geopandas": gpd.__version__, "matplotlib": matplotlib.__version__, "pandas": pd.__version__},
        "outputs_sha256": {str(pdf_path.relative_to(ROOT)): digest(pdf_path), str(png_path.relative_to(ROOT)): digest(png_path)},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
