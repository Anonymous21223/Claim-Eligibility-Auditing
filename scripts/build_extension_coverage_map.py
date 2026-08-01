"""Build the V6.1 locked crop-state coverage map from stored OOF rows."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "artifacts" / "extensions" / "v1" / "core"
PREDICTIONS = CORE / "outer_mean_predictions.parquet"
LOCK = CORE / "winner_lock.json"
BOUNDARY = (
    ROOT
    / "data"
    / "reference"
    / "census_tigerweb_2025_states"
    / "us_states_contiguous.geojson"
)
SOURCE = ROOT / "paper_versions" / "v6_1_extension_factorial_anonymous" / "source"
OUT_DIR = SOURCE / "figures"
PROVENANCE = SOURCE / "generated" / "coverage_map_provenance.json"
STATE_TABLE = SOURCE / "generated" / "coverage_map_state_counts.csv"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    winner = lock["winner"]
    predictions = pd.read_parquet(PREDICTIONS)
    rows = predictions[
        (predictions["arm_id"] == winner["arm_id"])
        & (predictions["model_id"] == winner["model_id"])
        & (predictions["feature_family"] == winner["feature_family"])
    ].copy()
    row_keys = ["country", "region", "crop", "year", "window", "outer_fold", "row_hash"]
    if len(rows) != 615 or rows.duplicated(row_keys).any() or rows["row_hash"].nunique() != 615:
        raise AssertionError("Locked winner rows are incomplete or duplicated")

    counts = (
        rows.groupby("region", as_index=False)
        .agg(
            crop_count=("crop", "nunique"),
            locked_rows=("row_hash", "nunique"),
            crops=("crop", lambda values: ";".join(sorted(set(values)))),
        )
        .sort_values("region")
    )
    expected_crops = {"Barley", "Canola", "Oats", "Wheat"}
    if set(rows["crop"]) != expected_crops or not counts["crop_count"].between(1, 4).all():
        raise AssertionError("Unexpected crop coverage")

    boundaries = gpd.read_file(BOUNDARY).to_crs("EPSG:5070")
    if not set(counts["region"]).issubset(set(boundaries["NAME"])):
        raise AssertionError("A locked state is absent from the boundary file")
    mapped = boundaries.merge(counts, left_on="NAME", right_on="region", how="left")
    mapped["crop_count"] = mapped["crop_count"].fillna(0).astype(int)

    colors = ["#ECEFF1", "#D7E9F5", "#91C4DD", "#3D8CB8", "#174F78"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.50, 2.42), facecolor="white")
    mapped.plot(
        ax=ax,
        column="crop_count",
        cmap=cmap,
        norm=norm,
        edgecolor="#6B747B",
        linewidth=0.35,
    )
    eligible = mapped[mapped["crop_count"] > 0]
    for _, state in eligible.iterrows():
        point = state.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            state["STUSAB"],
            ha="center",
            va="center",
            fontsize=6.1,
            fontweight="bold",
            color="#102A3A" if state["crop_count"] < 4 else "white",
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=6.3,
            markerfacecolor=colors[value],
            markeredgecolor="#6B747B",
            markeredgewidth=0.4,
            label=str(value),
        )
        for value in range(5)
    ]
    ax.legend(
        handles=handles,
        title="Eligible crops",
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.015),
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0,
        fontsize=6.5,
        title_fontsize=6.7,
    )
    ax.set_axis_off()
    ax.set_xlim(-2.45e6, 2.35e6)
    ax.set_ylim(2.20e5, 3.23e6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.075)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "locked_crop_state_coverage.pdf"
    png_path = OUT_DIR / "locked_crop_state_coverage.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        metadata={"Creator": "anonymous artifact script", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    counts.to_csv(STATE_TABLE, index=False)

    report = {
        "status": "PASS",
        "purpose": "descriptive coverage of eligible locked winner rows",
        "winner": winner,
        "locked_rows": int(len(rows)),
        "states": int(len(counts)),
        "crops": sorted(expected_crops),
        "input_sha256": {
            str(PREDICTIONS.relative_to(ROOT)): digest(PREDICTIONS),
            str(LOCK.relative_to(ROOT)): digest(LOCK),
            str(BOUNDARY.relative_to(ROOT)): digest(BOUNDARY),
        },
        "state_counts_sha256": digest(STATE_TABLE),
        "output_sha256": {pdf_path.name: digest(pdf_path), png_path.name: digest(png_path)},
        "boundary_source": "U.S. Census Bureau TIGERweb, January 1 2025 state layer",
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "geopandas": gpd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    PROVENANCE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
