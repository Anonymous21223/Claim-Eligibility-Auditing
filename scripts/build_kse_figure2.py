"""Build the KSE v5.8 evidence-board figure from the locked SHAP artifact."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "xai" / "local_case_decomposition.csv"
OUT_DIR = (
    ROOT
    / "paper_versions"
    / "v5_8_kse_revision"
    / "source"
    / "figures"
)
REPORT = (
    ROOT
    / "paper_versions"
    / "v5_8_kse_revision"
    / "source"
    / "generated"
    / "figure2_provenance_kse.json"
)
ROW_ID = "Barley|Colorado|2016|spring"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_case() -> dict:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["row_id"] == ROW_ID]
    if not rows:
        raise RuntimeError(f"{ROW_ID} not found in {SOURCE}")
    values = {row["driver_group"]: float(row["signed_group_shap"]) for row in rows}
    prediction = float(rows[0]["predicted_residual"])
    observed = float(rows[0]["observed_residual"])
    contributions = [
        ("Heat", values["heat"]),
        ("Drought", values["drought"]),
        ("Frost/cold", values["frost_cold"]),
        ("Excess rain", values["excess_rain"]),
        ("Radiation", values["radiation"]),
    ]
    contributions.append(
        ("Other/remainder", prediction - sum(value for _, value in contributions))
    )
    if not np.isclose(sum(value for _, value in contributions), prediction, atol=1e-10):
        raise AssertionError("Grouped terms do not reconstruct the prediction")
    if not (observed < -0.5 and prediction > 0.2):
        raise AssertionError((observed, prediction))
    return {
        "observed": observed,
        "prediction": prediction,
        "base_value": float(rows[0]["base_value"]),
        "contributions": contributions,
    }


def build(case: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.18, 3.18), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.30, 1.58], wspace=0.20)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    observed = case["observed"]
    prediction = case["prediction"]
    accent = "#315a7d"
    dark = "#25313c"
    light = "#d9dee3"

    ax_a.set_title("A. Locked outcome mismatch", loc="left", fontweight="bold", pad=6)
    ax_a.hlines(0, observed, prediction, color=dark, lw=2.1, zorder=2)
    ax_a.scatter(
        [observed],
        [0],
        s=78,
        marker="D",
        facecolors="white",
        edgecolors=dark,
        linewidths=1.2,
        zorder=4,
    )
    ax_a.scatter([prediction], [0], s=78, marker="s", color=accent, zorder=4)
    ax_a.vlines(
        [observed, prediction],
        0.035,
        0.50,
        colors=dark,
        linestyles=(0, (1, 2)),
        lw=0.9,
        zorder=1,
    )
    ax_a.text(
        observed,
        0.56,
        "observed\n$-0.510$",
        ha="center",
        va="bottom",
        fontweight="bold",
        linespacing=1.0,
    )
    ax_a.text(
        prediction,
        0.56,
        "prediction\n$+0.209$",
        ha="center",
        va="bottom",
        fontweight="bold",
        linespacing=1.0,
    )
    ax_a.text(
        -0.15,
        0.22,
        "wrong direction\nand magnitude",
        ha="center",
        va="center",
        color=dark,
        linespacing=1.0,
    )
    ax_a.add_patch(
        FancyArrowPatch(
            (observed + 0.05, 0.13),
            (prediction - 0.05, 0.13),
            arrowstyle="]-[",
            mutation_scale=12,
            color=dark,
            lw=0.9,
        )
    )
    ax_a.set_xlim(-0.6, 0.4)
    ax_a.set_ylim(-0.46, 0.84)
    ax_a.set_xticks(np.arange(-0.6, 0.41, 0.2))
    ax_a.set_yticks([])
    ax_a.set_xlabel(r"raw residual, t\,ha$^{-1}$", labelpad=5)
    ax_a.spines[["left", "top", "right"]].set_visible(False)

    contributions = sorted(
        case["contributions"], key=lambda item: abs(item[1]), reverse=True
    )
    labels = [label for label, _ in contributions]
    values = np.array([value for _, value in contributions])
    positions = np.arange(len(labels))
    colors = [accent if value >= 0 else light for value in values]
    bars = ax_b.barh(
        positions,
        values,
        height=0.72,
        color=colors,
        edgecolor=dark,
        linewidth=0.5,
    )
    for bar, value in zip(bars, values):
        if value < 0:
            bar.set_hatch("///")
        ax_b.text(
            value + 0.004 if value >= 0 else 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax_b.vlines(0, -0.42, len(labels) - 0.45, color=dark, lw=1.1)
    ax_b.set_yticks(positions, labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-0.030, 0.098)
    ax_b.set_xticks(
        [-0.025, 0.000, 0.025, 0.050, 0.075],
        ["-.025", "0", ".025", ".050", ".075"],
    )
    ax_b.set_xlabel(r"SHAP contribution, t\,ha$^{-1}$", labelpad=5)
    ax_b.set_title(
        "B. Why the model predicted $+0.209$",
        loc="left",
        fontweight="bold",
        pad=15,
    )
    ax_b.text(
        0.50,
        0.985,
        "Terms reconstruct the fitted prediction.",
        ha="center",
        va="top",
        transform=ax_b.transAxes,
        color=dark,
    )
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines[["top", "right", "left"]].set_visible(False)

    ax_c.set_axis_off()
    ax_c.set_title("C. Permitted interpretation", loc="left", fontweight="bold", pad=6)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.add_patch(
        FancyBboxPatch(
            (0.03, 0.62),
            0.94,
            0.25,
            boxstyle="round,pad=0.014,rounding_size=0.02",
            facecolor="white",
            edgecolor=dark,
            lw=1.0,
        )
    )
    ax_c.text(0.07, 0.81, "Ungated reading", fontweight="bold", va="center")
    ax_c.text(0.07, 0.70, "coherent model\nexplanation", va="center", linespacing=1.0)
    ax_c.annotate(
        "",
        xy=(0.58, 0.70),
        xytext=(0.43, 0.70),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=dark),
    )
    ax_c.text(
        0.64,
        0.70,
        "weather\nclaim",
        va="center",
        fontweight="bold",
        linespacing=1.0,
    )
    ax_c.text(0.91, 0.70, r"$\times$", fontsize=15, color=accent, va="center")
    ax_c.text(
        0.50,
        0.50,
        "apply A, B, and E\nto locked outcomes",
        ha="center",
        va="center",
        color=dark,
        linespacing=1.0,
    )
    ax_c.annotate(
        "",
        xy=(0.50, 0.39),
        xytext=(0.50, 0.45),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=dark),
    )
    ax_c.add_patch(
        FancyBboxPatch(
            (0.03, 0.07),
            0.94,
            0.29,
            boxstyle="round,pad=0.014,rounding_size=0.02",
            facecolor="white",
            edgecolor=accent,
            lw=1.1,
        )
    )
    ax_c.text(
        0.07,
        0.30,
        "Locked audit",
        color=accent,
        fontweight="bold",
        va="center",
    )
    ax_c.text(0.07, 0.19, "A: fail\nB: fail\nE: fail", va="center", linespacing=1.05)
    ax_c.annotate(
        "",
        xy=(0.55, 0.19),
        xytext=(0.42, 0.19),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=dark),
    )
    ax_c.text(
        0.60,
        0.19,
        "MODEL-DESCRIPTIVE\nEXPLANATION\nONLY",
        va="center",
        ha="left",
        fontweight="bold",
        fontsize=7.0,
        linespacing=0.95,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUT_DIR / f"figure_xai_evidence_board_kse.{extension}"
        fig.savefig(
            path,
            dpi=450 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            metadata=(
                {
                    "Creator": "build_kse_figure2.py",
                    "CreationDate": None,
                    "ModDate": None,
                }
                if extension == "pdf"
                else None
            ),
        )
    plt.close(fig)


def main() -> None:
    case = load_case()
    build(case)
    report = {
        "status": "PASS",
        "row_id": ROW_ID,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "observed_residual": case["observed"],
        "predicted_residual": case["prediction"],
        "uniform_panel_a_ticks": [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4],
        "minimum_font_pt": 7.0,
        "outputs": [
            str(
                (
                    OUT_DIR / "figure_xai_evidence_board_kse.pdf"
                ).relative_to(ROOT)
            ),
            str(
                (
                    OUT_DIR / "figure_xai_evidence_board_kse.png"
                ).relative_to(ROOT)
            ),
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
