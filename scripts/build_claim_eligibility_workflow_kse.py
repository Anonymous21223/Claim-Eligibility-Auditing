"""Build the KSE one-column claim-eligibility workflow.

The figure deliberately separates:
1. the locked study pipeline,
2. evaluation of all applicable evidence modules, and
3. the sequential rule that grants the highest permitted claim.

Outputs are deterministic vector PDF and high-resolution PNG files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    ROOT
    / "paper_versions"
    / "v5_9_kse_model_first_6pages"
    / "source"
    / "figures"
)
PDF_PATH = OUT_DIR / "claim_eligibility_workflow_kse.pdf"
PNG_PATH = OUT_DIR / "claim_eligibility_workflow_kse.png"

INK = "#111111"
MUTED = "#4B5563"
LINE = "#374151"
PIPE_FILL = "#F3F4F6"
MODULE_FILL = "#E9EFF5"
OUTCOME_FILL = "#FAFAFA"
WHITE = "#FFFFFF"


def rounded_box(
    ax,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    fontsize: float = 7.0,
    weight: str = "semibold",
    linewidth: float = 0.9,
    radius: float = 0.014,
    zorder: int = 3,
) -> FancyBboxPatch:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=LINE,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        fontstretch="condensed",
        color=INK,
        linespacing=1.06,
        zorder=zorder + 1,
    )
    return patch


def diamond(
    ax,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fontsize: float = 7.0,
) -> Polygon:
    x, y = center
    points = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    patch = Polygon(
        points,
        closed=True,
        facecolor=MODULE_FILL,
        edgecolor=LINE,
        linewidth=0.95,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="semibold",
        fontstretch="condensed",
        color=INK,
        linespacing=1.04,
        zorder=4,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    connectionstyle: str = "arc3",
    mutation_scale: float = 7.5,
    linewidth: float = 0.85,
    label_fontsize: float = 7.0,
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=LINE,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        lx, ly = label_xy if label_xy is not None else (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
        )
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            fontstretch="condensed",
            color=MUTED,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.6},
            zorder=5,
        )
    return patch


def layer_label(ax, y: float, number: str, text: str) -> None:
    ax.text(
        0.055,
        y,
        number,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=WHITE,
        bbox={
            "boxstyle": "round,pad=0.23,rounding_size=0.25",
            "facecolor": LINE,
            "edgecolor": LINE,
        },
        zorder=6,
    )
    ax.text(
        0.100,
        y,
        text,
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        fontstretch="condensed",
        color=INK,
        zorder=6,
    )


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.50, 6.75), facecolor=WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(WHITE)

    # Layer 1: locked study pipeline.
    layer_label(ax, 0.972, "1", "LOCKED STUDY PIPELINE")
    pipeline = [
        (0.915, "Raw snapshots + hashes"),
        (0.855, "Train-only target, scale, and preprocessing"),
        (0.795, "Validation model and feature selection"),
        (0.735, "Lock configuration + requested claim"),
        (0.675, "Predict locked future rows"),
    ]
    for index, (y, text) in enumerate(pipeline):
        rounded_box(
            ax,
            (0.50, y),
            0.79,
            0.041,
            text,
            facecolor=PIPE_FILL,
            fontsize=7.0,
            radius=0.012,
        )
        if index < len(pipeline) - 1:
            arrow(ax, (0.50, y - 0.027), (0.50, pipeline[index + 1][0] + 0.027))

    # Layer 2: all applicable modules are evaluated independently.
    layer_label(ax, 0.613, "2", "EVALUATE ALL APPLICABLE MODULES")
    module_y = 0.526
    module_specs = [
        (0.18, "A\nOverall\npredictive\nadequacy"),
        (0.50, "B\nIncremental\nfeature-group\nvalue"),
        (0.82, "E\nEvent\nrecovery\n(if requested)"),
    ]
    for x, text in module_specs:
        rounded_box(
            ax,
            (x, module_y),
            0.285,
            0.080,
            text,
            facecolor=MODULE_FILL,
            fontsize=7.0,
            radius=0.010,
        )

    # One prediction vector fans out to all applicable measurements.
    ax.plot([0.18, 0.82], [0.579, 0.579], color=LINE, linewidth=0.85, zorder=1)
    arrow(
        ax,
        (0.50, 0.653),
        (0.50, 0.579),
        mutation_scale=6.8,
    )
    for x in (0.18, 0.50, 0.82):
        arrow(ax, (x, 0.579), (x, module_y + 0.043), mutation_scale=6.8)
    ax.text(
        0.50,
        0.463,
        "All module measurements may be reported.\nPermission is decided below.",
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTED,
        zorder=5,
    )

    # Layer 3: sequential permission, separate from evidence evaluation.
    layer_label(ax, 0.420, "3", "SEQUENTIAL CLAIM PERMISSION")

    center_x = 0.50
    a_y, b_y, request_y, e_y = 0.344, 0.250, 0.156, 0.062
    diamond(ax, (center_x, a_y), 0.265, 0.078, "A passes?", fontsize=6.8)
    diamond(ax, (center_x, b_y), 0.265, 0.078, "B passes?", fontsize=6.8)
    diamond(
        ax,
        (center_x, request_y),
        0.340,
        0.096,
        "Event-level claim\nrequested?",
        fontsize=7.0,
    )
    diamond(ax, (center_x, e_y), 0.265, 0.078, "E passes?", fontsize=6.8)

    rounded_box(
        ax,
        (0.145, a_y),
        0.245,
        0.068,
        "MODEL-DESCRIPTIVE\nEXPLANATION ONLY",
        facecolor=OUTCOME_FILL,
        fontsize=5.5,
    )
    rounded_box(
        ax,
        (0.145, b_y),
        0.245,
        0.068,
        "OVERALL\nPREDICTIVE\nCLAIM ONLY",
        facecolor=OUTCOME_FILL,
        fontsize=7.0,
    )
    rounded_box(
        ax,
        (0.858, 0.123),
        0.255,
        0.110,
        "FEATURE\nSPECIFIC\nPREDICTIVE\nRELIANCE ONLY",
        facecolor=OUTCOME_FILL,
        fontsize=7.0,
    )
    rounded_box(
        ax,
        (0.145, e_y),
        0.245,
        0.068,
        "EVENT\nRECOVERY\nCLAIM",
        facecolor=OUTCOME_FILL,
        fontsize=7.0,
    )

    # Central PASS path.
    arrow(
        ax,
        (center_x, a_y - 0.041),
        (center_x, b_y + 0.041),
        label="PASS",
        label_xy=(0.558, 0.297),
        label_fontsize=5.8,
    )
    arrow(
        ax,
        (center_x, b_y - 0.041),
        (center_x, request_y + 0.044),
        label="PASS",
        label_xy=(0.558, 0.203),
        label_fontsize=5.8,
    )
    arrow(
        ax,
        (center_x, request_y - 0.050),
        (center_x, e_y + 0.041),
        label="YES",
        label_xy=(0.552, 0.109),
        label_fontsize=5.8,
    )

    # Stopping paths and claim outcomes.
    arrow(
        ax,
        (center_x - 0.137, a_y),
        (0.274, a_y),
        label="NOT\nPASSED",
        label_xy=(0.338, a_y + 0.027),
        label_fontsize=5.2,
    )
    arrow(
        ax,
        (center_x - 0.137, b_y),
        (0.274, b_y),
        label="NOT\nPASSED",
        label_xy=(0.338, b_y + 0.027),
        label_fontsize=5.2,
    )
    arrow(
        ax,
        (center_x + 0.172, request_y),
        (0.728, 0.145),
        label="NO",
        label_xy=(0.700, 0.175),
        label_fontsize=5.8,
        connectionstyle="arc3,rad=-0.06",
    )
    arrow(
        ax,
        (center_x + 0.137, e_y),
        (0.728, 0.091),
        label="NOT\nPASSED",
        label_xy=(0.688, 0.033),
        label_fontsize=5.2,
        connectionstyle="arc3,rad=-0.08",
    )
    arrow(
        ax,
        (center_x - 0.137, e_y),
        (0.274, e_y),
        label="PASS",
        label_xy=(0.319, e_y + 0.017),
        label_fontsize=5.8,
    )

    # Thin same-module links connect each measurement box to its decision.
    link_style = {"color": "#9AA3AD", "linewidth": 0.55, "linestyle": (0, (2, 2)), "zorder": 0}
    ax.plot([0.18, 0.30, 0.30, center_x], [0.484, 0.440, 0.405, a_y + 0.041], **link_style)
    ax.plot([0.645, 0.85, 0.85, center_x + 0.137], [module_y, module_y, b_y, b_y], **link_style)
    ax.plot([0.82, 0.70, 0.70, center_x + 0.137], [0.484, 0.440, e_y, e_y], **link_style)

    # Thin separators reinforce the three conceptual layers.
    for y in (0.646, 0.440):
        ax.plot([0.04, 0.96], [y, y], color="#CBD0D6", linewidth=0.65, zorder=0)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.008)
    fig.savefig(
        PDF_PATH,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor=WHITE,
        metadata={
            "Creator": "build_claim_eligibility_workflow_kse.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        PNG_PATH,
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor=WHITE,
    )
    plt.close(fig)


if __name__ == "__main__":
    build()
