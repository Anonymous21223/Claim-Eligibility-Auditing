from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "generated"
REPORT_DIR = ROOT / "reports" / "claim_eligibility_v5_5_final_polish"

XAI_SOURCE = ROOT / "artifacts" / "xai" / "local_case_decomposition.csv"
TABLE_SOURCE = GENERATED / "table_synthetic_scenario_decisions_v5_1.tex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def grouped_shap_case() -> dict:
    rows = []
    with XAI_SOURCE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["row_id"] == "Barley|Colorado|2016|spring":
                rows.append(row)
    if not rows:
        raise RuntimeError("Barley|Colorado|2016|spring not found in XAI source")

    values = {r["driver_group"]: float(r["signed_group_shap"]) for r in rows}
    base = float(rows[0]["base_value"])
    prediction = float(rows[0]["predicted_residual"])
    observed = float(rows[0]["observed_residual"])
    reconstruction_error = float(rows[0]["reconstruction_error"])

    ordered = [
        ("Heat", values["heat"]),
        ("Drought", values["drought"]),
        ("Frost/cold", values["frost_cold"]),
        ("Excess rain", values["excess_rain"]),
        ("Radiation", values["radiation"]),
    ]
    remainder = prediction - sum(v for _, v in ordered)
    ordered.append(("Other/remainder", remainder))

    reconstructed = sum(v for _, v in ordered)
    if abs(reconstructed - prediction) > 1e-10:
        raise AssertionError((reconstructed, prediction))
    if abs(reconstruction_error) > 1e-10:
        raise AssertionError(reconstruction_error)
    if not observed < -0.5:
        raise AssertionError(f"observed_x is not left of -0.5: {observed}")
    if not prediction > 0.2:
        raise AssertionError(f"prediction_x is not right of +0.2: {prediction}")

    return {
        "row_id": rows[0]["row_id"],
        "prediction": prediction,
        "observed": observed,
        "base_value": base,
        "reconstruction_error": reconstruction_error,
        "contributions": ordered,
        "source": str(XAI_SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(XAI_SOURCE),
    }


def build_fig2(case: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.35, 3.42), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.22, 1.62], wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    observed = case["observed"]
    prediction = case["prediction"]

    ax_a.set_title("A. Outcome mismatch", loc="left", fontweight="bold", pad=6)
    ax_a.hlines(0, observed, prediction, color="0.10", lw=2.2, zorder=2)
    ax_a.scatter([observed], [0], s=86, marker="D", facecolors="white",
                 edgecolors="0.05", linewidths=1.2, zorder=4)
    ax_a.scatter([prediction], [0], s=86, marker="s", color="0.05", zorder=4)
    guide_top = 0.58
    ax_a.vlines([observed, prediction], 0.035, guide_top - 0.05, colors="0.05",
                linestyles=(0, (1, 2)), lw=1.0, zorder=1)
    ax_a.text(observed, guide_top, "observed\n-0.510", ha="center", va="bottom",
              fontweight="bold", fontsize=7.0, linespacing=1.05)
    ax_a.text(prediction, guide_top, "prediction\n+0.209", ha="center", va="bottom",
              fontweight="bold", fontsize=7.0, linespacing=1.05)
    ax_a.text(-0.165, 0.24, "wrong direction\nand magnitude", ha="center",
              va="center", color="0.15", fontsize=7.0)
    bracket = FancyArrowPatch(
        (observed + 0.05, 0.145),
        (prediction - 0.05, 0.145),
        arrowstyle="]-[",
        mutation_scale=13,
        connectionstyle="arc3,rad=0.0",
        color="0.05",
        lw=0.9,
    )
    ax_a.add_patch(bracket)
    ax_a.set_xlim(-0.58, 0.25)
    ax_a.set_ylim(-0.54, 0.88)
    ax_a.set_yticks([])
    ax_a.set_xlabel(r"raw residual, t ha$^{-1}$", labelpad=6)
    ax_a.set_xticks([-0.5, -0.3, -0.1, 0.1, 0.2])
    ax_a.spines[["left", "top", "right"]].set_visible(False)

    contribs = sorted(case["contributions"], key=lambda x: abs(x[1]), reverse=True)
    labels = [k for k, _ in contribs]
    vals = np.array([v for _, v in contribs])
    y = np.arange(len(labels))
    colors = ["#0f5da8" if v >= 0 else "#c9c9c9" for v in vals]
    bars = ax_b.barh(y, vals, height=0.74, color=colors, edgecolor="0.20", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val < 0:
            bar.set_hatch("///")
        x = val + 0.004 if val >= 0 else 0.004
        ha = "left"
        ax_b.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                  ha=ha, va="center", fontweight="bold", fontsize=7.0)
    ax_b.vlines(0, -0.42, len(labels) - 0.45, color="0.05", lw=1.2)
    ax_b.set_yticks(y, labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-0.030, 0.095)
    ax_b.set_xlabel(r"contribution to fitted residual, t ha$^{-1}$", labelpad=6)
    ax_b.set_title("B. Why the model predicted +0.209", loc="left", fontweight="bold", pad=18)
    ax_b.text(0.50, 0.985, "SHAP explains the fitted prediction.",
              ha="center", va="top", color="0.15", fontsize=7.3,
              transform=ax_b.transAxes)
    ax_b.set_xticks([-0.025, 0.000, 0.025, 0.050, 0.075],
                    ["-.025", "0", ".025", ".050", ".075"])
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines[["top", "right", "left"]].set_visible(False)

    ax_c.set_axis_off()
    ax_c.set_title("C. Audited interpretation scope", loc="left", fontweight="bold", pad=6)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    red = "#d21f2b"
    blue = "#0b62ad"
    ax_c.add_patch(
        FancyBboxPatch((0.02, 0.61), 0.96, 0.28, boxstyle="round,pad=0.012,rounding_size=0.025",
                       facecolor="white", edgecolor=red, lw=1.1)
    )
    ax_c.text(0.06, 0.83, "Ungated reading", ha="left", va="center",
              fontweight="bold", fontsize=7.2)
    ax_c.text(0.06, 0.72, "Coherent model\nexplanation", ha="left", va="center",
              fontsize=6.1, linespacing=1.0)
    ax_c.annotate("", xy=(0.47, 0.72), xytext=(0.37, 0.72),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.05"))
    ax_c.text(0.53, 0.72, "weather\nattribution", ha="left", va="center",
              fontweight="bold", fontsize=6.0, linespacing=0.92)
    ax_c.text(0.88, 0.72, "X", color=red, fontsize=18, ha="center",
              va="center", fontweight="bold")
    ax_c.text(0.79, 0.64, "unsupported", color=red, ha="center", va="center",
              fontweight="bold", fontsize=6.2)

    ax_c.text(0.50, 0.52, "audit gates applied\nto locked outcomes",
              ha="center", va="center", color="0.10", fontsize=6.8, linespacing=1.05)
    ax_c.annotate("", xy=(0.50, 0.42), xytext=(0.50, 0.48),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.05"))

    ax_c.add_patch(
        FancyBboxPatch((0.02, 0.08), 0.96, 0.30, boxstyle="round,pad=0.012,rounding_size=0.025",
                       facecolor="white", edgecolor=blue, lw=1.1)
    )
    ax_c.text(0.06, 0.32, "Claim-eligibility audit", ha="left", va="center",
              fontweight="bold", color=blue, fontsize=7.1)
    ax_c.text(0.06, 0.20, "A does not pass\nB does not pass\nE does not pass",
              ha="left", va="center", fontsize=6.5, linespacing=1.12)
    ax_c.annotate("", xy=(0.55, 0.20), xytext=(0.49, 0.20),
                  arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.05"))
    ax_c.text(0.60, 0.20, "MODEL-\nDESCRIPTIVE\nEXPLANATION\nONLY",
              ha="left", va="center", fontweight="bold", fontsize=6.3, linespacing=0.86)

    for ext in ["pdf", "png"]:
        out = GENERATED / f"figure_xai_evidence_board_v5_5.{ext}"
        fig.savefig(out, dpi=450 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_table_source() -> tuple[list[dict], str]:
    text = TABLE_SOURCE.read_text(encoding="utf-8")
    source_hash = sha256(TABLE_SOURCE)
    name_map = {
        "No signal": "No signal",
        "Weak signal": "Weak signal",
        "Meas. error": "Measurement error",
        "Temporal drift": "Temporal drift",
        "Leakage": "Leakage",
        "Omitted conf.": "Omitted confounding",
        "Corr. feat.": "Correlated features",
        "Geo. shift": "Geographic shift",
    }
    group_map = {
        "No signal": "Signal absent or degraded",
        "Weak signal": "Signal absent or degraded",
        "Measurement error": "Signal absent or degraded",
        "Temporal drift": "Invalidity that preserves predictive performance",
        "Leakage": "Invalidity that preserves predictive performance",
        "Omitted confounding": "Invalidity that preserves predictive performance",
        "Correlated features": "Invalidity that preserves predictive performance",
        "Geographic shift": "Invalidity that preserves predictive performance",
    }
    order = [
        "No signal",
        "Weak signal",
        "Measurement error",
        "Temporal drift",
        "Leakage",
        "Omitted confounding",
        "Correlated features",
        "Geographic shift",
    ]
    parsed = {}
    for line in text.splitlines():
        if "&" not in line or "\\%" not in line:
            continue
        parts = [part.strip() for part in line.rstrip("\\").split("&")]
        if len(parts) != 7:
            continue
        scenario = parts[0]
        gt = parts[1]
        if scenario not in name_map or gt != "no":
            continue
        label = name_map[scenario]
        audit_match = re.search(r"(\d+)\\%", parts[-1])
        if audit_match is None:
            raise RuntimeError(f"Could not parse Rule value from {line}")
        audit = int(audit_match.group(1))
        parsed[label] = {
            "scenario": label,
            "group": group_map[label],
            "ungated_false_permission_pct": 100,
            "audit_false_permission_pct": audit,
            "reduction_pp": 100 - audit,
        }
    missing = [name for name in order if name not in parsed]
    if missing:
        raise RuntimeError(f"Missing invalid scenarios in Table III source: {missing}")
    rows = [parsed[name] for name in order]
    expected = {
        "No signal": (100, 0, 100),
        "Weak signal": (100, 7, 93),
        "Measurement error": (100, 67, 33),
        "Temporal drift": (100, 97, 3),
        "Leakage": (100, 100, 0),
        "Omitted confounding": (100, 100, 0),
        "Correlated features": (100, 100, 0),
        "Geographic shift": (100, 100, 0),
    }
    for row in rows:
        got = (
            row["ungated_false_permission_pct"],
            row["audit_false_permission_pct"],
            row["reduction_pp"],
        )
        if got != expected[row["scenario"]]:
            raise AssertionError((row["scenario"], got, expected[row["scenario"]]))
    return rows, source_hash


def cell_color(value: float, column: int):
    if column == 0:
        cmap = plt.cm.Blues
    elif column == 1:
        cmap = plt.cm.Greys
    else:
        cmap = plt.cm.Purples
    norm = Normalize(vmin=0, vmax=100)
    return cmap(0.18 + 0.72 * norm(value))


def contrast_text_color(rgba) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.46 else "0.08"


def build_fig3(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.2,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cols = ["Ungated\nFP", "Audit\nFP", "Reduction\n(pp)"]
    groups = [
        ("Signal absent or degraded", rows[:3]),
        ("Invalidity that preserves predictive performance", rows[3:]),
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 3.12),
        gridspec_kw={"height_ratios": [3, 5], "hspace": 0.23},
    )

    def draw_group(ax, title: str, group_rows: list[dict], show_header: bool) -> None:
        labels = [r["scenario"] for r in group_rows]
        data = np.array(
            [
                [r["ungated_false_permission_pct"], r["audit_false_permission_pct"], r["reduction_pp"]]
                for r in group_rows
            ]
        )
        ax.set_xlim(0, 3)
        ax.set_ylim(len(labels), -0.55 if show_header else -0.22)
        ax.set_xticks([])
        if show_header:
            for j, col in enumerate(cols):
                ax.text(j + 0.5, -0.35, col, ha="center", va="center",
                        fontweight="bold", fontsize=5.8, linespacing=0.95)
        ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
        ax.tick_params(axis="both", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.0, 1.03, title, ha="left", va="bottom",
                fontweight="bold", color="0.25", fontsize=5.9,
                transform=ax.transAxes)

        for i in range(len(labels)):
            for j in range(3):
                val = data[i, j]
                color = cell_color(val, j)
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color,
                                       edgecolor="white", lw=1.0))
                ax.text(j + 0.5, i + 0.5, f"{int(val)}", ha="center", va="center",
                        color=contrast_text_color(color), fontweight="bold", fontsize=7.2)
        ax.add_patch(Rectangle((0, 0), 3, len(labels), fill=False, edgecolor="0.25", lw=0.8))

    draw_group(axes[0], groups[0][0], groups[0][1], show_header=True)
    draw_group(axes[1], groups[1][0], groups[1][1], show_header=False)
    axes[1].text(0, len(groups[1][1]) + 0.28, "Values are percentages across 30 seeds.",
                 ha="left", va="top", color="0.25", fontsize=5.8)
    fig.subplots_adjust(left=0.39, right=0.98, top=0.88, bottom=0.12)

    for ext in ["pdf", "png"]:
        out = GENERATED / f"figure_synthetic_heatmap_v5_5.{ext}"
        fig.savefig(out, dpi=450 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fig2_case = grouped_shap_case()
    build_fig2(fig2_case)
    fig3_rows, table_hash = parse_table_source()
    build_fig3(fig3_rows)

    provenance = {
        "fig2": {
            "config_id": "v5_5_final_polish_evidence_board",
            "row_id": fig2_case["row_id"],
            "prediction_x": fig2_case["prediction"],
            "observed_x": fig2_case["observed"],
            "geometry_assertions": {
                "observed_x_equals_source_value": fig2_case["observed"],
                "prediction_x_equals_source_value": fig2_case["prediction"],
                "observed_left_of_minus_0_5": fig2_case["observed"] < -0.5,
                "prediction_right_of_plus_0_2": fig2_case["prediction"] > 0.2,
                "reconstructs_prediction": True,
            },
            "shap_source": fig2_case["source"],
            "shap_source_sha256": fig2_case["source_sha256"],
            "exact_values": fig2_case,
            "outputs": [
                "paper/generated/figure_xai_evidence_board_v5_5.pdf",
                "paper/generated/figure_xai_evidence_board_v5_5.png",
            ],
        },
        "fig3": {
            "config_id": "v5_5_final_polish_synthetic_heatmap",
            "table_iii_source": str(TABLE_SOURCE.relative_to(ROOT)),
            "table_iii_source_sha256": table_hash,
            "values": fig3_rows,
            "assertion": "Fig. 3 values are parsed from the Table III machine-readable source hash above; ungated false permission is fixed at 100% for invalid regimes by the synthetic benchmark summary.",
            "outputs": [
                "paper/generated/figure_synthetic_heatmap_v5_5.pdf",
                "paper/generated/figure_synthetic_heatmap_v5_5.png",
            ],
        },
    }
    (REPORT_DIR / "figure_provenance_v5_5.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
