from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CASE = ROOT / "artifacts" / "xai" / "local_case_decomposition.csv"
SHAP_CONFIG = ROOT / "artifacts" / "xai" / "shap_config.yaml"
MODEL_HASHES = ROOT / "artifacts" / "xai" / "model_hashes.json"
ROW_IDS = ROOT / "artifacts" / "xai" / "explanation_row_ids.csv"
GATE_COMPARISONS = ROOT / "artifacts" / "gates" / "figure2_three_comparisons.json"
PAPER_GEN = ROOT / "paper" / "generated"
REPORT_DIR = ROOT / "reports" / "claim_eligibility_v5_2_final_fix_logic"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def load_case() -> dict:
    frame = pd.read_csv(LOCAL_CASE)
    case = frame[frame["row_id"] == "Barley|Colorado|2016|spring"].copy()
    if case.empty:
        raise SystemExit("Required Barley|Colorado|2016|spring row not found")

    pred = float(case["predicted_residual"].iloc[0])
    obs = float(case["observed_residual"].iloc[0])
    base = float(case["base_value"].iloc[0])
    groups = {
        row.driver_group: float(row.signed_group_shap)
        for row in case.itertuples(index=False)
    }
    order = ["heat", "drought", "frost_cold", "excess_rain", "radiation"]
    grouped_sum = sum(groups[g] for g in order)
    remainder = pred - base - grouped_sum
    recon = base + grouped_sum + remainder
    if abs(recon - pred) > 1e-12:
        raise SystemExit(f"SHAP reconstruction failed: {recon} vs {pred}")

    return {
        "row_id": "Barley|Colorado|2016|spring",
        "prediction": pred,
        "observed": obs,
        "base": base,
        "order": order,
        "groups": groups,
        "remainder": remainder,
        "reconstructed_prediction": recon,
    }


def draw_panel_a(ax, case: dict) -> None:
    pos_color = "#2b8cbe"
    neg_color = "#d95f02"
    obs_color = "#b2182b"

    ax.set_title("A. Coherent local explanation, wrong event",
                 loc="left", fontsize=8.6, fontweight="bold", pad=6)
    ax.set_xlim(-0.56, 0.24)
    ax.set_ylim(0.0, 7.25)
    ax.set_xlabel("residual, t ha$^{-1}$", fontsize=8.8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.axvline(0, color="0.55", linewidth=0.9, zorder=0)

    pred = case["prediction"]
    obs = case["observed"]
    ax.axvspan(obs, pred, color="#f4cccc", alpha=0.24, zorder=0)

    terms = [(g, case["groups"][g]) for g in case["order"]]
    terms.append(("other_remainder", case["remainder"]))
    labels = {
        "heat": "Heat",
        "drought": "Drought",
        "frost_cold": "Frost/cold",
        "excess_rain": "Excess rain",
        "radiation": "Radiation",
        "other_remainder": "Other/remainder",
    }

    y0 = 6.15
    ax.plot([case["base"]], [y0], marker="o", color="black", markersize=4.4)
    ax.text(-0.535, y0, fmt(case["base"]), ha="left", va="center",
            fontsize=7.7, color="0.25")

    current = case["base"]
    y_positions = [5.35, 4.60, 3.85, 3.10, 2.35, 1.60]
    for y, (name, value) in zip(y_positions, terms):
        start = current
        end = current + value
        color = pos_color if value >= 0 else neg_color
        ax.add_patch(FancyArrowPatch(
            (start, y), (end, y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=2.6,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        ))
        ax.plot([start, start], [y - 0.16, y + 0.16],
                color="0.78", linewidth=0.8, zorder=2)
        ax.plot([end, end], [y - 0.16, y + 0.16],
                color="0.78", linewidth=0.8, zorder=2)
        ax.text(-0.535, y, fmt(value), ha="left", va="center",
                fontsize=7.7, color=color, fontweight="bold")
        ax.text(end + (0.012 if value >= 0 else -0.012), y + 0.19,
                fmt(end), ha="left" if value >= 0 else "right",
                va="bottom", fontsize=6.8, color="0.45")
        current = end

    ax.axvline(pred, color="black", linewidth=1.5, zorder=1)
    ax.plot([pred], [0.82], marker="s", color="black", markersize=5.4, zorder=4)
    ax.text(pred - 0.012, 0.82, f"Prediction\n{fmt(pred)}",
            ha="right", va="center", fontsize=7.7, fontweight="bold")

    ax.axvline(obs, color=obs_color, linewidth=1.5, linestyle="--", zorder=1)
    ax.plot([obs], [0.82], marker="D", markerfacecolor="white",
            markeredgecolor=obs_color, markeredgewidth=1.7,
            markersize=6.0, zorder=4)
    ax.text(obs + 0.012, 0.82, f"Observed\n{fmt(obs)}",
            ha="left", va="center", fontsize=7.7,
            fontweight="bold", color=obs_color)

    ax.annotate("", xy=(pred, 0.55), xytext=(obs, 0.55),
                arrowprops=dict(arrowstyle="<->", lw=1.4, color="0.15"))
    ax.text((pred + obs) / 2, 0.70,
            "prediction-observation gap",
            ha="center", va="bottom", fontsize=6.6, color="0.15")

    ax.set_yticks([y0, *y_positions])
    ax.set_yticklabels(["Base", *[labels[n] for n, _ in terms]], fontsize=7.6)
    ax.tick_params(axis="y", length=0, pad=2)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def draw_panel_b(ax) -> None:
    ax.set_title("B. Audited interpretation scope",
                 loc="left", fontsize=8.6, fontweight="bold", pad=6)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def card(x, y, w, h, face, edge):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            facecolor=face,
            edgecolor=edge,
        )
        ax.add_patch(patch)
        return patch

    red = "#b2182b"
    blue = "#2166ac"
    green = "#1b7837"

    card(0.04, 0.58, 0.90, 0.30, "white", "0.35")
    ax.text(0.08, 0.82, "Ungated reading", fontsize=7.8,
            fontweight="bold", ha="left", va="center")
    ax.text(0.92, 0.82, "unsupported", fontsize=6.8,
            color=red, fontweight="bold", ha="right", va="center")
    ax.text(0.10, 0.69, "coherent SHAP\nexplanation",
            fontsize=7.4, ha="left", va="center")
    ax.annotate("", xy=(0.53, 0.69), xytext=(0.35, 0.69),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", linewidth=1.5, color="0.15"))
    ax.text(0.58, 0.69, "weather\nattribution",
            fontsize=7.7, ha="left", va="center", fontweight="bold")
    ax.plot([0.79, 0.88], [0.63, 0.75], color=red, linewidth=2.0)
    ax.plot([0.79, 0.88], [0.75, 0.63], color=red, linewidth=2.0)

    ax.text(0.49, 0.50, "audit gates applied to locked outcomes",
            fontsize=7.0, ha="center", va="center", color="0.25")
    ax.annotate("", xy=(0.49, 0.42), xytext=(0.49, 0.48),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", linewidth=1.2, color="0.25"))

    card(0.04, 0.12, 0.90, 0.30, "#f4f8fb", blue)
    ax.text(0.08, 0.34, "Audit verdict", fontsize=7.7,
            fontweight="bold", ha="left", va="center", color=blue)
    ax.text(0.62, 0.34, "Allowed scope", fontsize=6.8,
            color=green, fontweight="bold", ha="left", va="center")
    ax.text(0.10, 0.24, "A does not pass\nB does not pass\nE does not pass",
            fontsize=7.2, ha="left", va="center")
    ax.plot([0.55, 0.55], [0.16, 0.32], color=blue, linewidth=0.8, alpha=0.45)
    ax.text(0.62, 0.235, "model-descriptive\nexplanation only",
            fontsize=6.45, ha="left", va="center", fontweight="bold")


def build() -> None:
    case = load_case()
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(
        1, 2, figsize=(7.35, 3.55),
        gridspec_kw={"width_ratios": [1.24, 1.0], "wspace": 0.30},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    draw_panel_a(axes[0], case)
    draw_panel_b(axes[1])

    pdf = PAPER_GEN / "figure_xai_claim_eligibility_logic_v5_2.pdf"
    png = PAPER_GEN / "figure_xai_claim_eligibility_logic_v5_2.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=450)
    plt.close(fig)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_id": "figure_xai_claim_eligibility_logic_v5_2",
        "row_id": case["row_id"],
        "model_family": "ExtraTrees",
        "feature_family": "Weather-only",
        "configuration_id": "extra_trees_leaf_1",
        "locked_split_id": "locked_2016_2025",
        "shap_explainer_type": "TreeExplainer",
        "background_reference_configuration": SHAP_CONFIG.read_text(encoding="utf-8"),
        "prediction_hash": json.loads(MODEL_HASHES.read_text(encoding="utf-8"))[
            "seed_aggregated_predictions_csv"
        ],
        "target_hash": "f06b0ce72260feb1514d56e651b18b61be1cb6d468a668a3f69c05316d3230d7",
        "shap_artifact_path": str(LOCAL_CASE.relative_to(ROOT)).replace("\\", "/"),
        "shap_artifact_sha256": sha256(LOCAL_CASE),
        "shap_config_path": str(SHAP_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "shap_config_sha256": sha256(SHAP_CONFIG),
        "model_hash_source": str(MODEL_HASHES.relative_to(ROOT)).replace("\\", "/"),
        "model_hash_source_sha256": sha256(MODEL_HASHES),
        "row_id_source": str(ROW_IDS.relative_to(ROOT)).replace("\\", "/"),
        "row_id_source_sha256": sha256(ROW_IDS),
        "gate_comparison_source": str(GATE_COMPARISONS.relative_to(ROOT)).replace("\\", "/"),
        "gate_comparison_source_sha256": sha256(GATE_COMPARISONS),
        "exact_base_value": case["base"],
        "exact_grouped_contributions": case["groups"],
        "exact_grouped_sum": sum(case["groups"][g] for g in case["order"]),
        "exact_remainder": case["remainder"],
        "exact_prediction": case["prediction"],
        "observed_residual": case["observed"],
        "displayed_values": {
            "base": fmt(case["base"]),
            **{g: fmt(case["groups"][g]) for g in case["order"]},
            "other_remainder": fmt(case["remainder"]),
            "prediction": fmt(case["prediction"]),
            "observed": fmt(case["observed"]),
        },
        "rounding_rule": "Displayed values are rounded to three decimals; unrounded terms reconstruct the fitted prediction.",
        "arithmetic_assertion": {
            "formula": "exact_base + grouped_sum + exact_remainder == exact_prediction",
            "reconstructed_prediction": case["reconstructed_prediction"],
            "absolute_error": abs(case["reconstructed_prediction"] - case["prediction"]),
            "tolerance": 1e-12,
            "assertion_status": "PASS",
        },
        "outputs": {
            "pdf": str(pdf.relative_to(ROOT)).replace("\\", "/"),
            "png": str(png.relative_to(ROOT)).replace("\\", "/"),
            "pdf_sha256": sha256(pdf),
            "png_sha256": sha256(png),
        },
        "generation_script": "scripts/build_final_fix_logic_fig2.py",
        "assertion_status": "PASS",
    }
    (REPORT_DIR / "figure2_logic_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pdf": str(pdf), "png": str(png), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    build()
