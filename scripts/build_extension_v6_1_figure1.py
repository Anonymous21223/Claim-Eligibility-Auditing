"""Build the V6.1 leaderboard delta-RMSE figure from the locked leaderboard."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "artifacts" / "extensions" / "v1" / "core"
SOURCE = ROOT / "paper_versions" / "v6_1_extension_factorial_anonymous" / "source"
OUT = SOURCE / "figures"
REPORT = SOURCE / "generated" / "figure1_provenance.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    leaderboard = pd.read_csv(CORE / "validation_leaderboard.csv")
    lock = json.loads((CORE / "winner_lock.json").read_text(encoding="utf-8"))
    winner = lock["winner"]
    winner_row = leaderboard[
        (leaderboard.arm_id == winner["arm_id"])
        & (leaderboard.model_id == winner["model_id"])
        & (leaderboard.feature_family == winner["feature_family"])
    ].iloc[0]
    top = leaderboard.sort_values("pooled_rmse", kind="stable").head(15).copy()
    top["delta_milli"] = 1000 * (top["pooled_rmse"] - winner_row.pooled_rmse)
    top["label"] = top.apply(
        lambda row: f"{row.arm_id} | {row.model_id.replace('_', ' ')} | {row.feature_family.replace('_', ' ')}",
        axis=1,
    )
    top = top.sort_values("delta_milli", ascending=False, kind="stable")

    colors = {True: "#176FA6", False: "#9AA3AA"}
    markers = {"A+B": "s", "A": "o", "none": "x"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.50, 3.78), facecolor="white")
    y = list(range(len(top)))
    for position, (_, row) in zip(y, top.iterrows()):
        color = colors[bool(row.R1_stable_positive_pass)]
        ax.plot([0, row.delta_milli], [position, position], color="#C8CDD1", linewidth=0.75, zorder=1)
        ax.scatter(
            row.delta_milli,
            position,
            s=30,
            marker=markers[row.claim_tier],
            color=color,
            edgecolor="#1D2A32" if markers[row.claim_tier] != "x" else color,
            linewidth=0.45,
            zorder=2,
        )
    winner_position = top.index[top.arm_id.eq(winner["arm_id"]) & top.model_id.eq(winner["model_id"]) & top.feature_family.eq(winner["feature_family"])]
    if len(winner_position) != 1:
        raise AssertionError("Winner is absent or duplicated in the displayed candidates")
    display_position = list(top.index).index(winner_position[0])
    ax.scatter(0, display_position, s=74, marker="*", color="#C53B37", edgecolor="#651A18", linewidth=0.5, zorder=3)

    ax.axvline(0, color="#30363A", linewidth=0.8)
    ax.set_yticks(y, top.label)
    ax.set_xlabel(r"Pooled RMSE minus C310 ($10^{-3}$ t ha$^{-1}$)")
    ax.grid(axis="x", color="#E3E6E8", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=6.15)
    ax.margins(y=0.035)

    legend_items = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#176FA6", markeredgecolor="#1D2A32", markersize=5, label="R1 pass"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#9AA3AA", markeredgecolor="#1D2A32", markersize=5, label="R1 fail"),
        plt.Line2D([0], [0], marker="s", linestyle="none", color="#4C5961", markersize=5, label="A+B"),
        plt.Line2D([0], [0], marker="o", linestyle="none", color="#4C5961", markersize=5, label="A"),
        plt.Line2D([0], [0], marker="x", linestyle="none", color="#4C5961", markersize=5, label="none"),
        plt.Line2D([0], [0], marker="*", linestyle="none", color="#C53B37", markersize=7, label="selected"),
    ]
    ax.legend(handles=legend_items, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), frameon=False, fontsize=6.1, columnspacing=0.8, handletextpad=0.3)
    fig.subplots_adjust(left=0.44, right=0.985, top=0.82, bottom=0.13)

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "leaderboard_delta_rmse.pdf"
    png = OUT / "leaderboard_delta_rmse.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, metadata={"Creator": "anonymous artifact script", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    report = {
        "status": "PASS",
        "displayed_candidates": len(top),
        "x_definition": "1000 * (pooled_rmse - locked winner pooled_rmse)",
        "selection_highlight_uses_external_results": False,
        "input_sha256": {"validation_leaderboard.csv": digest(CORE / "validation_leaderboard.csv"), "winner_lock.json": digest(CORE / "winner_lock.json")},
        "output_sha256": {pdf.name: digest(pdf), png.name: digest(png)},
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
