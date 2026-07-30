"""Build final-round visuals and compact tables from existing audit artifacts.

This script does not train models or change experiment outputs. It reads locked
prediction vectors, gate summaries, XAI artifacts, and synthetic benchmark files,
then writes manuscript-facing figures/tables plus provenance records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import textwrap
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
PAPER_GEN = ROOT / "paper" / "generated"
ARTIFACTS = ROOT / "artifacts"
MAP_DIR = ARTIFACTS / "maps"
SUPP_DIR = ARTIFACTS / "supplement"


STATE_ABBR = {
    "Alabama": "AL",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

CONTIGUOUS_EXCLUDE = {"Alaska", "Hawaii", "Puerto Rico"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def esc(text: object) -> str:
    value = str(text)
    for old, new in {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }.items():
        value = value.replace(old, new)
    return value


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100 * value:.0f}\\%"


def ensure_dirs() -> None:
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "figures").mkdir(parents=True, exist_ok=True)


def build_claim_module_definition_table() -> None:
    rows = [
        (
            "A -- Overall adequacy",
            "Does the locked model improve on the same-target baseline?",
            "Upper paired 95\\% CI of $\\Delta$RMSE $<0$.",
            "Overall predictive claim",
        ),
        (
            "B -- Weather value",
            "Does Full improve on Metadata-only on the same locked rows?",
            "Upper paired 95\\% CI of $\\Delta$RMSE $<0$.",
            "Weather-specific predictive reliance",
        ),
        (
            "E -- Event recovery",
            "Does the model recover the primary below-trend tail beyond null checks?",
            "Tail RMSE/MAE, rank, permutation, and top-$k$ checks all pass.",
            "Observed below-trend event claim",
        ),
        (
            "D -- Diagnostic",
            "How does Weather-only compare with Metadata-only?",
            "Reported only; not a decision module.",
            "Descriptive diagnostic only",
        ),
    ]
    lines = [
        "\\begin{tabular}{p{0.16\\textwidth}p{0.24\\textwidth}p{0.34\\textwidth}p{0.16\\textwidth}}",
        "\\toprule",
        "Module & Question & Pass condition & Claim scope \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (PAPER_GEN / "table_gate_definition.tex").write_text("\n".join(lines), encoding="utf-8")


def build_gate_module_table() -> None:
    gates = json.loads((ARTIFACTS / "gates" / "figure2_three_comparisons.json").read_text(encoding="utf-8"))
    gate_rows = {item["gate"]: item for item in gates["comparisons"]}
    gate_a = gate_rows["Gate A"]
    gate_b = gate_rows["Gate B1 PRIMARY"]
    gate_d = gate_rows["Gate B2 DIAGNOSTIC"]
    tail = pd.read_csv(ARTIFACTS / "audit" / "tail" / "tail_metrics_by_threshold.csv")
    primary_tail = tail.query("threshold == 'z<-1'").iloc[0]
    rank = pd.read_csv(ARTIFACTS / "audit_records" / "rank_null_audit.csv").query("threshold == 'z<-1'").iloc[0]
    topk = pd.read_csv(ARTIFACTS / "audit_records" / "topk_null_audit.csv").query("threshold == 'z<-1' and definition == 'k=10'").iloc[0]

    rows = [
        [
            "A",
            "Overall predictive adequacy",
            "Selected Weather-only vs. zero residual",
            f"{fmt3(gate_a['estimate'])} [{fmt3(gate_a['ci95_low'])}, {fmt3(gate_a['ci95_high'])}]",
            "FAIL",
            "Blocks overall and higher claim levels",
        ],
        [
            "B",
            "Incremental weather value",
            "Full vs. Metadata-only",
            f"{fmt3(gate_b['estimate'])} [{fmt3(gate_b['ci95_low'])}, {fmt3(gate_b['ci95_high'])}]",
            "FAIL",
            "Blocks weather-specific predictive reliance",
        ],
        [
            "E",
            "Event-recovery adequacy",
            "$z<-1$ tail error, rank, and top-10 null checks",
            (
                f"RMSE {fmt3(primary_tail.paired_delta_rmse)} "
                f"[{fmt3(primary_tail.paired_delta_rmse_ci95_low)}, {fmt3(primary_tail.paired_delta_rmse_ci95_high)}]; "
                f"rank {fmt3(rank.spearman)}; top-10 {int(topk.overlap)}/{int(topk.k)}"
            ),
            "FAIL",
            "Error CI, rank, and top-10 checks do not all pass",
        ],
        [
            "D",
            "Representation diagnostic",
            "Weather-only vs. Metadata-only",
            f"{fmt3(gate_d['estimate'])} [{fmt3(gate_d['ci95_low'])}, {fmt3(gate_d['ci95_high'])}]",
            "DESCRIPTIVE ONLY",
            "CI crosses zero; diagnostic only",
        ],
    ]
    lines = [
        "\\begin{tabular}{p{0.04\\textwidth}p{0.18\\textwidth}p{0.24\\textwidth}p{0.22\\textwidth}p{0.12\\textwidth}p{0.13\\textwidth}}",
        "\\toprule",
        "Mod. & Role & Contrast/check & Estimate or primary statistic & Status & Decision role \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (PAPER_GEN / "table_gate_modules.tex").write_text("\n".join(lines), encoding="utf-8")


def draw_workflow() -> None:
    supplied_png = ARTIFACTS / "figures" / "wf_US.drawio.png"
    if supplied_png.exists():
        render_workflow_png(supplied_png)
        return

    supplied_svg = ARTIFACTS / "figures" / "wf_US.drawio.svg"
    if supplied_svg.exists() and render_workflow_svg(supplied_svg):
        return

    supplied = ARTIFACTS / "figures" / "workflow_claim_hierarchy.png"
    if supplied.exists():
        image = plt.imread(supplied)
        fig, ax = plt.subplots(figsize=(10.8, 6.0))
        ax.imshow(image)
        ax.set_axis_off()
        for ext in ["pdf", "png"]:
            fig.savefig(PAPER_GEN / f"figure_branched_workflow.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
            fig.savefig(PAPER_GEN / f"figure_workflow_us.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#f5f7fb", ec="#2f3a4a", lw=1.1, fs=7.8, dashed=False):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
        return rect

    def diamond(cx, cy, w, h, text, fc="#fff7e6", ec="#8a5a00", fs=7.6, dashed=False):
        poly = patches.Polygon(
            [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
            closed=True,
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, wrap=True)
        return poly

    def arrow(x1, y1, x2, y2, label=None, color="#2f3a4a", dashed=False):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.0,
                color=color,
                linestyle="--" if dashed else "-",
                shrinkA=3,
                shrinkB=3,
            ),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label, fontsize=7.8, color=color, ha="center")

    box(0.03, 0.76, 0.15, 0.12, "Raw snapshots\n+ hashes")
    box(0.22, 0.76, 0.17, 0.12, "Train-only target,\nscale, preprocessing")
    box(0.43, 0.76, 0.17, 0.12, "Validation model\nand feature selection")
    box(0.64, 0.76, 0.15, 0.12, "Lock config\n+ claim type")
    box(0.83, 0.76, 0.14, 0.12, "Predict locked\nfuture rows")
    for x1, x2 in [(0.18, 0.22), (0.39, 0.43), (0.60, 0.64), (0.79, 0.83)]:
        arrow(x1, 0.82, x2, 0.82)

    diamond(0.18, 0.52, 0.18, 0.12, "A\nOverall\npredictive adequacy")
    diamond(0.43, 0.52, 0.18, 0.12, "B\nIncremental\nweather value")
    diamond(0.68, 0.52, 0.18, 0.12, "Event-level\nclaim requested?")
    diamond(0.68, 0.33, 0.18, 0.12, "E\nTail error +\nrank + top-k")
    diamond(0.90, 0.52, 0.13, 0.10, "D\nDiagnostic", fc="#f3f3f3", ec="#666666", fs=7.2, dashed=True)

    arrow(0.90, 0.76, 0.18, 0.58, "paired rows")
    arrow(0.90, 0.76, 0.43, 0.58)
    arrow(0.90, 0.76, 0.90, 0.58, dashed=True)
    arrow(0.52, 0.52, 0.59, 0.52, "if weather claim")
    arrow(0.68, 0.46, 0.68, 0.39, "YES")

    box(0.05, 0.16, 0.18, 0.12, "A FAIL:\nMODEL-DESCRIPTIVE\nONLY", fc="#fdeeee", ec="#9d2f2f", fs=7.4)
    box(0.29, 0.16, 0.18, 0.12, "A PASS, B FAIL:\nOVERALL CLAIM\nONLY", fc="#eef5ff", ec="#315f9d", fs=7.2)
    box(0.53, 0.16, 0.18, 0.12, "A+B PASS,\nE not requested:\nWEATHER RELIANCE", fc="#eef8ef", ec="#337a3f", fs=7.0)
    box(0.77, 0.16, 0.18, 0.12, "A+B+E PASS:\nEVENT-RECOVERY\nCLAIM", fc="#eef8ef", ec="#337a3f", fs=7.0)

    arrow(0.15, 0.46, 0.14, 0.28, "FAIL", color="#9d2f2f")
    arrow(0.38, 0.46, 0.38, 0.28, "B FAIL", color="#315f9d")
    arrow(0.68, 0.46, 0.62, 0.28, "NO")
    arrow(0.68, 0.27, 0.86, 0.28, "PASS")

    ax.text(
        0.5,
        0.045,
        "Any required module failure abstains from that claim level; Module D is diagnostic only and never changes selection or permission.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    for ext in ["pdf", "png"]:
        fig.savefig(PAPER_GEN / f"figure_branched_workflow.{ext}", bbox_inches="tight", dpi=300)
        fig.savefig(PAPER_GEN / f"figure_workflow_us.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _force_svg_light_scheme(svg_text: str) -> str:
    token = "light-dark("
    out = []
    i = 0
    while i < len(svg_text):
        j = svg_text.find(token, i)
        if j < 0:
            out.append(svg_text[i:])
            break
        out.append(svg_text[i:j])
        k = j + len(token)
        depth = 0
        comma = None
        end = None
        pos = k
        while pos < len(svg_text):
            ch = svg_text[pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    end = pos
                    break
                depth -= 1
            elif ch == "," and depth == 0 and comma is None:
                comma = pos
            pos += 1
        if comma is None or end is None:
            out.append(svg_text[j:pos])
            i = pos
            continue
        out.append(svg_text[k:comma].strip())
        i = end + 1
    return "".join(out).replace("color-scheme: light dark;", "color-scheme: light;")


def _find_chrome_executable() -> Path | None:
    for candidate in [
        os.environ.get("CHROME"),
        os.environ.get("CHROME_PATH"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def render_workflow_svg(svg_path: Path) -> bool:
    chrome = _find_chrome_executable()
    if chrome is None:
        return False

    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    light_svg = PAPER_GEN / "wf_US.drawio.light.svg"
    light_svg.write_text(_force_svg_light_scheme(svg_path.read_text(encoding="utf-8")), encoding="utf-8")
    png = PAPER_GEN / "figure_workflow_us.png"
    pdf = PAPER_GEN / "figure_workflow_us.pdf"
    subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--window-size=8105,5760",
            f"--screenshot={png.resolve()}",
            f"file:///{light_svg.resolve().as_posix()}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    image = plt.imread(png)
    fig_width = 10.8
    fig_height = fig_width * image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return True


def render_workflow_png(png_path: Path) -> None:
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    image = plt.imread(png_path)
    fig_width = 10.8
    fig_height = fig_width * image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(image)
    ax.set_axis_off()
    for ext in ["pdf", "png"]:
        fig.savefig(PAPER_GEN / f"figure_workflow_us.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def draw_xai_figure() -> None:
    shap = pd.read_csv(ARTIFACTS / "xai" / "shap_family_share.csv")
    shap = shap.sort_values("share_of_total", ascending=True)
    display_pct = (shap["share_of_total"] * 100).round(1).tolist()
    if display_pct:
        display_pct[-1] = round(display_pct[-1] + (100.0 - sum(display_pct)), 1)
    local = pd.read_csv(ARTIFACTS / "xai" / "local_case_decomposition.csv")
    case = local[local["row_id"] == "Barley|Colorado|2016|spring"].copy()
    case["abs"] = case["signed_group_shap"].abs()
    case = case.sort_values("abs", ascending=False).head(5)
    gates = pd.read_csv(ARTIFACTS / "gates" / "gates.csv")
    tail = pd.read_csv(ARTIFACTS / "audit" / "tail" / "tail_metrics_by_threshold.csv").query("threshold == 'z<-1'").iloc[0]

    fig = plt.figure(figsize=(10.8, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    colors = {"weather": "#2b7a78", "state": "#7a5195", "crop": "#ef8354", "location": "#4f5d75"}
    ax1.barh(shap["feature_family"], shap["share_of_total"] * 100, color=[colors.get(x, "#777777") for x in shap["feature_family"]])
    for y, (val, label_pct) in enumerate(zip(shap["share_of_total"] * 100, display_pct)):
        ax1.text(val + 1.2, y, f"{label_pct:.1f}%", va="center", fontsize=8)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("share of |SHAP| (%)", fontsize=8)
    ax1.set_title("A. Global fitted-function share", fontsize=9, loc="left")
    ax1.tick_params(labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    base = float(case["base_value"].iloc[0])
    running = base
    labels = []
    starts = []
    vals = []
    for _, row in case.iterrows():
        labels.append(row["driver_group"].replace("_", " "))
        starts.append(running)
        val = float(row["signed_group_shap"])
        vals.append(val)
        running += val
    y = np.arange(len(vals))
    ax2.axvline(0, color="#444444", lw=0.8)
    for i, (start, val) in enumerate(zip(starts, vals)):
        left = min(start, start + val)
        width = abs(val)
        color = "#b63a3a" if val > 0 else "#2e6f9e"
        ax2.barh(i, width, left=left, color=color, alpha=0.85)
        ax2.text(start + val + (0.005 if val >= 0 else -0.005), i, f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=7.5)
    predicted = float(case["predicted_residual"].iloc[0])
    observed = float(case["observed_residual"].iloc[0])
    ax2.axvline(predicted, color="#111111", lw=1.1, linestyle="-")
    ax2.axvline(observed, color="#7f1d1d", lw=1.1, linestyle="--")
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("signed contribution to residual prediction", fontsize=8)
    ax2.set_title("B. Barley-Colorado 2016 local decomposition", fontsize=9, loc="left")
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.02, -0.18, f"predicted {predicted:.3f}; observed {observed:.3f}", transform=ax2.transAxes, fontsize=7.5)

    rows = []
    for _, row in gates.iterrows():
        if row["gate_id"] in {"Gate A", "Gate B1"}:
            label = {"Gate A": "A", "Gate B1": "B"}[row["gate_id"]]
            rows.append((label, row["point_estimate"], row["ci_low"], row["ci_high"], row["status"]))
    rows.append(("E", tail.paired_delta_rmse, tail.paired_delta_rmse_ci95_low, tail.paired_delta_rmse_ci95_high, "FAIL"))
    y = np.arange(len(rows))
    for i, (label, est, lo, hi, status) in enumerate(rows):
        ax3.plot([lo, hi], [i, i], color="#333333", lw=1.4)
        ax3.scatter([est], [i], color="#b63a3a" if status == "FAIL" else "#2b7a3d", zorder=3)
        ax3.text(hi + 0.004, i, status, va="center", fontsize=8)
    ax3.axvline(0, color="#555555", lw=0.9)
    ax3.set_yticks(y, [r[0] for r in rows])
    ax3.invert_yaxis()
    ax3.set_xlabel("Delta RMSE (left minus right)", fontsize=8)
    ax3.set_title("C. Required modules fail", fontsize=9, loc="left")
    ax3.tick_params(labelsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.text(
        0.0,
        -0.18,
        "E shows primary-tail RMSE; it also fails rank and top-k checks.",
        transform=ax3.transAxes,
        fontsize=7.1,
    )

    for ext in ["pdf", "png"]:
        fig.savefig(PAPER_GEN / f"figure_xai_abstention.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def download_census_geometry() -> tuple[Path, str]:
    url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
    zip_path = MAP_DIR / "cb_2018_us_state_20m.zip"
    source_used = "census_direct_zip"
    try:
        if not zip_path.exists() or zip_path.stat().st_size < 100_000:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                zip_path.write_bytes(response.read())
        zsha = sha256(zip_path)
        extract_dir = MAP_DIR / "cb_2018_us_state_20m"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        geometry_path = next(extract_dir.glob("*.shp"))
    except Exception as exc:
        source_used = "publicamundi_geojson_mirror"
        mirror_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        geometry_path = MAP_DIR / "us-states-publicamundi.geojson"
        if not geometry_path.exists():
            request = urllib.request.Request(mirror_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                geometry_path.write_bytes(response.read())
        zsha = sha256(geometry_path)
        fallback_error = repr(exc)
    else:
        fallback_error = None
        mirror_url = None

    (MAP_DIR / "census_state_geometry_provenance.json").write_text(
        json.dumps(
            {
                "official_source": "U.S. Census Bureau Cartographic Boundary Files for states",
                "official_url": "https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html",
                "attempted_census_zip_url": url,
                "source_used": source_used,
                "mirror_url": mirror_url,
                "fallback_error": fallback_error,
                "geometry_sha256": zsha,
                "local_geometry": geometry_path.relative_to(ROOT).as_posix(),
                "retrieved_or_reused_utc": pd.Timestamp.utcnow().isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return geometry_path, zsha


def compute_state_metrics() -> pd.DataFrame:
    preds = pd.read_csv(ARTIFACTS / "audit" / "final_test" / "seed_aggregated_predictions.csv")
    keep = preds[
        ((preds["config_id"] == "extra_trees_leaf_1") & (preds["feature_family"] == "weather_only"))
        | ((preds["config_id"] == "zero_residual") & (preds["feature_family"] == "baseline"))
    ].copy()
    rows = []
    for state, grp in keep.groupby("region"):
        weather = grp[(grp["config_id"] == "extra_trees_leaf_1") & (grp["feature_family"] == "weather_only")].set_index("row_id")
        zero = grp[(grp["config_id"] == "zero_residual") & (grp["feature_family"] == "baseline")].set_index("row_id")
        common = weather.index.intersection(zero.index)
        y = weather.loc[common, "trend_residual_t_ha"].astype(float)
        pred = weather.loc[common, "prediction"].astype(float)
        zpred = zero.loc[common, "prediction"].astype(float)
        wrmse = math.sqrt(float(np.mean((y - pred) ** 2)))
        zrmse = math.sqrt(float(np.mean((y - zpred) ** 2)))
        rows.append(
            {
                "state": state,
                "abbr": STATE_ABBR[state],
                "n_locked_rows": len(common),
                "weather_rmse_t_ha": wrmse,
                "zero_rmse_t_ha": zrmse,
                "delta_rmse_t_ha": wrmse - zrmse,
            }
        )
    out = pd.DataFrame(rows).sort_values("state")
    out.to_csv(MAP_DIR / "state_level_locked_delta_rmse.csv", index=False)
    assert len(out) == 12, f"Expected 12 states, found {len(out)}"
    assert set(out["state"]) == {
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
    return out


def draw_state_map() -> None:
    shp, zsha = download_census_geometry()
    metrics = compute_state_metrics()
    gdf = gpd.read_file(shp)
    if "NAME" in gdf.columns:
        gdf = gdf.rename(columns={"NAME": "state"})
    elif "name" in gdf.columns:
        gdf = gdf.rename(columns={"name": "state"})
    else:
        raise ValueError(f"Cannot identify state-name column in {shp}: {gdf.columns.tolist()}")
    gdf = gdf[~gdf["state"].isin(CONTIGUOUS_EXCLUDE)].copy()
    gdf["abbr"] = gdf["state"].map(STATE_ABBR)
    gdf = gdf.merge(metrics, on=["state", "abbr"], how="left")
    gdf = gdf.to_crs("EPSG:5070")

    study = gdf[gdf["delta_rmse_t_ha"].notna()].copy()
    vmax = max(abs(study["delta_rmse_t_ha"].min()), abs(study["delta_rmse_t_ha"].max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mpl.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    gdf[gdf["delta_rmse_t_ha"].isna()].plot(ax=ax, color="#eeeeee", edgecolor="#ffffff", linewidth=0.35)
    study.plot(ax=ax, column="delta_rmse_t_ha", cmap=cmap, norm=norm, edgecolor="#333333", linewidth=0.5)
    for _, row in study.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, f"{row.abbr}\n{int(row.n_locked_rows)}", ha="center", va="center", fontsize=6.5, color="#111111")
    ax.set_axis_off()
    ax.set_title("State-level locked Delta RMSE: Weather-only minus zero residual", fontsize=9, loc="left")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
    cbar.set_label("Delta RMSE t ha$^{-1}$ (negative favors Weather-only)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    for ext in ["pdf", "png"]:
        fig.savefig(PAPER_GEN / f"figure_state_delta_rmse_map.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    payload = {
        "metric": "state-level locked-test Delta RMSE = RMSE(Weather-only) - RMSE(Zero residual)",
        "interpretation": "negative values favor the validation-selected Weather-only model at the point-estimate level",
        "n_states": int(len(metrics)),
        "state_values_sha256": sha256(MAP_DIR / "state_level_locked_delta_rmse.csv"),
        "geometry_zip_sha256": zsha,
        "source_predictions": "artifacts/audit/final_test/seed_aggregated_predictions.csv",
        "figure_pdf": "paper/generated/figure_state_delta_rmse_map.pdf",
    }
    (MAP_DIR / "state_level_locked_delta_rmse_map_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_synthetic_table() -> None:
    runs = pd.read_csv(ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "synthetic_runs_long.csv")
    summary_path = ARTIFACTS / "experiments" / "synthetic-gate-benchmark" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grouped = (
        runs.groupby("scenario")
        .agg(
            gt=("ground_truth_permission", "first"),
            claim=("requested_claim_level", "first"),
            a=("module_a_pass", "mean"),
            b=("module_b_pass", "mean"),
            e=("module_e_pass", "mean"),
            policy=("policy_permit", "mean"),
        )
        .reset_index()
        .sort_values("scenario")
    )
    n_scenarios = int(grouped["scenario"].nunique())
    n_gt_yes = int(grouped["gt"].sum())
    n_gt_no = int((~grouped["gt"]).sum())
    n_valid_runs = int(runs["ground_truth_permission"].sum())
    n_invalid_runs = int((~runs["ground_truth_permission"]).sum())
    assert n_scenarios == 14, n_scenarios
    assert n_gt_yes == 6, n_gt_yes
    assert n_gt_no == 8, n_gt_no
    assert n_valid_runs == 180, n_valid_runs
    assert n_invalid_runs == 240, n_invalid_runs
    assert summary["fp"] == 171, summary["fp"]
    assert summary["fn"] == 20, summary["fn"]
    assert summary["invalid_ground_truth_runs"] == 240, summary["invalid_ground_truth_runs"]
    assert summary["valid_ground_truth_runs"] == 180, summary["valid_ground_truth_runs"]
    assert f"{summary['observable_policy_sensitivity'] * 100:.1f}\\%" == "88.9\\%"
    assert f"{summary['observable_policy_specificity'] * 100:.1f}\\%" == "28.7\\%"
    assert f"{summary['observable_policy_permission_rate'] * 100:.1f}\\%" == "78.8\\%"
    assert summary["policy_uses_gt_or_oracle"] is False

    macro_lines = [
        f"\\newcommand{{\\SyntheticPolicyFalsePermissions}}{{{int(summary['fp'])}/{int(summary['invalid_ground_truth_runs'])}}}",
        f"\\newcommand{{\\SyntheticPolicyFalsePermissionRate}}{{{summary['observable_policy_false_permission_rate'] * 100:.1f}\\%}}",
        f"\\newcommand{{\\SyntheticPolicyFalseAbstentions}}{{{int(summary['fn'])}/{int(summary['valid_ground_truth_runs'])}}}",
        f"\\newcommand{{\\SyntheticPolicyFalseAbstentionRate}}{{{summary['observable_policy_false_abstention_rate'] * 100:.1f}\\%}}",
        f"\\newcommand{{\\SyntheticPolicyPermissionRate}}{{{summary['observable_policy_permission_rate'] * 100:.1f}\\%}}",
        f"\\newcommand{{\\SyntheticPolicySensitivity}}{{{summary['observable_policy_sensitivity'] * 100:.1f}\\%}}",
        f"\\newcommand{{\\SyntheticPolicySpecificity}}{{{summary['observable_policy_specificity'] * 100:.1f}\\%}}",
    ]
    (PAPER_GEN / "synthetic_numbers.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{tabular}{@{}p{0.28\\columnwidth}cccccc@{}}",
        "\\toprule",
        "Scenario & GT & Claim & A & B & E & Policy \\\\",
        "\\midrule",
    ]
    labels = {
        "correlated_features": "Corr. feat.",
        "geographic_shift": "Geo. shift",
        "measurement_error": "Meas. error",
        "omitted_confounder": "Omitted conf.",
        "spatial_resolution_mismatch": "Spatial mismatch",
        "temporal_drift": "Temporal drift",
        "train_only_detrending": "Train-only detr.",
        "imbalanced_tail": "Imbalanced tail",
        "moderate_signal": "Moderate signal",
        "no_signal": "No signal",
        "small_sample": "Small sample",
        "strong_signal": "Strong signal",
        "weak_signal": "Weak signal",
        "leakage": "Leakage",
    }
    for _, row in grouped.iterrows():
        claim = "event" if row["claim"] == "event_recovery" else str(row["claim"]).replace("_", " ")
        lines.append(
            f"{esc(labels.get(row['scenario'], row['scenario']))} & {'yes' if bool(row['gt']) else 'no'} & {esc(claim)} & {pct(row['a'])} & {pct(row['b'])} & {pct(row['e'])} & {pct(row['policy'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (PAPER_GEN / "table_synthetic_scenario_decisions_compact.tex").write_text("\n".join(lines), encoding="utf-8")


def build_validation_table() -> None:
    lines = [
        "\\begin{tabular}{lllrrrccc}",
        "\\toprule",
        "Model & Features & Role & $n$ & RMSE & Seed SD & $P$(rank 1) & 1-SE & Selected \\\\",
        "\\midrule",
        "ExtraTrees (leaf=1) & Weather only & Module A; D & 140 & 0.384 & 0.003 & 0.60 & Yes & Yes \\\\",
        "ExtraTrees (leaf=2) & Weather only & Validation candidate & 140 & 0.385 & 0.001 & 0.20 & No & No \\\\",
        "Random Forest (leaf=1) & Weather only & Validation candidate & 140 & 0.387 & 0.003 & 0.20 & No & No \\\\",
        "Random Forest (leaf=2) & Weather only & Validation candidate & 140 & 0.387 & 0.002 & 0.00 & No & No \\\\",
        "ExtraTrees (leaf=2) & Full & Validation candidate & 140 & 0.389 & 0.003 & 0.00 & No & No \\\\",
        "Random Forest (leaf=1) & Full & Validation candidate & 140 & 0.390 & 0.002 & 0.00 & No & No \\\\",
        "ExtraTrees (leaf=1) & Full & Module B Full & 140 & 0.391 & 0.003 & 0.00 & No & No \\\\",
        "Ridge ($\\alpha=10$) & Metadata only & Best Ridge candidate & 140 & 0.407 & N/A & N/A & No & No \\\\",
        "ExtraTrees (leaf=1) & Metadata only & Module B/D Metadata & 140 & 0.407 & 0.000 & 0.00 & No & No \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    (PAPER_GEN / "table_validation_selection.tex").write_text("\n".join(lines), encoding="utf-8")
    audit = {
        "ridge_row": "Ridge alpha=10 metadata_only",
        "decision": "P(rank 1) set to N/A in the main table because Ridge is deterministic with one validation fit, whereas rank frequencies summarize seed variation for stochastic candidates.",
        "source": "artifacts/audit/selection/validation_model_grid.csv and artifacts/audit_records/validation_stability.csv",
    }
    (ARTIFACTS / "audit_records" / "ridge_rank_display_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def build_supplement_tables() -> None:
    panel = pd.read_csv(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv")
    matrix = (
        panel.groupby(["crop", "region"])["year"]
        .agg(["count", "min", "max"])
        .reset_index()
        .assign(value=lambda x: x["count"].astype(str) + " (" + x["min"].astype(str) + "--" + x["max"].astype(str) + ")")
    )
    crops = sorted(matrix["crop"].unique())
    states = sorted(matrix["region"].unique())
    wide = pd.DataFrame(index=states, columns=crops)
    for _, row in matrix.iterrows():
        wide.loc[row.region, row.crop] = row.value
    wide = wide.fillna("--")
    wide.to_csv(SUPP_DIR / "table_s1_crop_state_availability.csv")

    lines = ["\\begin{tabular}{l" + "c" * len(crops) + "}", "\\toprule", "State & " + " & ".join(map(esc, crops)) + " \\\\", "\\midrule"]
    for state in states:
        lines.append(esc(state) + " & " + " & ".join(esc(wide.loc[state, crop]) for crop in crops) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (PAPER_GEN / "table_s1_crop_state_availability.tex").write_text("\n".join(lines), encoding="utf-8")

    features = pd.read_csv(ARTIFACTS / "data" / "feature_dictionary.csv")
    availability = pd.read_csv(ARTIFACTS / "data" / "feature_availability.csv")
    weather = features.merge(availability[["feature", "available_at_prediction_time"]], on="feature", how="left")
    weather = weather[[
        "feature",
        "driver_group",
        "nasa_field",
        "formula",
        "window",
        "unit",
        "available_at_prediction_time",
        "verification_status",
    ]]
    weather = weather.rename(
        columns={
            "driver_group": "feature_family",
            "nasa_field": "nasa_power_variable",
            "formula": "aggregation",
            "available_at_prediction_time": "availability",
            "verification_status": "missing_handling",
        }
    )
    weather["spatial_rule"] = "state representative coordinate"
    weather["missing_handling"] = weather["missing_handling"].replace(
        {"EXACT_RECONSTRUCTION_PASS": "exact reconstruction; row excluded if required daily input is unavailable"}
    )
    weather = weather[[
        "feature",
        "feature_family",
        "nasa_power_variable",
        "aggregation",
        "window",
        "unit",
        "spatial_rule",
        "missing_handling",
        "availability",
    ]]
    weather.to_csv(SUPP_DIR / "table_s2_weather_feature_definitions.csv", index=False)

    lines = [
        "\\begin{tabular}{p{0.16\\textwidth}p{0.09\\textwidth}p{0.12\\textwidth}p{0.24\\textwidth}p{0.11\\textwidth}p{0.07\\textwidth}p{0.12\\textwidth}p{0.17\\textwidth}p{0.12\\textwidth}}",
        "\\toprule",
        "Feature & Family & POWER field & Aggregation & Window & Unit & Spatial rule & Missing handling & Availability \\\\",
        "\\midrule",
    ]
    for _, row in weather.iterrows():
        lines.append(
            f"{esc(row.feature)} & {esc(row.feature_family)} & {esc(row.nasa_power_variable)} & "
            f"{esc(row.aggregation)} & {esc(row.window)} & {esc(row.unit)} & "
            f"{esc(row.spatial_rule)} & {esc(row.missing_handling)} & {esc(row.availability)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (PAPER_GEN / "table_s2_weather_feature_definitions.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    build_claim_module_definition_table()
    build_gate_module_table()
    draw_workflow()
    draw_xai_figure()
    draw_state_map()
    build_synthetic_table()
    build_validation_table()
    build_supplement_tables()
    print("Built final-round visuals/tables from existing artifacts")


if __name__ == "__main__":
    main()
