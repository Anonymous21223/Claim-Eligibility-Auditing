from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_yield_xai.core import EXPECTED_DRIVER_GROUPS, make_project_paths  # noqa: E402


PATHS = make_project_paths(ROOT)

METHOD_LABELS = {
    "group_ablation": "ablation",
    "group_permutation": "permutation",
    "grouped_shap": "grouped SHAP",
    "grouped_shap_abs": "grouped SHAP",
    "grouped_shap_anomaly_abs": "grouped SHAP",
    "lime_selected": "LIME cases",
}


FIGURE_MAP = {
    "fig01_method_workflow.png": "fig01_method_workflow.png",
    "fig02_anomaly_timeline.png": "fig02_anomaly_timeline.png",
    "fig03_shap_summary.png": "fig03_shap_summary.png",
    "fig04_grouped_shap.png": "fig04_grouped_shap.png",
    "fig05_group_importance.png": "fig05_group_importance.png",
    "fig07_ale_curves.png": "fig07_ale_curves.png",
    "fig08_method_agreement.png": "fig08_method_agreement.png",
}


def ensure_dirs() -> None:
    for folder in [PATHS.paper_figures, PATHS.paper_tables, PATHS.paper_table_csv]:
        folder.mkdir(parents=True, exist_ok=True)
    if (PATHS.latex / "supplement.tex").exists():
        (PATHS.latex / "supplement.tex").unlink()
    if PATHS.paper_supplement.exists():
        shutil.rmtree(PATHS.paper_supplement)
    for folder, patterns in [
        (PATHS.paper_figures, ["fig*.png"]),
        (PATHS.paper_tables, ["table*.tex"]),
        (PATHS.paper_table_csv, ["*.csv"]),
    ]:
        for pattern in patterns:
            for path in folder.glob(pattern):
                path.unlink()


def read_csv(name: str) -> pd.DataFrame:
    path = PATHS.xai_outputs / name
    if not path.exists():
        raise AssertionError(f"Missing XAI input: {path}")
    return pd.read_csv(path)


def latex_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    placement: str = "!htbp",
    *,
    column_spec: str | None = None,
    use_tabularx: bool = False,
    font_size: str | None = None,
    tabcolsep: str | None = None,
    arraystretch: str | None = None,
) -> str:
    columns = list(df.columns)
    align = "l" * len(columns)
    lines = [
        rf"\begin{{table}}[{placement}]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
    ]
    if column_spec is None:
        lines.extend(
            [
                r"\resizebox{\linewidth}{!}{%",
                rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
            ]
        )
        table_env = "tabular"
        needs_resize_close = True
        needs_group_close = False
    else:
        table_env = "tabularx" if use_tabularx else "tabular"
        needs_resize_close = False
        needs_group_close = True
        lines.append(r"\begingroup")
        if font_size:
            lines.append(rf"\{font_size}")
        if tabcolsep:
            lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
        if arraystretch:
            lines.append(rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}")
        if use_tabularx:
            lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{column_spec}}}")
        else:
            lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.extend(
        [
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
        ]
    )
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[column]) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", rf"\end{{{table_env}}}"])
    if needs_resize_close:
        lines.append(r"}")
    if needs_group_close:
        lines.append(r"\endgroup")
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)


def write_table(df: pd.DataFrame, name: str, caption: str, label: str, **table_kwargs: object) -> None:
    target_dir = PATHS.paper_tables
    csv_name = name.replace(".tex", ".csv")
    df.to_csv(PATHS.paper_table_csv / csv_name, index=False)
    (target_dir / name).write_text(latex_table(df, caption, label, **table_kwargs), encoding="utf-8")


def write_support_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(PATHS.paper_table_csv / name, index=False)


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def compact_level(value: object) -> str:
    mapping = {
        "all rows": "eval rows",
        "forward-time all rows": "eval rows",
        "anomaly rows": "eval anomalies",
        "forward-time anomaly rows": "eval anomalies",
        "selected events": "LIME cases",
    }
    return mapping.get(str(value), str(value))


RANKING_TABLE_SPEC = (
    r"@{}"
    r">{\raggedright\arraybackslash}p{0.08\linewidth}"
    r">{\raggedright\arraybackslash}p{0.12\linewidth}"
    r">{\centering\arraybackslash}p{0.04\linewidth}"
    r">{\raggedright\arraybackslash}p{0.13\linewidth}"
    r">{\raggedright\arraybackslash}X"
    r">{\raggedright\arraybackslash}X"
    r">{\raggedright\arraybackslash}X"
    r"@{}"
)

AGREEMENT_TABLE_SPEC = (
    r"@{}"
    r">{\raggedright\arraybackslash}p{0.08\linewidth}"
    r">{\centering\arraybackslash}p{0.08\linewidth}"
    r">{\raggedright\arraybackslash}p{0.13\linewidth}"
    r">{\raggedright\arraybackslash}p{0.09\linewidth}"
    r">{\raggedright\arraybackslash}X"
    r">{\raggedright\arraybackslash}X"
    r"@{}"
)


def display_driver(value: object) -> str:
    return str(value).replace("_", " ")


def ranking_table(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ["global", "anomaly"]:
        sub = rankings[rankings["scope"] == scope].copy()
        for method, group in sub.groupby("method", sort=True):
            ordered = group.sort_values("rank").head(3)
            first = ordered.iloc[0]
            row: dict[str, object] = {
                "scope": scope,
                "method": METHOD_LABELS.get(method, method),
                "n": int(first["n_eval"]) if "n_eval" in ordered.columns and not pd.isna(first["n_eval"]) else "",
                "level": compact_level(first["level"]) if "level" in ordered.columns else "",
            }
            for i, item in enumerate(ordered.itertuples(index=False), start=1):
                row[f"rank {i}"] = f"{display_driver(item.driver_group)} ({fmt_float(item.score, 4)})"
            rows.append(row)
    return pd.DataFrame(rows)


def build_tables() -> None:
    dataset = read_csv("dataset_summary.csv").rename(columns={"low_yield_anomalies": "anomalies"})
    write_table(dataset, "table01_dataset_summary.tex", "Dataset summary by crop.", "tab:dataset_summary")

    drivers = read_csv("driver_groups.csv")
    write_table(
        drivers,
        "table02_driver_groups.tex",
        "Extreme-weather driver groups used in the multi-method XAI analysis.",
        "tab:driver_groups",
    )

    performance = read_csv("model_performance.csv")
    performance = performance[
        (
            (performance["model"] == "ExtraTrees yield")
            & performance["protocol"].isin(
                [
                    "forward_time_train_1990_2015_test_2016_2021",
                    "forward_time_train_1990_2018_test_2019_2025",
                ]
            )
        )
        | (
            (performance["model"] == "ExtraTrees residual")
            & (performance["protocol"] == "forward_time_train_1990_2015_test_2016_2025")
        )
    ].copy()
    performance = performance[["model", "target", "protocol", "scope", "n_test", "r2", "rmse_t_ha", "mae_t_ha"]].copy()
    for column in ["r2", "rmse_t_ha", "mae_t_ha"]:
        performance[column] = performance[column].map(lambda value: fmt_float(value, 3))
    write_table(
        performance,
        "table03_model_performance.tex",
        "Forward-time yield and residual model performance.",
        "tab:model_performance",
    )

    methods = read_csv("xai_methods.csv")
    write_table(methods, "table04_xai_methods.tex", "Explanation methods compared in the analysis.", "tab:xai_methods")
    settings = read_csv("method_settings.csv")
    write_support_csv(settings, "support_method_settings.csv")

    rankings = read_csv("method_driver_rankings.csv")
    write_table(
        ranking_table(rankings),
        "table05_driver_rankings.tex",
        "Global and anomaly-focused driver-group rankings across explanation methods.",
        "tab:driver_rankings",
        placement="H",
        column_spec=RANKING_TABLE_SPEC,
        use_tabularx=True,
        font_size="footnotesize",
        tabcolsep="2pt",
        arraystretch="1.02",
    )

    agreement = read_csv("method_agreement_matrix.csv")
    agreement = agreement[["scope", "n_methods", "consensus_driver", "rank_1_votes", "methods_supporting", "interpretation"]].copy()
    agreement = agreement.rename(
        columns={
            "n_methods": "n methods",
            "consensus_driver": "consensus driver",
            "rank_1_votes": "rank-1 votes",
            "methods_supporting": "supporting methods",
        }
    )
    write_table(
        agreement,
        "table07_method_agreement.tex",
        "Scale-free rank-one support summary across heterogeneous explanation methods.",
        "tab:method_agreement",
        column_spec=AGREEMENT_TABLE_SPEC,
        use_tabularx=True,
        font_size="small",
        tabcolsep="3pt",
        arraystretch="1.12",
    )

    events = read_csv("representative_event_explanations.csv")
    events = events[
        [
            "crop",
            "region",
            "year",
            "trend_residual_z",
            "grouped_shap_driver",
            "lime_driver",
            "drivers_agree",
        ]
    ].copy()
    events = events.rename(columns={"region": "state"})
    events["year"] = events["year"].astype(int)
    events["trend_residual_z"] = events["trend_residual_z"].map(lambda value: fmt_float(value, 2))
    write_table(
        events,
        "table08_event_explanations.tex",
        "Six selected event-level method-sensitivity checks using grouped SHAP and LIME.",
        "tab:event_explanations",
    )

    features = read_csv("driver_group_features.csv")
    feature_table = (
        features.groupby("driver_group", as_index=False)
        .agg(features=("feature", lambda values: ", ".join(values)))
        .sort_values("driver_group")
    )
    write_support_csv(feature_table, "support_driver_group_features.csv")

    threshold = read_csv("anomaly_threshold_sensitivity.csv")
    for column in ["share_of_dataset", "jaccard_with_z_minus_1"]:
        threshold[column] = threshold[column].map(lambda value: fmt_float(value, 3))
    write_support_csv(threshold, "support_anomaly_threshold_sensitivity.csv")

    detrending = read_csv("detrending_robustness.csv")
    detrending["jaccard_with_linear"] = detrending["jaccard_with_linear"].map(lambda value: fmt_float(value, 3))
    write_support_csv(detrending, "support_detrending_robustness.csv")
    threshold_raw = read_csv("anomaly_threshold_sensitivity.csv")
    detrending_raw = read_csv("detrending_robustness.csv")
    robustness_rows = []
    for _, row in threshold_raw.iterrows():
        threshold_text = f"z below {float(row['threshold']):.1f}"
        if float(row["threshold"]) == -1.0:
            interpretation = "Default broad anomaly set"
        elif float(row["threshold"]) == -1.5:
            interpretation = "Stricter low-yield subset"
        else:
            interpretation = "Most severe low-yield subset"
        robustness_rows.append(
            {
                "check": "Threshold",
                "setting": threshold_text,
                "result": f"{int(row['anomaly_count'])} anomalies",
                "interpretation": interpretation,
            }
        )
    rolling = detrending_raw[detrending_raw["detrending_method"] == "centered_7_year_rolling"].iloc[0]
    robustness_rows.append(
        {
            "check": "Detrending",
            "setting": "centered 7-year rolling",
            "result": f"{int(rolling['anomaly_count'])} anomalies; Jaccard {float(rolling['jaccard_with_linear']):.3f}",
            "interpretation": "Moderate overlap with linear detrending",
        }
    )
    write_table(
        pd.DataFrame(robustness_rows),
        "table09_robustness_checks.tex",
        "Compact anomaly-screening robustness checks.",
        "tab:robustness_checks",
    )

    sanity = read_csv("event_year_sanity.csv")
    sanity["match_rate"] = sanity["match_rate"].map(lambda value: fmt_float(value, 3))
    write_support_csv(sanity, "support_event_year_sanity.csv")

    lime_status = read_csv("lime_status.csv")
    write_support_csv(lime_status, "support_lime_status.csv")


def copy_figures() -> None:
    for source_name, target_name in FIGURE_MAP.items():
        source = PATHS.xai_figures / source_name
        target = PATHS.paper_figures / target_name
        if not source.exists():
            raise AssertionError(f"Missing XAI figure: {source}")
        shutil.copy2(source, target)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    files = [
        ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv",
        ROOT / "data" / "raw" / "nasa_power_daily.zip",
        ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv",
        PATHS.xai_outputs / "low_yield_anomalies.csv",
        PATHS.xai_outputs / "model_performance.csv",
        PATHS.xai_outputs / "method_settings.csv",
        PATHS.xai_outputs / "method_driver_rankings.csv",
        PATHS.xai_outputs / "method_agreement_matrix.csv",
        PATHS.overleaf_zip,
    ]
    lines = [
        "# Data Manifest",
        "",
        "Generated by `python scripts/build_paper_assets.py`.",
        "",
        "## Dataset Checks",
        "",
        "- Processed frame rows: 1257",
        "- Year range: 1990-2025",
        "- Crops: Barley, Canola, Oats, Wheat",
        "- Low-yield anomalies: 214",
        "- Full-season weather features: 35",
        "",
        "## File Checksums",
        "",
        "| Path | Bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for path in files:
        rel = path.relative_to(ROOT)
        if path.exists():
            lines.append(f"| `{rel}` | {path.stat().st_size} | `{sha256(path)}` |")
        else:
            lines.append(f"| `{rel}` | pending | pending |")
    PATHS.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_outputs() -> None:
    expected_tables = [
        "table01_dataset_summary.tex",
        "table02_driver_groups.tex",
        "table03_model_performance.tex",
        "table04_xai_methods.tex",
        "table05_driver_rankings.tex",
        "table07_method_agreement.tex",
        "table08_event_explanations.tex",
        "table09_robustness_checks.tex",
    ]
    missing = [str(PATHS.paper_tables / name) for name in expected_tables if not (PATHS.paper_tables / name).exists()]
    missing += [
        str(PATHS.paper_figures / name)
        for name in FIGURE_MAP.values()
        if not (PATHS.paper_figures / name).exists()
    ]
    if (PATHS.latex / "supplement.tex").exists():
        missing.append("supplement.tex should not exist for the main-only paper package")
    if PATHS.paper_supplement.exists() and any(PATHS.paper_supplement.iterdir()):
        missing.append("supplement directory should be absent or empty")
    if missing:
        raise AssertionError(f"Missing paper assets: {missing}")
    for figure in FIGURE_MAP.values():
        img = plt.imread(PATHS.paper_figures / figure)
        if img.size == 0 or float(np.std(img)) == 0.0:
            raise AssertionError(f"Blank paper figure: {figure}")


def main() -> None:
    ensure_dirs()
    build_tables()
    copy_figures()
    write_manifest()
    assert_outputs()
    print("Paper assets built.")
    print(f"Figures: {PATHS.paper_figures}")
    print(f"Tables: {PATHS.paper_tables}")


if __name__ == "__main__":
    main()
