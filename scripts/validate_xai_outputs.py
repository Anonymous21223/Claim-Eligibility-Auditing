from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_yield_xai.core import EXPECTED_DRIVER_GROUPS, is_leakage_feature, make_project_paths  # noqa: E402


EXPECTED_CSV = [
    "anomaly_scores_all_rows.csv",
    "low_yield_anomalies.csv",
    "dataset_summary.csv",
    "driver_groups.csv",
    "driver_group_features.csv",
    "model_performance.csv",
    "xai_methods.csv",
    "method_settings.csv",
    "shap_feature_ranking.csv",
    "shap_local_anomaly_values.csv",
    "grouped_shap_summary.csv",
    "group_permutation_importance.csv",
    "group_ablation_importance.csv",
    "lime_event_explanations.csv",
    "lime_status.csv",
    "ale_curves.csv",
    "event_method_top_drivers.csv",
    "method_agreement_matrix.csv",
    "method_driver_rankings.csv",
    "representative_event_explanations.csv",
    "anomaly_threshold_sensitivity.csv",
    "detrending_robustness.csv",
    "event_year_sanity.csv",
]
EXPECTED_FIGURES = [
    "fig01_method_workflow.png",
    "fig02_anomaly_timeline.png",
    "fig03_shap_summary.png",
    "fig04_grouped_shap.png",
    "fig05_group_importance.png",
    "fig07_ale_curves.png",
    "fig08_method_agreement.png",
]
BANNED_MAIN_TEXT = re.compile(r"\b(counterfactual|SCAA|recoverable|recovery)\b", re.IGNORECASE)
BANNED_PUBLIC_TEXT = re.compile(
    r"(local median|local group resampling|median replacement|support CSV|Generated support|stress score)",
    re.IGNORECASE,
)
BANNED_PUBLIC_METHODS = {"local_median_replacement", "local_permutation", "local group resampling", "local median replacement"}
MISLEADING_RANK_TEXT = re.compile(r"(rank agreement|agreement matrix|pairwise agreement)", re.IGNORECASE)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def assert_files(paths) -> None:
    missing = [str(paths.xai_outputs / name) for name in EXPECTED_CSV if not (paths.xai_outputs / name).exists()]
    missing += [str(paths.xai_figures / name) for name in EXPECTED_FIGURES if not (paths.xai_figures / name).exists()]
    if missing:
        raise AssertionError(f"Missing required XAI files: {missing}")
    forbidden_files = [
        paths.xai_outputs / "local_group_diagnostics.csv",
        paths.xai_figures / "fig09_event_explanations.png",
        paths.xai_figures / "fig09_event_explanations.pdf",
    ]
    present_forbidden = [str(path) for path in forbidden_files if path.exists()]
    if present_forbidden:
        raise AssertionError(f"Forbidden local/public files remain: {present_forbidden}")
    for figure in EXPECTED_FIGURES:
        path = paths.xai_figures / figure
        img = plt.imread(path)
        if img.size == 0 or float(np.std(img)) == 0.0:
            raise AssertionError(f"Blank figure: {path}")
        if img.shape[0] < 400 or img.shape[1] < 500:
            raise AssertionError(f"Figure resolution is too small: {path} has shape {img.shape}")


def assert_data_integrity(paths) -> None:
    scored = read_csv(paths.xai_outputs / "anomaly_scores_all_rows.csv")
    anomalies = read_csv(paths.xai_outputs / "low_yield_anomalies.csv")
    features = read_csv(paths.xai_outputs / "driver_group_features.csv")
    if len(scored) != 1257:
        raise AssertionError(f"Expected 1257 scored rows, found {len(scored)}")
    if len(anomalies) != 214:
        raise AssertionError(f"Expected 214 anomaly rows, found {len(anomalies)}")
    if len(features) != 35:
        raise AssertionError(f"Expected 35 driver-group features, found {len(features)}")
    groups = set(features["driver_group"])
    if groups != set(EXPECTED_DRIVER_GROUPS):
        raise AssertionError(f"Unexpected driver groups: {sorted(groups)}")
    bad_features = [feature for feature in features["feature"] if is_leakage_feature(str(feature))]
    if bad_features:
        raise AssertionError(f"Leakage-like features in driver table: {bad_features}")


def assert_method_outputs(paths) -> None:
    metrics = read_csv(paths.xai_outputs / "model_performance.csv")
    numeric_cols = ["r2", "rmse_t_ha", "mae_t_ha"]
    if metrics.empty or not np.isfinite(metrics[numeric_cols].to_numpy(dtype=float)).all():
        raise AssertionError("Model performance metrics must be finite")

    shap = read_csv(paths.xai_outputs / "shap_feature_ranking.csv")
    if shap.empty or shap["mean_abs_shap_all"].max() <= 0:
        raise AssertionError("SHAP feature ranking is empty or uninformative")

    for name, score_col in [
        ("group_permutation_importance.csv", "rmse_increase_t_ha"),
        ("group_ablation_importance.csv", "rmse_increase_t_ha"),
    ]:
        table = read_csv(paths.xai_outputs / name)
        if table.empty:
            raise AssertionError(f"{name} is empty")
        if not set(table["driver_group"]).issubset(set(EXPECTED_DRIVER_GROUPS)):
            raise AssertionError(f"{name} contains unexpected driver groups")
        if not np.isfinite(table[score_col].to_numpy(dtype=float)).all():
            raise AssertionError(f"{name} contains non-finite scores")

    votes = read_csv(paths.xai_outputs / "event_method_top_drivers.csv")
    if votes.empty or votes["event_key"].nunique() < 214:
        raise AssertionError("Event method votes must cover all anomaly events")
    if not set(votes["top_driver_group"]).issubset(set(EXPECTED_DRIVER_GROUPS)):
        raise AssertionError("Event votes contain unexpected top driver groups")
    if set(votes["method"]).intersection(BANNED_PUBLIC_METHODS):
        raise AssertionError("Event votes still contain local replacement/resampling methods")

    agreement = read_csv(paths.xai_outputs / "method_agreement_matrix.csv")
    required_agreement = {"scope", "n_methods", "consensus_driver", "rank_1_votes", "methods_supporting", "interpretation"}
    if agreement.empty or not required_agreement.issubset(set(agreement.columns)):
        raise AssertionError("Method agreement table is incomplete")
    if not {"global", "anomaly"}.issubset(set(agreement["scope"])):
        raise AssertionError("Method agreement must include global and anomaly scopes")

    ale = read_csv(paths.xai_outputs / "ale_curves.csv")
    if ale.empty or ale["feature"].nunique() < 6:
        raise AssertionError("ALE curves must cover the six planned weather features")

    representative = read_csv(paths.xai_outputs / "representative_event_explanations.csv")
    if representative.empty or not {2012, 2021, 2022}.issubset(set(representative["year"].astype(int))):
        raise AssertionError("Representative event explanations must include 2012, 2021, and 2022")

    settings = read_csv(paths.xai_outputs / "method_settings.csv")
    required_settings = {"random_state", "xai_n_estimators", "permutation_repeats", "lime_selection_rule"}
    if settings.empty or not required_settings.issubset(set(settings["setting"])):
        raise AssertionError("Method settings metadata is incomplete")
    if "local_reference_samples" in set(settings["setting"]):
        raise AssertionError("Method settings still expose local reference sampling")

    methods = read_csv(paths.xai_outputs / "xai_methods.csv")
    if set(methods.get("method", pd.Series(dtype=str)).astype(str).str.lower()).intersection(BANNED_PUBLIC_METHODS):
        raise AssertionError("Public method descriptions still include local replacement/resampling diagnostics")

    rankings = read_csv(paths.xai_outputs / "method_driver_rankings.csv")
    grouped = rankings[rankings["method"].isin(["grouped_shap_abs", "grouped_shap_anomaly_abs"])].copy()
    expected_n = {
        ("global", "grouped_shap_abs"): 343,
        ("anomaly", "grouped_shap_anomaly_abs"): 67,
    }
    for (scope, method), n_eval in expected_n.items():
        subset = grouped[(grouped["scope"] == scope) & (grouped["method"] == method)]
        if subset.empty or set(subset["n_eval"].astype(int)) != {n_eval}:
            raise AssertionError(f"{method} {scope} must use n_eval={n_eval}")

    lime_status = read_csv(paths.xai_outputs / "lime_status.csv")
    if lime_status.empty or "status" not in lime_status:
        raise AssertionError("LIME status output is missing")


def assert_paper_text(paths) -> None:
    for path in [paths.latex / "main.tex", paths.root / "README.md", paths.root / "paper" / "REPRODUCIBILITY.md"]:
        if not path.exists():
            raise AssertionError(f"Missing text file for wording check: {path}")
        text = path.read_text(encoding="utf-8")
        hits = BANNED_MAIN_TEXT.findall(text)
        if hits:
            raise AssertionError(f"Old framing terms remain in {path}: {sorted(set(hits))}")
        public_hits = BANNED_PUBLIC_TEXT.findall(text)
        if public_hits:
            raise AssertionError(f"Forbidden public wording remains in {path}: {sorted(set(public_hits))}")
        misleading_hits = MISLEADING_RANK_TEXT.findall(text)
        if misleading_hits:
            raise AssertionError(f"Misleading rank wording remains in {path}: {sorted(set(misleading_hits))}")
        if "supplement" in text.lower():
            raise AssertionError(f"Supplement wording remains in {path}")
    if (paths.latex / "supplement.tex").exists():
        raise AssertionError("supplement.tex should not exist in the main-only paper")
    if paths.paper_supplement.exists() and any(paths.paper_supplement.glob("*.tex")):
        raise AssertionError("Supplement TeX tables should not exist in the main-only paper")


def main() -> None:
    paths = make_project_paths(PROJECT_ROOT)
    assert_files(paths)
    assert_data_integrity(paths)
    assert_method_outputs(paths)
    assert_paper_text(paths)
    print("XAI output validation passed.")


if __name__ == "__main__":
    main()
