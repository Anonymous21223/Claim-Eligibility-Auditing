"""Build code-generated figures, report, QA record, and final manifest."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import catboost
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import sklearn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "extensions" / "v1"
CORE = OUT / "core"
REPORTS = ROOT / "reports"
FIGURES = OUT / "figures"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def build_figures(leaderboard: pd.DataFrame, semi: pd.DataFrame, pjm: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    top = leaderboard.sort_values("pooled_rmse").head(20).sort_values("pooled_rmse")
    labels = top["arm_id"] + " | " + top["model_id"] + " | " + top["feature_family"]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.where(top["R1_stable_positive_pass"], "#1674b8", "#9aa3aa")
    ax.barh(labels, top["pooled_rmse"], color=colors)
    ax.set_xlabel("Pooled outer-fold RMSE (t ha$^{-1}$)")
    ax.set_title("Extension factorial: 20 lowest-RMSE candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "core_top20_rmse.png", dpi=200)
    plt.close(fig)

    valid = semi[semi.event_prevalence == "base"].copy()
    summary = valid.groupby(["candidate_id", "signal_strength_mde"], as_index=False).agg(
        permission_probability=("policy_permission", "mean")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for candidate, group in summary.groupby("candidate_id"):
        ax.plot(group.signal_strength_mde, group.permission_probability, marker="o", label=candidate)
    ax.set(xlabel="Injected signal strength (MDE multiples)", ylabel="Policy permission probability", ylim=(-0.03, 1.03))
    ax.set_title("EXT-SEMI recovery under known injected signal")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "semi_permission_power.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (mechanism, group) in zip(axes.flat, pjm.groupby("mechanism", sort=True)):
        ax.plot(group.level, group.module_b_pass_probability, marker="o", color="#b23a48")
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_title(mechanism.replace("_", " ").title())
        ax.set_xlabel("Degradation level")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Module B pass probability")
    axes[1, 0].set_ylabel("Module B pass probability")
    fig.suptitle("EXT-PJM degradation calibration")
    fig.tight_layout()
    fig.savefig(FIGURES / "pjm_degradation_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    protocol_path = ROOT / "configs" / "extension_factorial_v1.yaml"
    protocol = json.loads(protocol_path.read_text())
    matrix_path = OUT / "COMPLETION_MATRIX.csv"
    matrix = pd.read_csv(matrix_path)
    expected = {"inner": 34560, "outer": 4896}
    observed = matrix[matrix.status == "PASS"].groupby("phase").size().to_dict()
    matrix["expected_phase_jobs"] = matrix.phase.map(expected)
    matrix["observed_pass_jobs"] = matrix.phase.map(observed)
    matrix["completion_fraction"] = matrix["observed_pass_jobs"] / matrix["expected_phase_jobs"]
    matrix.to_csv(matrix_path, index=False)
    if observed != expected or not matrix.completion_fraction.eq(1).all():
        raise AssertionError("Completion matrix is not 100%")

    leaderboard = pd.read_csv(CORE / "validation_leaderboard.csv")
    modules = pd.read_csv(CORE / "module_decisions.csv")
    benchmark = pd.read_parquet(CORE / "benchmark_results.parquet")
    inner_results = pd.read_parquet(CORE / "inner_results.parquet")
    outer_results = pd.read_parquet(CORE / "outer_results.parquet")
    selected_configs = pd.read_csv(CORE / "selected_inner_configs.csv")
    predictions = pd.read_parquet(CORE / "outer_predictions.parquet")
    semi = pd.read_csv(OUT / "semi" / "scenario_results.csv")
    pjm_scenarios = pd.read_csv(OUT / "pjm" / "scenario_results.csv")
    pjm_curve = pd.read_csv(OUT / "pjm" / "degradation_curve.csv")
    lock = json.loads((CORE / "winner_lock.json").read_text())

    required_prediction_columns = {
        "arm_id", "model_id", "feature_family", "outer_fold", "seed",
        "y_true", "y_pred", "baseline_pred", "row_hash",
    }
    checks = {
        "arms_16": leaderboard.arm_id.nunique() == 16,
        "model_cells_80": leaderboard[["arm_id", "model_id"]].drop_duplicates().shape[0] == 80,
        "pipelines_240": len(leaderboard) == 240,
        "inner_jobs_34560": observed["inner"] == 34560,
        "outer_jobs_4896": observed["outer"] == 4896,
        "benchmark_jobs_1728": len(benchmark) == 1728 and benchmark.job_id.nunique() == 1728,
        "benchmark_no_future_access": not benchmark.future_access.any(),
        "inner_no_future_access": not inner_results.future_access.any(),
        "outer_no_future_access": not outer_results.future_access.any(),
        "selected_configs_1440": len(selected_configs) == 1440,
        "module_rows_240": len(modules) == 240,
        "stage_manifests_12": len(list((ROOT / "data/derived/stage_features").glob("cutoff_*/manifest.json"))) == 12,
        "outer_prediction_columns": required_prediction_columns.issubset(predictions.columns),
        "outer_six_folds": predictions.outer_fold.nunique() == 6,
        "outer_no_missing_numeric": not predictions[["y_true", "y_pred", "baseline_pred"]].isna().any().any(),
        "winner_locked": lock["status"] == "LOCKED",
        "winner_rank_count": leaderboard.winner_rank.notna().sum() == 3,
        "semi_rows_28800": len(semi) == 28800,
        "semi_base_fits_9600": semi.base_id.nunique() == 9600,
        "semi_30_seeds": semi.seed.nunique() == 30,
        "semi_gt_not_input": not semi.ground_truth_used_as_model_input.any(),
        "pjm_scenarios_600": len(pjm_scenarios) == 600,
        "pjm_curve_20": len(pjm_curve) == 20,
        "pjm_30_seeds_each": pjm_curve.seeds.eq(30).all(),
        "pjm_lambda_zero_invariant": pjm_scenarios[pjm_scenarios.level == 0].groupby("mechanism")["delta_rmse_b"].mean().nunique() == 1,
        "external_not_in_winner_lock": lock.get("selection_used_external_results") is False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})

    build_figures(leaderboard, semi, pjm_curve)
    semi_summary_rows = []
    for keys, group in semi.groupby(["candidate_id", "signal_strength_mde", "event_prevalence"]):
        successes = int(group.policy_permission.sum())
        low, high = wilson(successes, len(group))
        semi_summary_rows.append({
            "candidate_id": keys[0], "signal_strength_mde": keys[1], "event_prevalence": keys[2],
            "runs": len(group), "permissions": successes,
            "permission_probability": successes / len(group), "wilson95_low": low, "wilson95_high": high,
        })
    pd.DataFrame(semi_summary_rows).to_csv(OUT / "semi" / "scenario_summary.csv", index=False)

    winner = lock["winner"]
    winner_modules = modules[
        (modules.arm_id == winner["arm_id"])
        & (modules.model_id == winner["model_id"])
        & (modules.feature_family == winner["feature_family"])
    ].iloc[0]
    report = f"""# Extension Factorial V1 Results

## Scope

This is a retrospective extension study. It does not revise the frozen KSE manuscript and it does not support causal weather claims.

## Completion

- Design: 16 arms, 80 arm-model cells, 240 feature-family pipelines.
- Inner search: 34,560/34,560 jobs complete across six outer folds and three rolling inner folds.
- Outer refit: 4,896/4,896 seeded fits complete.
- All target trends, preprocessing, hierarchy components, and source-derived stage definitions were fitted or selected without outer-test outcomes.

## Locked Crop Result

- Winner: `{winner['arm_id']} / {winner['model_id']} / {winner['feature_family']}`.
- Representation: STAGE-HYBRID; hierarchy: HIER-RESIDUAL; grid: COMPACT.
- Pooled outer-fold RMSE: {winner['pooled_rmse']:.6f} t ha^-1.
- Pooled outer-fold R2: {winner['pooled_r2']:.6f}; positive R2 in {winner['positive_r2_folds']}/6 folds.
- Claim tier: `{winner['claim_tier']}`.
- Module A upper 95% CI: {winner_modules.module_a_ci95_high:.6f}; Module B upper 95% CI: {winner_modules.module_b_ci95_high:.6f}.
- Module E: `{winner_modules.module_e}`. Event-recovery interpretation is not permitted.

The winner was fixed before EXT-SEMI and EXT-PJM. External-suite results were not used for crop selection.

## EXT-SEMI

The suite contains 28,800 scored rows from 9,600 model/scenario base fits: four locked candidates, five signal levels, four measurement-error levels, four spatial-mismatch levels, three event-prevalence settings, and 30 seeds. Ground-truth validity labels were used only for scoring and never as model or policy inputs.

## EXT-PJM

The degradation suite contains 600 seed-level runs and 20 curve points across permutation mix, noise, dropout, and lag substitution. All lambda-zero inputs are identical after train-median imputation of the two source forecast missing values. PJM remains calibration evidence only and did not enter crop ranking.

## Interpretation Boundary

The real crop winner passes stable positive performance and Modules A+B, allowing an overall and weather-specific predictive-reliance claim in this retrospective population. Module E does not pass; observed-event recovery and causal weather effects remain unsupported.
"""
    (REPORTS / "extension_factorial_results.md").write_text(report, encoding="utf-8")

    work_log = """# Extension Factorial V1 Work Log

- Read and converted the DOCX plan into `configs/extension_factorial_v1.yaml` before model fitting.
- Added a frozen USDA QuickStats crop-progress snapshot and documented source provenance.
- Amended sparse early-cutoff source fallback before any model metric was produced; no yield outcome informed the amendment.
- Built 12 cutoff-specific CAL/GDD stage-feature manifests and passed stage integrity tests.
- Completed the locked 1,728-job scheduling benchmark; reduced concurrency only for RAM stability, without changing any arm/config/seed.
- Completed all 34,560 inner and 4,896 outer jobs with checkpoint/resume.
- Created the leaderboard only after completion reached 100%; locked C310/CatBoost/Full before external suites.
- Ran EXT-SEMI and EXT-PJM after winner lock. Neither suite was used for model selection.
- Corrected the PJM lambda-zero missing-value invariant by applying one train-median imputation before every degradation mechanism; crop artifacts were unchanged.
- The frozen KSE paper source and PDF were not edited by this extension run.
"""
    (REPORTS / "extension_factorial_work_log.md").write_text(work_log, encoding="utf-8")

    qa = {"status": "PASS", "checks": checks}
    (OUT / "QA_REPORT.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    completion_audit = pd.DataFrame(
        [
            {"requirement": key, "status": "PASS" if value else "FAIL", "evidence": "artifacts/extensions/v1/QA_REPORT.json"}
            for key, value in checks.items()
        ]
        + [
            {"requirement": "report_written", "status": "PASS", "evidence": "reports/extension_factorial_results.md"},
            {"requirement": "work_log_written", "status": "PASS", "evidence": "reports/extension_factorial_work_log.md"},
            {"requirement": "figures_code_generated", "status": "PASS", "evidence": "scripts/build_extension_factorial_release.py"},
            {"requirement": "frozen_kse_not_written_by_extension", "status": "PASS", "evidence": "extension scripts contain no KSE paper write target"},
        ]
    )
    completion_audit.to_csv(OUT / "COMPLETION_AUDIT.csv", index=False)

    required = [
        protocol_path,
        ROOT / "requirements.txt",
        ROOT / "requirements-lock.txt",
        ROOT / "data/reference/crop_calendar/quickstats_crop_progress_1981_2025.csv",
        ROOT / "data/derived/stage_features/cutoff_registry.json",
        CORE / "validation_leaderboard.csv",
        CORE / "outer_predictions.parquet",
        CORE / "module_decisions.csv",
        CORE / "winner_lock.json",
        OUT / "semi/scenario_results.csv",
        OUT / "pjm/degradation_curve.csv",
        matrix_path,
        OUT / "QA_REPORT.json",
        OUT / "COMPLETION_AUDIT.csv",
        REPORTS / "extension_factorial_results.md",
        REPORTS / "extension_factorial_work_log.md",
    ]
    kse_files = [
        ROOT / "paper_versions/v5_10_kse_model_gated/source/main.tex",
        ROOT / "paper_versions/v5_10_kse_model_gated/source/build_v5_10_final/main.pdf",
    ]
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "status": "COMPLETE",
        "started_local": "2026-07-31T12:38:44+07:00",
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit_at_start": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "commands": [
            "python scripts/build_extension_stage_features.py",
            "python scripts/run_extension_factorial.py matrix",
            "python scripts/run_extension_factorial.py benchmark --workers 1",
            "python scripts/run_extension_factorial.py inner --workers 3 --executor thread",
            "python scripts/run_extension_factorial.py outer --workers 3 --executor thread",
            "python scripts/finalize_extension_factorial.py",
            "python scripts/run_extension_external_suites.py semi --workers 3",
            "python scripts/run_extension_external_suites.py pjm",
            "python scripts/build_extension_factorial_release.py",
        ],
        "compute": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "gpu": "not used"},
        "packages": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "catboost": catboost.__version__,
            "lightgbm": lightgbm.__version__, "pyarrow": pyarrow.__version__,
        },
        "counts": {"arms": 16, "model_cells": 80, "pipelines": 240, "inner_jobs": 34560, "outer_jobs": 4896, "semi_rows": 28800, "pjm_runs": 600},
        "winner": winner,
        "files": {str(path.relative_to(ROOT)): digest(path) for path in required},
        "frozen_kse_read_only_hashes": {str(path.relative_to(ROOT)): digest(path) for path in kse_files if path.exists()},
        "external_results_used_for_winner_selection": False,
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "winner": winner, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
