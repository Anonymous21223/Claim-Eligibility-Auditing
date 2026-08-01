"""Audit stored factorial artifacts before preparing the V6.1 manuscript.

This script performs no model fitting. It verifies run-key completeness,
row-level integrity, train-only fields, locked aggregates, module calculations,
external-suite isolation, and the authored V6 freeze.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from finalize_extension_factorial import (
    IDENTITY,
    ROW_KEYS,
    bootstrap_units,
    event_module,
    paired_ci,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "extensions" / "v1"
CORE = ART / "core"
REPORT_DIR = ROOT / "reports" / "extension_factorial_v6_1_submission"
AUTHORED_SOURCE = ROOT / "paper_versions" / "v6_extension_factorial_authored" / "source" / "main.tex"
AUTHORED_PDF = ROOT / "paper" / "final" / "ictai2026_claim_eligibility_extension_factorial_v6_authored.pdf"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(frame["y_true"], frame["y_pred"]) ** 0.5),
        "mae": float(mean_absolute_error(frame["y_true"], frame["y_pred"])),
        "r2": float(r2_score(frame["y_true"], frame["y_pred"])),
    }


def close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return bool(np.isclose(left, right, rtol=0, atol=tolerance, equal_nan=True))


def main() -> None:
    protocol = json.loads((ROOT / "configs" / "extension_factorial_v1.yaml").read_text())
    manifest = json.loads((ART / "RUN_MANIFEST.json").read_text())
    lock = json.loads((CORE / "winner_lock.json").read_text())
    winner = lock["winner"]
    checks: dict[str, dict[str, object]] = {}
    prior_audit_path = REPORT_DIR / "artifact_audit.json"
    prior_audit = json.loads(prior_audit_path.read_text()) if prior_audit_path.exists() else None

    def record(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"status": "PASS" if passed else "FAIL", "evidence": evidence}

    # Freeze authored source/PDF and verify frozen KSE hashes from the run manifest.
    if AUTHORED_SOURCE.exists() and AUTHORED_PDF.exists():
        freeze = {
            str(AUTHORED_SOURCE.relative_to(ROOT)): digest(AUTHORED_SOURCE),
            str(AUTHORED_PDF.relative_to(ROOT)): digest(AUTHORED_PDF),
        }
        freeze_evidence: object = freeze
    elif prior_audit and prior_audit.get("status") == "PASS":
        freeze = prior_audit["authored_freeze"]
        freeze_evidence = {
            "mode": "anonymous package",
            "note": "Identity-bearing authored files are intentionally not distributed; hashes are retained from the passing workspace audit.",
            "sha256": freeze,
        }
    else:
        freeze = {}
        freeze_evidence = "Authored freeze files and a prior passing audit are both absent"
    record("authored_v6_frozen", bool(freeze) and all(len(value) == 64 for value in freeze.values()), freeze_evidence)
    frozen_kse = {}
    for relative, expected in manifest["frozen_kse_read_only_hashes"].items():
        if (ROOT / relative).exists():
            actual = digest(ROOT / relative)
        elif prior_audit and prior_audit.get("status") == "PASS":
            actual = prior_audit["checks"]["frozen_kse_unchanged"]["evidence"][relative]["actual"]
        else:
            actual = "MISSING"
        frozen_kse[relative] = {"expected": expected, "actual": actual}
    record(
        "frozen_kse_unchanged",
        all(item["actual"] == item["expected"] for item in frozen_kse.values()),
        frozen_kse,
    )

    # Unique run keys and one-to-one result completion.
    inner_jobs = pd.read_parquet(CORE / "inner_jobs.parquet")
    inner_results = pd.read_parquet(CORE / "inner_results.parquet")
    outer_jobs = pd.read_parquet(CORE / "outer_jobs.parquet")
    outer_results = pd.read_parquet(CORE / "outer_results.parquet")
    record(
        "inner_jobs_unique_complete",
        len(inner_jobs) == 34560
        and inner_jobs["job_id"].nunique() == 34560
        and inner_results["job_id"].nunique() == 34560
        and set(inner_jobs["job_id"]) == set(inner_results["job_id"]),
        {"jobs": len(inner_jobs), "unique_job_ids": inner_jobs["job_id"].nunique(), "results": len(inner_results)},
    )
    record(
        "outer_jobs_unique_complete",
        len(outer_jobs) == 4896
        and outer_jobs["job_id"].nunique() == 4896
        and outer_results["job_id"].nunique() == 4896
        and set(outer_jobs["job_id"]) == set(outer_results["job_id"]),
        {"jobs": len(outer_jobs), "unique_job_ids": outer_jobs["job_id"].nunique(), "results": len(outer_results)},
    )
    matrix_counts = {
        "arms": inner_jobs["arm_id"].nunique(),
        "arm_model_cells": inner_jobs[["arm_id", "model_id"]].drop_duplicates().shape[0],
        "pipelines": inner_jobs[IDENTITY].drop_duplicates().shape[0],
    }
    record("factorial_matrix_complete", matrix_counts == {"arms": 16, "arm_model_cells": 80, "pipelines": 240}, matrix_counts)

    expected_seeds = set(protocol["seeds"])
    stochastic_models = {"extra_trees", "lightgbm", "catboost"}
    outer_seed_sets = outer_jobs.groupby(["arm_id", "model_id", "feature_family", "outer_fold"])["seed"].agg(lambda x: set(x))
    seed_ok = all(
        seeds == (expected_seeds if key[1] in stochastic_models else {protocol["seeds"][0]})
        for key, seeds in outer_seed_sets.items()
    )
    record("configured_seeds_complete", seed_ok, {"configured": sorted(expected_seeds), "groups": len(outer_seed_sets)})

    # Train-only and row-level integrity checks.
    inner_audit = inner_jobs[["job_id", "train_end"]].merge(inner_results, on="job_id", validate="one_to_one")
    outer_audit = outer_jobs[["job_id", "train_end"]].merge(outer_results, on="job_id", validate="one_to_one")
    train_only = (
        not inner_audit["future_access"].any()
        and not outer_audit["future_access"].any()
        and (inner_audit["max_trend_fit_year"] <= inner_audit["train_end"]).all()
        and (outer_audit["max_trend_fit_year"] <= outer_audit["train_end"]).all()
    )
    record("train_only_execution_fields", train_only, {"inner_future_access": int(inner_audit.future_access.sum()), "outer_future_access": int(outer_audit.future_access.sum())})

    stage_manifests = sorted((ROOT / "data" / "derived" / "stage_features").glob("cutoff_*/manifest.json"))
    stage_ok = len(stage_manifests) == 12
    stage_hashes = {}
    for path in stage_manifests:
        payload = json.loads(path.read_text())
        stage_ok = stage_ok and bool(payload)
        stage_hashes[str(path.relative_to(ROOT))] = digest(path)
    record("stage_manifest_provenance", stage_ok, {"count": len(stage_manifests), "sha256": stage_hashes})

    predictions = pd.read_parquet(CORE / "outer_predictions.parquet")
    mean_predictions = pd.read_parquet(CORE / "outer_mean_predictions.parquet")
    prediction_key = IDENTITY + ROW_KEYS + ["seed"]
    row_constant = (
        predictions.groupby("row_hash")[["country", "region", "crop", "year", "window", "outer_fold", "y_true", "baseline_pred"]]
        .nunique(dropna=False)
        .max()
        .max()
        == 1
    )
    record(
        "outer_prediction_row_integrity",
        not predictions.duplicated(prediction_key).any() and row_constant,
        {"rows": len(predictions), "duplicate_prediction_keys": int(predictions.duplicated(prediction_key).sum()), "unique_row_hashes": predictions.row_hash.nunique()},
    )
    reconstructed = (
        predictions.groupby(IDENTITY + ROW_KEYS, as_index=False)
        .agg(y_pred=("y_pred", "mean"), n_seeds=("seed", "nunique"), seed_prediction_sd=("y_pred", "std"))
    )
    joined_mean = reconstructed.merge(
        mean_predictions[IDENTITY + ROW_KEYS + ["y_pred", "n_seeds", "seed_prediction_sd"]],
        on=IDENTITY + ROW_KEYS,
        suffixes=("_calc", "_stored"),
        validate="one_to_one",
    )
    mean_ok = (
        len(joined_mean) == len(mean_predictions)
        and np.allclose(joined_mean.y_pred_calc, joined_mean.y_pred_stored, atol=1e-12, rtol=0)
        and (joined_mean.n_seeds_calc == joined_mean.n_seeds_stored).all()
        and np.allclose(joined_mean.seed_prediction_sd_calc.fillna(0), joined_mean.seed_prediction_sd_stored.fillna(0), atol=1e-12, rtol=0)
    )
    record("seed_mean_predictions_reproduced", mean_ok, {"stored_rows": len(mean_predictions), "reconstructed_rows": len(joined_mean)})

    # Recompute the complete fold table and pooled leaderboard metrics.
    stored_folds = pd.read_csv(CORE / "outer_fold_metrics.csv")
    fold_rows = []
    pooled_rows = []
    for key, group in mean_predictions.groupby(IDENTITY, sort=True):
        pooled_rows.append({**dict(zip(IDENTITY, key)), **metric_values(group)})
        for fold, fold_group in group.groupby("outer_fold", sort=True):
            fold_rows.append({**dict(zip(IDENTITY, key)), "outer_fold": fold, **metric_values(fold_group)})
    calc_folds = pd.DataFrame(fold_rows)
    fold_compare = calc_folds.merge(stored_folds, on=IDENTITY + ["outer_fold"], suffixes=("_calc", "_stored"), validate="one_to_one")
    fold_ok = len(fold_compare) == 1440 and all(
        np.allclose(fold_compare[f"{metric}_calc"], fold_compare[f"{metric}_stored"], atol=1e-11, rtol=0)
        for metric in ["rmse", "mae", "r2"]
    )
    record("all_outer_fold_metrics_reproduced", fold_ok, {"rows": len(fold_compare)})

    leaderboard = pd.read_csv(CORE / "validation_leaderboard.csv")
    pooled = pd.DataFrame(pooled_rows)
    pool_compare = pooled.merge(leaderboard, on=IDENTITY, validate="one_to_one")
    pool_ok = all(
        np.allclose(pool_compare[metric], pool_compare[f"pooled_{metric}"], atol=1e-11, rtol=0)
        for metric in ["rmse", "mae", "r2"]
    )
    record("all_pooled_metrics_reproduced", pool_ok, {"pipelines": len(pool_compare)})

    # Recompute winner A/B and all event statistics from the stored paired rows.
    candidates = {key: group.reset_index(drop=True) for key, group in mean_predictions.groupby(IDENTITY, sort=True)}
    winner_key = (winner["arm_id"], winner["model_id"], winner["feature_family"])
    winner_rows = candidates[winner_key]
    draws = bootstrap_units(winner_rows, ["year"], protocol["bootstrap"]["draws"], protocol["bootstrap"]["seed"])
    module_a = paired_ci(winner_rows, "y_pred", "baseline_pred", draws)
    metadata = candidates[(winner["arm_id"], winner["model_id"], "metadata_only")]
    paired = winner_rows.merge(metadata[ROW_KEYS + ["y_pred"]], on=ROW_KEYS, suffixes=("", "_metadata"), validate="one_to_one")
    b_draws = bootstrap_units(paired, ["year"], protocol["bootstrap"]["draws"], protocol["bootstrap"]["seed"])
    module_b = paired_ci(paired, "y_pred", "y_pred_metadata", b_draws)
    candidate_index = list(candidates).index(winner_key)
    event_seed = protocol["bootstrap"]["seed"] + candidate_index * 17
    year_draws = bootstrap_units(mean_predictions.drop_duplicates(ROW_KEYS), ["year"], protocol["bootstrap"]["draws"], protocol["bootstrap"]["seed"])
    event = event_module(winner_rows, year_draws, event_seed)
    stored_modules = pd.read_csv(CORE / "module_decisions.csv")
    stored_winner = stored_modules[
        (stored_modules.arm_id == winner["arm_id"])
        & (stored_modules.model_id == winner["model_id"])
        & (stored_modules.feature_family == winner["feature_family"])
    ].iloc[0]
    module_ok = all(
        [
            close(module_a["delta_rmse"], stored_winner.module_a_delta_rmse),
            close(module_a["ci95_low"], stored_winner.module_a_ci95_low),
            close(module_a["ci95_high"], stored_winner.module_a_ci95_high),
            close(module_b["delta_rmse"], stored_winner.module_b_delta_rmse),
            close(module_b["ci95_low"], stored_winner.module_b_ci95_low),
            close(module_b["ci95_high"], stored_winner.module_b_ci95_high),
            close(event["event_spearman"], stored_winner.event_spearman),
            close(event["event_topk_lift"], stored_winner.event_topk_lift),
            event["module_e"] == stored_winner.module_e,
        ]
    )
    tail = winner_rows[winner_rows["trend_residual_z"] < -1].reset_index(drop=True)
    tail_draws = bootstrap_units(tail, ["year"], protocol["bootstrap"]["draws"], event_seed)
    y_true = tail.y_true.to_numpy(float)
    y_pred = tail.y_pred.to_numpy(float)
    zero = np.zeros(len(tail))
    event_full = {}
    for label, metric in [("tail_rmse", lambda y, p: mean_squared_error(y, p) ** 0.5), ("tail_mae", mean_absolute_error)]:
        values = np.asarray([metric(y_true[index], y_pred[index]) - metric(y_true[index], zero[index]) for index in tail_draws])
        event_full[label] = {
            "estimate": float(metric(y_true, y_pred) - metric(y_true, zero)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
    record(
        "winner_modules_reproduced",
        module_ok and len(tail) == 136,
        {"bootstrap_draws": 2000, "bootstrap_unit": "calendar year", "module_a": module_a, "module_b": module_b, "event": {key: value.item() if isinstance(value, np.generic) else value for key, value in event.items()}, "event_full": event_full},
    )

    counts = {
        "pipelines": int(len(leaderboard)),
        "pass_r1": int(leaderboard.R1_stable_positive_pass.sum()),
        "pass_a": int((stored_modules.module_a == "PASS").sum()),
        "full_pipelines": int((stored_modules.feature_family == "full").sum()),
        "pass_b_full": int(((stored_modules.feature_family == "full") & (stored_modules.module_b == "PASS")).sum()),
        "pass_a_b_full": int(((stored_modules.feature_family == "full") & (stored_modules.module_a == "PASS") & (stored_modules.module_b == "PASS")).sum()),
        "pass_e": int((stored_modules.module_e == "PASS").sum()),
        "evaluated_e": int((stored_modules.module_e != "NOT_EVALUATED").sum()),
    }
    record("candidate_wide_counts", counts == {"pipelines": 240, "pass_r1": 38, "pass_a": 122, "full_pipelines": 80, "pass_b_full": 46, "pass_a_b_full": 46, "pass_e": 0, "evaluated_e": 46}, counts)

    # Winner-lock integrity and external isolation.
    lock_hash_ok = digest(CORE / "validation_leaderboard.csv") == lock["leaderboard_sha256"] and digest(ROOT / "configs" / "extension_factorial_v1.yaml") == lock["protocol_sha256"]
    command_text = "\n".join(manifest["commands"])
    order_ok = command_text.index("finalize_extension_factorial.py") < command_text.index("run_extension_external_suites.py semi") < command_text.index("run_extension_external_suites.py pjm")
    external_script = (ROOT / "scripts" / "run_extension_external_suites.py").read_text(encoding="utf-8")
    hard_precondition = "winner_lock.json" in external_script and "status" in external_script and "LOCKED" in external_script
    finalize_source = (ROOT / "scripts" / "finalize_extension_factorial.py").read_text(encoding="utf-8")
    dependency_isolated = (
        'ART / "semi"' not in finalize_source
        and 'ART / "pjm"' not in finalize_source
        and "scenario_results.csv" not in finalize_source
        and "degradation_curve.csv" not in finalize_source
    )
    record(
        "winner_lock_and_external_order",
        lock_hash_ok and order_ok and hard_precondition and dependency_isolated and not lock["selection_used_external_results"],
        {
            "leaderboard_hash_matches": lock_hash_ok,
            "manifest_command_order": order_ok,
            "external_runner_requires_lock": hard_precondition,
            "selection_source_has_no_external_dependency": dependency_isolated,
            "filesystem_mtime_note": "winner_lock was later rewritten; mtime is not treated as execution-order evidence",
        },
    )

    semi = pd.read_csv(ART / "semi" / "scenario_results.csv")
    pjm = pd.read_csv(ART / "pjm" / "scenario_results.csv")
    curve = pd.read_csv(ART / "pjm" / "degradation_curve.csv")
    record(
        "external_suite_completion",
        len(semi) == 28800
        and semi["seed"].nunique() == 30
        and len(pjm) == 600
        and pjm.groupby(["mechanism", "level"])["seed"].nunique().eq(30).all()
        and len(curve) == 20,
        {"semi_rows": len(semi), "semi_seeds": semi.seed.nunique(), "pjm_rows": len(pjm), "pjm_curve_rows": len(curve)},
    )

    all_pass = all(item["status"] == "PASS" for item in checks.values())
    report = {
        "status": "PASS" if all_pass else "FAIL",
        "decision": "NO_RERUN" if all_pass else "BLOCK_AND_INVESTIGATE",
        "model_fits_executed": False,
        "authored_freeze": freeze,
        "checks": checks,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "artifact_audit.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Extension Factorial V6.1 Artifact Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Decision: **{report['decision']}**",
        "- Model fitting performed by this audit: **No**",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {item['status']} |" for name, item in checks.items())
    lines.extend(
        [
            "",
            "## Candidate-Wide Counts",
            "",
            f"- R1: {counts['pass_r1']}/{counts['pipelines']}",
            f"- Module A: {counts['pass_a']}/{counts['pipelines']}",
            f"- Module B: {counts['pass_b_full']}/{counts['full_pipelines']} full pipelines",
            f"- A+B: {counts['pass_a_b_full']}/{counts['full_pipelines']} full pipelines",
            f"- Module E: {counts['pass_e']}/{counts['evaluated_e']} evaluated candidates",
            "",
            "## Lock-Order Evidence",
            "",
            "The final filesystem modification time is not used as proof because the lock was rewritten during finalization. The stored command order places core finalization before both external suites, the external runner requires an existing LOCKED file, the lock hashes the crop-only leaderboard and protocol, and the selection source does not read EXT-SEMI or EXT-PJM artifacts.",
            "",
        ]
    )
    (REPORT_DIR / "artifact_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "checks": {key: value["status"] for key, value in checks.items()}}, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
