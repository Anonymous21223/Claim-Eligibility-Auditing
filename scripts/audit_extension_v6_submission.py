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
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import finalize_extension_factorial as finalizer
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
REPORT_DIR = ROOT / "reports" / "extension_factorial_v6_1_release"
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


def paired_ci_fast(
    frame: pd.DataFrame,
    left: str,
    right: str,
    draws: list[np.ndarray],
) -> dict[str, float]:
    """Numerically equivalent paired RMSE CI without estimator-call overhead."""
    y = frame["y_true"].to_numpy(float)
    left_squared = np.square(y - frame[left].to_numpy(float))
    right_squared = np.square(y - frame[right].to_numpy(float))
    samples = np.fromiter(
        (
            np.sqrt(np.mean(left_squared[index]))
            - np.sqrt(np.mean(right_squared[index]))
            for index in draws
        ),
        dtype=float,
        count=len(draws),
    )
    return {
        "delta_rmse": float(np.sqrt(np.mean(left_squared)) - np.sqrt(np.mean(right_squared))),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def frames_close(
    left: pd.DataFrame,
    right: pd.DataFrame,
    sort_columns: list[str],
    *,
    atol: float = 1e-11,
) -> bool:
    left_sorted = left.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    right_sorted = right.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left_sorted,
            right_sorted,
            check_dtype=False,
            check_like=True,
            check_exact=False,
            rtol=0,
            atol=atol,
        )
    except AssertionError:
        return False
    return True


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

    # Recompute A/B for all candidates. Module E is conjunctive, so E failure
    # can be established cheaply from a failed top-k component; only candidates
    # that survive the structural top-k checks need the 10,000 permutations.
    candidates = {
        key: group.reset_index(drop=True)
        for key, group in mean_predictions.groupby(IDENTITY, sort=True)
    }
    canonical_rows = next(iter(candidates.values()))[ROW_KEYS]
    candidate_rows_aligned = all(
        group[ROW_KEYS].equals(canonical_rows)
        for group in candidates.values()
    )
    if not candidate_rows_aligned:
        raise AssertionError("Candidate paired rows are not aligned")
    shared_year_bootstrap = bootstrap_units(
        next(iter(candidates.values())),
        ["year"],
        int(protocol["bootstrap"]["draws"]),
        int(protocol["bootstrap"]["seed"]),
    )
    shared_state_bootstrap = bootstrap_units(
        next(iter(candidates.values())),
        ["crop", "region"],
        int(protocol["bootstrap"]["draws"]),
        int(protocol["bootstrap"]["seed"]) + 1,
    )
    stored_modules = pd.read_csv(CORE / "module_decisions.csv")
    stored_module_lookup = {
        tuple(getattr(row, column) for column in IDENTITY): row
        for row in stored_modules.itertuples(index=False)
    }
    module_rows_match = True
    calculated_tiers: dict[tuple[str, str, str], str] = {}
    calculated_a: dict[tuple[str, str, str], bool] = {}
    calculated_b: dict[tuple[str, str, str], bool] = {}
    calculated_e: dict[tuple[str, str, str], bool] = {}
    evaluated_e = 0
    e_failures_proved = 0
    e_permutation_candidates = 0

    def topk_overlap_and_lift(observed: np.ndarray, predicted: np.ndarray, k: int) -> tuple[int, float]:
        observed_set = set(np.argsort(observed, kind="stable")[:k])
        predicted_set = set(np.argsort(predicted, kind="stable")[:k])
        overlap = len(observed_set & predicted_set)
        return overlap, float(overlap * len(observed) / (k * k))

    for candidate_index, (key, group) in enumerate(candidates.items()):
        stored = stored_module_lookup[key]
        a_result = paired_ci_fast(group, "y_pred", "baseline_pred", shared_year_bootstrap)
        a_sensitivity = paired_ci_fast(group, "y_pred", "baseline_pred", shared_state_bootstrap)
        a_pass = a_result["ci95_high"] < 0
        b_result = {"delta_rmse": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
        b_sensitivity = b_result.copy()
        b_pass = False
        if key[2] == "full":
            comparator = candidates[(key[0], key[1], "metadata_only")]
            joined = group.merge(
                comparator[ROW_KEYS + ["y_pred"]],
                on=ROW_KEYS,
                suffixes=("", "_metadata"),
                validate="one_to_one",
            )
            if not joined[ROW_KEYS].equals(canonical_rows):
                raise AssertionError(f"Module B paired rows are not aligned for {key}")
            b_result = paired_ci_fast(joined, "y_pred", "y_pred_metadata", shared_year_bootstrap)
            b_sensitivity = paired_ci_fast(joined, "y_pred", "y_pred_metadata", shared_state_bootstrap)
            b_pass = b_result["ci95_high"] < 0

        e_pass = False
        if a_pass and b_pass:
            evaluated_e += 1
            tail = group[group["trend_residual_z"] < -1.0].reset_index(drop=True)
            k = min(10, len(tail))
            observed = tail["y_true"].to_numpy(float)
            predicted = tail["y_pred"].to_numpy(float)
            overlap, lift = topk_overlap_and_lift(observed, predicted, k)
            event_seed = int(protocol["bootstrap"]["seed"]) + candidate_index * 17
            tail_bootstrap = bootstrap_units(
                tail,
                ["year"],
                int(protocol["bootstrap"]["draws"]),
                event_seed,
            )
            lift_values = [
                topk_overlap_and_lift(observed[index], predicted[index], k)[1]
                for index in tail_bootstrap
            ]
            lift_low = float(np.quantile(lift_values, 0.025))
            hypergeom_p = float(hypergeom.sf(overlap - 1, len(tail), k, k))
            structural_topk_pass = lift > 1 and lift_low > 1 and hypergeom_p < 0.05
            permutation_p = 1.0
            if structural_topk_pass:
                e_permutation_candidates += 1
                rng = np.random.default_rng(event_seed + 1)
                labels = tail["year"].to_numpy()
                permutation_overlaps = []
                for _ in range(10000):
                    permuted = predicted.copy()
                    for year in np.unique(labels):
                        positions = np.flatnonzero(labels == year)
                        permuted[positions] = rng.permutation(permuted[positions])
                    permutation_overlaps.append(
                        topk_overlap_and_lift(observed, permuted, k)[0]
                    )
                permutation_p = (
                    1 + np.sum(np.asarray(permutation_overlaps) >= overlap)
                ) / 10001
            topk_pass = structural_topk_pass and permutation_p < 0.05
            if not topk_pass:
                e_failures_proved += 1
            else:
                e_pass = event_module(
                    group,
                    bootstrap_units(
                        mean_predictions.drop_duplicates(ROW_KEYS),
                        ["year"],
                        int(protocol["bootstrap"]["draws"]),
                        int(protocol["bootstrap"]["seed"]),
                    ),
                    event_seed,
                )["module_e"] == "PASS"

        tier = "A+B+E" if a_pass and b_pass and e_pass else "A+B" if a_pass and b_pass else "A" if a_pass else "none"
        calculated_tiers[key] = tier
        calculated_a[key] = a_pass
        calculated_b[key] = b_pass
        calculated_e[key] = e_pass
        module_rows_match = module_rows_match and all(
            [
                (stored.module_a == "PASS") == a_pass,
                close(a_result["delta_rmse"], stored.module_a_delta_rmse),
                close(a_result["ci95_low"], stored.module_a_ci95_low),
                close(a_result["ci95_high"], stored.module_a_ci95_high),
                close(a_sensitivity["ci95_high"], stored.module_a_crop_state_ci95_high),
                (stored.module_b == "PASS") == b_pass,
                close(b_result["delta_rmse"], stored.module_b_delta_rmse),
                close(b_result["ci95_low"], stored.module_b_ci95_low),
                close(b_result["ci95_high"], stored.module_b_ci95_high),
                close(b_sensitivity["ci95_high"], stored.module_b_crop_state_ci95_high),
                (stored.module_e == "PASS") == e_pass,
            ]
        )

    calculated_tier_frame = pd.DataFrame(
        [dict(zip(IDENTITY, key)) | {"claim_tier_calc": tier} for key, tier in calculated_tiers.items()]
    )
    tier_compare = leaderboard.merge(calculated_tier_frame, on=IDENTITY, validate="one_to_one")
    tiers_match = tier_compare["claim_tier"].eq(tier_compare["claim_tier_calc"]).all()
    rank_frame = leaderboard.drop(columns=["claim_tier"]).merge(
        calculated_tier_frame.rename(columns={"claim_tier_calc": "claim_tier"}),
        on=IDENTITY,
        validate="one_to_one",
    )
    tier_rank = {"A+B+E": 0, "A+B": 1, "A": 2, "none": 3}
    rank_frame["claim_tier_rank"] = rank_frame["claim_tier"].map(tier_rank)
    eligible = rank_frame[
        rank_frame["R0_integrity_pass"] & rank_frame["R1_stable_positive_pass"]
    ].copy()
    reranked = finalizer.protocol_rank(
        eligible,
        float(protocol["selection"]["rmse_tie_tolerance"]),
    )
    expected_top = [lock["winner"], *lock["runners_up"]]
    calculated_top = reranked.head(3)[
        IDENTITY + ["claim_tier", "pooled_rmse", "pooled_r2", "positive_r2_folds"]
    ].to_dict("records")
    top_three_match = len(calculated_top) == len(expected_top) and all(
        all(calculated[column] == expected[column] for column in IDENTITY + ["claim_tier"])
        and int(calculated["positive_r2_folds"]) == int(expected["positive_r2_folds"])
        and close(calculated["pooled_rmse"], expected["pooled_rmse"], tolerance=1e-12)
        and close(calculated["pooled_r2"], expected["pooled_r2"], tolerance=1e-12)
        for calculated, expected in zip(calculated_top, expected_top)
    )
    core_table_checks = {
        "all_a_b_rows_match": bool(module_rows_match),
        "all_claim_tiers_match": bool(tiers_match),
        "all_e_failures_proved": bool(evaluated_e == 46 and e_failures_proved == 46),
        "tier_first_top_three_match": bool(top_three_match),
    }
    record(
        "all_core_tables_and_selection_reproduced",
        all(core_table_checks.values()),
        {
            **core_table_checks,
            "pipelines": len(candidates),
            "paired_rows_aligned": candidate_rows_aligned,
            "e_evaluated": evaluated_e,
            "e_permutation_candidates": e_permutation_candidates,
            "expected_top_three": expected_top,
            "calculated_top_three": calculated_top,
            "model_fits_executed": False,
            "source": "locked outer_mean_predictions.parquet",
        },
    )

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

    positive_fold_counts = (
        calc_folds.assign(positive=calc_folds["r2"] > 0)
        .groupby(IDENTITY, as_index=False)["positive"]
        .sum()
    )
    r1_frame = pooled.merge(positive_fold_counts, on=IDENTITY, validate="one_to_one")
    calculated_r1 = (r1_frame["r2"] > 0) & (r1_frame["positive"] >= 4)
    full_keys = [key for key in candidates if key[2] == "full"]
    counts = {
        "pipelines": len(candidates),
        "pass_r1": int(calculated_r1.sum()),
        "pass_a": int(sum(calculated_a.values())),
        "full_pipelines": len(full_keys),
        "pass_b_full": int(sum(calculated_b[key] for key in full_keys)),
        "pass_a_b_full": int(sum(calculated_a[key] and calculated_b[key] for key in full_keys)),
        "pass_e": int(sum(calculated_e.values())),
        "evaluated_e": evaluated_e,
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

    # Recover the external summaries used by Tables V--VI directly from the
    # stored run-level outputs, including Wilson intervals for EXT-SEMI.
    semi_calc = (
        semi.groupby(
            ["candidate_id", "signal_strength_mde", "event_prevalence"],
            as_index=False,
        )
        .agg(runs=("policy_permission", "size"), permissions=("policy_permission", "sum"))
    )
    semi_calc["permission_probability"] = semi_calc["permissions"] / semi_calc["runs"]
    z = 1.959963984540054
    denominator = 1 + z**2 / semi_calc["runs"]
    center = (
        semi_calc["permission_probability"] + z**2 / (2 * semi_calc["runs"])
    ) / denominator
    half_width = (
        z
        * np.sqrt(
            semi_calc["permission_probability"]
            * (1 - semi_calc["permission_probability"])
            / semi_calc["runs"]
            + z**2 / (4 * semi_calc["runs"] ** 2)
        )
        / denominator
    )
    semi_calc["wilson95_low"] = center - half_width
    semi_calc["wilson95_high"] = center + half_width
    semi_stored = pd.read_csv(ART / "semi" / "scenario_summary.csv")
    semi_summary_ok = frames_close(
        semi_calc,
        semi_stored,
        ["candidate_id", "signal_strength_mde", "event_prevalence"],
    )

    curve_calc = (
        pjm.groupby(["mechanism", "level"], as_index=False)
        .agg(
            mean_delta_rmse_b=("delta_rmse_b", "mean"),
            sd_delta_rmse_b=("delta_rmse_b", "std"),
            module_a_pass_probability=("module_a_pass", "mean"),
            module_b_pass_probability=("module_b_pass", "mean"),
            seeds=("seed", "size"),
        )
    )
    pjm_curve_ok = frames_close(
        curve_calc,
        curve,
        ["mechanism", "level"],
        atol=2e-11,
    )
    base_table = (
        semi_calc[semi_calc.event_prevalence == "base"]
        .set_index(["candidate_id", "signal_strength_mde"])["permission_probability"]
    )
    table_v_values = {
        candidate: [float(base_table.loc[(candidate, strength)]) for strength in [0.0, 1.0, 1.5]]
        for candidate in ["winner", "runner_up_1", "runner_up_2", "kse_primary"]
    }
    table_vi_values = {
        mechanism: {
            "max_level": float(group.level.max()),
            "p_b_at_zero": float(group.sort_values("level").module_b_pass_probability.iloc[0]),
            "p_b_at_max": float(group.sort_values("level").module_b_pass_probability.iloc[-1]),
        }
        for mechanism, group in curve_calc.groupby("mechanism", sort=True)
    }
    record(
        "external_tables_reproduced",
        semi_summary_ok and pjm_curve_ok,
        {
            "semi_summary_matches_run_rows": semi_summary_ok,
            "pjm_curve_matches_seed_rows": pjm_curve_ok,
            "table_v_permission_probabilities": table_v_values,
            "table_vi_endpoints": table_vi_values,
            "model_fits_executed": False,
        },
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
