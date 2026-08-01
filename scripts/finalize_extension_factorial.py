"""Create the locked core leaderboard and claim-module decisions."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.extension_factorial import file_sha256  # noqa: E402


OUT = ROOT / "artifacts" / "extensions" / "v1"
CORE = OUT / "core"
IDENTITY = ["arm_id", "model_id", "feature_family"]
ROW_KEYS = ["country", "region", "crop", "year", "window", "outer_fold"]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def bootstrap_units(
    frame: pd.DataFrame,
    unit_columns: list[str],
    draws: int,
    seed: int,
) -> list[np.ndarray]:
    labels = frame[unit_columns].astype(str).agg("|".join, axis=1)
    units = labels.drop_duplicates().tolist()
    indices = {
        unit: np.flatnonzero(labels.to_numpy() == unit)
        for unit in units
    }
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([indices[unit] for unit in rng.choice(units, len(units))])
        for _ in range(draws)
    ]


def paired_ci(
    frame: pd.DataFrame,
    left: str,
    right: str,
    draws: list[np.ndarray],
) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    left_value = frame[left].to_numpy(dtype=float)
    right_value = frame[right].to_numpy(dtype=float)
    point = (
        mean_squared_error(y, left_value) ** 0.5
        - mean_squared_error(y, right_value) ** 0.5
    )
    samples = np.asarray(
        [
            mean_squared_error(y[index], left_value[index]) ** 0.5
            - mean_squared_error(y[index], right_value[index]) ** 0.5
            for index in draws
        ]
    )
    return {
        "delta_rmse": float(point),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def event_module(
    frame: pd.DataFrame,
    year_draws: list[np.ndarray],
    seed: int,
) -> dict[str, object]:
    tail = frame[frame["trend_residual_z"] < -1.0].reset_index(drop=True)
    if len(tail) < 10:
        return {"module_e": "FAIL", "event_reason": "fewer_than_10_primary_tail_rows"}
    tail_draws = bootstrap_units(tail, ["year"], len(year_draws), seed)
    y = tail["y_true"].to_numpy(dtype=float)
    prediction = tail["y_pred"].to_numpy(dtype=float)
    zero = np.zeros(len(tail))
    delta_rmse = []
    delta_mae = []
    rho_values = []
    lift_values = []
    k = min(10, len(tail))

    def topk_lift(observed: np.ndarray, predicted: np.ndarray) -> tuple[int, float]:
        observed_set = set(np.argsort(observed, kind="stable")[:k])
        predicted_set = set(np.argsort(predicted, kind="stable")[:k])
        overlap = len(observed_set & predicted_set)
        return overlap, float(overlap * len(observed) / (k * k))

    overlap, lift = topk_lift(y, prediction)
    for index in tail_draws:
        sample_y = y[index]
        sample_prediction = prediction[index]
        delta_rmse.append(
            mean_squared_error(sample_y, sample_prediction) ** 0.5
            - mean_squared_error(sample_y, np.zeros(len(index))) ** 0.5
        )
        delta_mae.append(
            mean_absolute_error(sample_y, sample_prediction)
            - mean_absolute_error(sample_y, np.zeros(len(index)))
        )
        rho_values.append(float(spearmanr(sample_y, sample_prediction).statistic))
        _, sampled_lift = topk_lift(sample_y, sample_prediction)
        lift_values.append(sampled_lift)

    observed_rho = float(spearmanr(y, prediction).statistic)
    rng = np.random.default_rng(seed + 1)
    labels = tail["year"].to_numpy()
    perm_rho = []
    perm_overlap = []
    for _ in range(10000):
        permuted = prediction.copy()
        for year in np.unique(labels):
            positions = np.flatnonzero(labels == year)
            permuted[positions] = rng.permutation(permuted[positions])
        perm_rho.append(float(spearmanr(y, permuted).statistic))
        perm_overlap.append(topk_lift(y, permuted)[0])
    rank_p = (1 + np.sum(np.asarray(perm_rho) >= observed_rho)) / 10001
    topk_permutation_p = (
        1 + np.sum(np.asarray(perm_overlap) >= overlap)
    ) / 10001
    topk_hypergeom_p = float(hypergeom.sf(overlap - 1, len(tail), k, k))
    checks = {
        "tail_rmse_pass": float(np.quantile(delta_rmse, 0.975)) < 0,
        "tail_mae_pass": float(np.quantile(delta_mae, 0.975)) < 0,
        "rank_pass": (
            float(np.quantile(rho_values, 0.025)) > 0 and rank_p < 0.05
        ),
        "topk_pass": (
            lift > 1
            and float(np.quantile(lift_values, 0.025)) > 1
            and topk_hypergeom_p < 0.05
            and topk_permutation_p < 0.05
        ),
    }
    return {
        "module_e": "PASS" if all(checks.values()) else "FAIL",
        "event_n": len(tail),
        "event_delta_rmse_ci95_high": float(np.quantile(delta_rmse, 0.975)),
        "event_delta_mae_ci95_high": float(np.quantile(delta_mae, 0.975)),
        "event_spearman": observed_rho,
        "event_spearman_ci95_low": float(np.quantile(rho_values, 0.025)),
        "event_rank_permutation_p": float(rank_p),
        "event_topk_k": k,
        "event_topk_overlap": overlap,
        "event_topk_lift": lift,
        "event_topk_lift_ci95_low": float(np.quantile(lift_values, 0.025)),
        "event_topk_hypergeom_p": topk_hypergeom_p,
        "event_topk_permutation_p": float(topk_permutation_p),
        **checks,
    }


def aggregate_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    expected_constant = [
        "row_hash",
        "y_true",
        "baseline_pred",
        "trend_residual_z",
    ]
    for column in expected_constant:
        if predictions.groupby(IDENTITY + ROW_KEYS)[column].nunique().max() != 1:
            raise AssertionError(f"{column} varies within a seeded prediction row")
    return (
        predictions.groupby(IDENTITY + ROW_KEYS, as_index=False)
        .agg(
            row_hash=("row_hash", "first"),
            y_true=("y_true", "first"),
            trend_residual_z=("trend_residual_z", "first"),
            baseline_pred=("baseline_pred", "first"),
            y_pred=("y_pred", "mean"),
            n_seeds=("seed", "nunique"),
            seed_prediction_sd=("y_pred", "std"),
        )
        .fillna({"seed_prediction_sd": 0.0})
    )


def protocol_rank(eligible: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    ranked = []
    remaining = eligible.copy()
    for tier in sorted(remaining["claim_tier_rank"].unique()):
        tier_rows = remaining[remaining["claim_tier_rank"] == tier].copy()
        while not tier_rows.empty:
            best_rmse = float(tier_rows["pooled_rmse"].min())
            tie_band = tier_rows[
                tier_rows["pooled_rmse"] - best_rmse < tolerance
            ].sort_values(
                [
                    "worst_outer_fold_rmse",
                    "complexity_rank",
                    "arm_id",
                    "model_id",
                    "feature_family",
                ],
                kind="stable",
            )
            chosen_index = tie_band.index[0]
            ranked.append(tier_rows.loc[chosen_index])
            tier_rows = tier_rows.drop(index=chosen_index)
    return pd.DataFrame(ranked).reset_index(drop=True)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "extension_factorial_v1.yaml").read_text()
    )
    matrix = pd.read_csv(OUT / "COMPLETION_MATRIX.csv")
    expected = protocol["expected_counts"]
    observed = matrix.groupby("phase").size().to_dict()
    if (
        observed.get("inner") != expected["inner_fit_calls"]
        or observed.get("outer") != expected["outer_seeded_fits"]
        or not matrix["status"].eq("PASS").all()
    ):
        raise AssertionError("Leaderboard is blocked until completion reaches 100%")
    predictions = pd.read_parquet(CORE / "outer_predictions.parquet")
    required = {
        "arm_id",
        "model_id",
        "feature_family",
        "outer_fold",
        "seed",
        "y_true",
        "y_pred",
        "baseline_pred",
        "row_hash",
    }
    if required - set(predictions):
        raise AssertionError("Outer predictions lack required columns")
    mean_predictions = aggregate_seed_predictions(predictions)
    mean_predictions.to_parquet(
        CORE / "outer_mean_predictions.parquet",
        index=False,
    )
    year_draws = bootstrap_units(
        mean_predictions.drop_duplicates(ROW_KEYS),
        ["year"],
        int(protocol["bootstrap"]["draws"]),
        int(protocol["bootstrap"]["seed"]),
    )
    rows = []
    fold_rows = []
    module_rows = []
    candidates = {
        key: group.reset_index(drop=True)
        for key, group in mean_predictions.groupby(IDENTITY, sort=True)
    }
    for candidate_index, (key, group) in enumerate(candidates.items()):
        identity = dict(zip(IDENTITY, key))
        pooled = metrics(group["y_true"], group["y_pred"])
        fold_metrics = []
        for fold, fold_group in group.groupby("outer_fold", sort=True):
            scored = metrics(fold_group["y_true"], fold_group["y_pred"])
            fold_metrics.append(scored)
            fold_rows.append({**identity, "outer_fold": fold, **scored})
        r0 = (
            len(fold_metrics) == 6
            and group["row_hash"].notna().all()
            and np.isfinite(group[["y_true", "y_pred"]]).all().all()
        )
        positive_folds = sum(item["r2"] > 0 for item in fold_metrics)
        r1 = pooled["r2"] > 0 and positive_folds >= 4
        draws = bootstrap_units(
            group,
            ["year"],
            int(protocol["bootstrap"]["draws"]),
            int(protocol["bootstrap"]["seed"]),
        )
        crop_state_draws = bootstrap_units(
            group,
            ["crop", "region"],
            int(protocol["bootstrap"]["draws"]),
            int(protocol["bootstrap"]["seed"]) + 1,
        )
        module_a = paired_ci(group, "y_pred", "baseline_pred", draws)
        module_a_sensitivity = paired_ci(
            group,
            "y_pred",
            "baseline_pred",
            crop_state_draws,
        )
        a_pass = module_a["ci95_high"] < 0
        metadata_key = (key[0], key[1], "metadata_only")
        b_result = {"delta_rmse": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
        b_sensitivity = b_result.copy()
        b_pass = False
        if key[2] == "full":
            comparator = candidates[metadata_key]
            joined = group.merge(
                comparator[ROW_KEYS + ["y_pred"]],
                on=ROW_KEYS,
                suffixes=("", "_metadata"),
                validate="one_to_one",
            )
            if len(joined) != len(group):
                raise AssertionError(f"Module B row mismatch for {key}")
            b_draws = bootstrap_units(
                joined,
                ["year"],
                int(protocol["bootstrap"]["draws"]),
                int(protocol["bootstrap"]["seed"]),
            )
            b_crop_state = bootstrap_units(
                joined,
                ["crop", "region"],
                int(protocol["bootstrap"]["draws"]),
                int(protocol["bootstrap"]["seed"]) + 1,
            )
            b_result = paired_ci(joined, "y_pred", "y_pred_metadata", b_draws)
            b_sensitivity = paired_ci(
                joined,
                "y_pred",
                "y_pred_metadata",
                b_crop_state,
            )
            b_pass = b_result["ci95_high"] < 0
        event = {"module_e": "NOT_EVALUATED"}
        if a_pass and b_pass:
            event = event_module(
                group,
                year_draws,
                int(protocol["bootstrap"]["seed"]) + candidate_index * 17,
            )
        e_pass = event["module_e"] == "PASS"
        tier = (
            "A+B+E"
            if a_pass and b_pass and e_pass
            else "A+B"
            if a_pass and b_pass
            else "A"
            if a_pass
            else "none"
        )
        rows.append(
            {
                **identity,
                "n": len(group),
                **{f"pooled_{name}": value for name, value in pooled.items()},
                "worst_outer_fold_rmse": max(item["rmse"] for item in fold_metrics),
                "positive_r2_folds": positive_folds,
                "mean_seed_prediction_sd": float(group["seed_prediction_sd"].mean()),
                "R0_integrity_pass": r0,
                "R1_stable_positive_pass": r1,
                "claim_tier": tier,
            }
        )
        module_rows.append(
            {
                **identity,
                "module_a": "PASS" if a_pass else "FAIL",
                "module_a_delta_rmse": module_a["delta_rmse"],
                "module_a_ci95_low": module_a["ci95_low"],
                "module_a_ci95_high": module_a["ci95_high"],
                "module_a_crop_state_ci95_high": module_a_sensitivity["ci95_high"],
                "module_b": (
                    "PASS" if b_pass else "FAIL"
                    if key[2] == "full"
                    else "NOT_APPLICABLE"
                ),
                "module_b_delta_rmse": b_result["delta_rmse"],
                "module_b_ci95_low": b_result["ci95_low"],
                "module_b_ci95_high": b_result["ci95_high"],
                "module_b_crop_state_ci95_high": b_sensitivity["ci95_high"],
                **event,
            }
        )
    leaderboard = pd.DataFrame(rows)
    modules = pd.DataFrame(module_rows)
    leaderboard = leaderboard.merge(
        modules[IDENTITY + ["module_a", "module_b", "module_e"]],
        on=IDENTITY,
        validate="one_to_one",
    )
    model_rank = {
        "linear_shrinkage": 0,
        "hist_gradient_boosting": 1,
        "extra_trees": 2,
        "lightgbm": 3,
        "catboost": 4,
    }
    tier_rank = {"A+B+E": 0, "A+B": 1, "A": 2, "none": 3}
    arm_lookup = {arm["arm_id"]: arm for arm in protocol["arms"]}
    arm_rank = {
        arm_id: (
            {"BASE": 0, "STAGE-CAL": 1, "STAGE-GDD": 1, "STAGE-HYBRID": 2}[
                arm["representation"]
            ]
            + (2 if arm["hierarchy"] == "HIER-RESIDUAL" else 0)
            + (1 if arm["grid"] == "EXPANDED" else 0)
        )
        for arm_id, arm in arm_lookup.items()
    }
    leaderboard["complexity_rank"] = (
        leaderboard["model_id"].map(model_rank) * 10
        + leaderboard["feature_family"].map(
            {"metadata_only": 0, "weather_only": 1, "full": 2}
        )
        + leaderboard["arm_id"].map(arm_rank) / 10
    )
    leaderboard["claim_tier_rank"] = leaderboard["claim_tier"].map(tier_rank)
    eligible = leaderboard[
        leaderboard["R0_integrity_pass"] & leaderboard["R1_stable_positive_pass"]
    ].copy()
    eligible = protocol_rank(
        eligible,
        float(protocol["selection"]["rmse_tie_tolerance"]),
    )
    leaderboard["winner_rank"] = pd.NA
    for rank, selected_row in enumerate(eligible.head(3).itertuples(index=False), 1):
        selected_mask = np.logical_and.reduce(
            [
                leaderboard[column].eq(getattr(selected_row, column))
                for column in IDENTITY
            ]
        )
        leaderboard.loc[selected_mask, "winner_rank"] = rank
    leaderboard = leaderboard.sort_values(
        ["winner_rank", "pooled_rmse"],
        na_position="last",
        kind="stable",
    )
    leaderboard.to_csv(CORE / "validation_leaderboard.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(CORE / "outer_fold_metrics.csv", index=False)
    modules.to_csv(CORE / "module_decisions.csv", index=False)
    if eligible.empty:
        lock = {
            "status": "NO_STABLE_POSITIVE_WINNER",
            "reason": "No candidate passed R1; EXT-SEMI and EXT-PJM are blocked by protocol.",
            "winner": None,
            "runners_up": [],
        }
    else:
        records = eligible.head(3)[
            IDENTITY
            + [
                "claim_tier",
                "pooled_rmse",
                "pooled_r2",
                "positive_r2_folds",
            ]
        ].to_dict("records")
        lock = {
            "status": "LOCKED",
            "winner": records[0],
            "runners_up": records[1:],
            "selection_used_external_results": False,
        }
    lock["protocol_sha256"] = file_sha256(
        ROOT / "configs" / "extension_factorial_v1.yaml"
    )
    lock["leaderboard_sha256"] = file_sha256(
        CORE / "validation_leaderboard.csv"
    )
    (CORE / "winner_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
