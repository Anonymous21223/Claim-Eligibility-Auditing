"""Power calibration for Module A under the locked year-block design.

This is a retrospective sensitivity analysis, not a new predictive result.
For an injection fraction ``lambda``, the locked prediction is moved toward
the observed residual:

    p_lambda = p + lambda * (y - p)

The transformation preserves the observed row and year-block structure.  A
nested year-block bootstrap then estimates how often the pre-specified Module A
rule (upper percentile 95% CI for delta RMSE below zero) would pass.  The MDE is
the first injected effect whose estimated pass probability reaches 80%.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data"
    / "legacy"
    / "state_panel_v1"
    / "snapshot"
    / "artifacts"
    / "audit"
    / "final_test"
    / "seed_aggregated_predictions.csv"
)
OUT_DIR = ROOT / "artifacts" / "experiments" / "kse-v5-8" / "mde"

SELECTED_CONFIG = "extra_trees_leaf_1"
SELECTED_FAMILY = "weather_only"
BASELINE_CONFIG = "zero_residual"
OUTER_REPLICATES = 1000
INNER_REPLICATES = 1000
OUTER_SEED = 2026073001
INNER_SEED = 2026073002
TARGET_POWER = 0.80
EFFECT_GRID = np.round(np.arange(0.0, 0.101, 0.001), 3)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_pair() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    selected = frame[
        (frame["config_id"] == SELECTED_CONFIG)
        & (frame["feature_family"] == SELECTED_FAMILY)
    ][["row_id", "year", "trend_residual_t_ha", "prediction"]].rename(
        columns={"prediction": "selected_prediction"}
    )
    baseline = frame[
        (frame["config_id"] == BASELINE_CONFIG)
        & (frame["feature_family"] == "baseline")
    ][["row_id", "prediction"]].rename(columns={"prediction": "baseline_prediction"})
    paired = selected.merge(baseline, on="row_id", validate="one_to_one")
    paired = paired.sort_values(["year", "row_id"], kind="stable").reset_index(drop=True)
    if len(paired) != 333:
        raise AssertionError(f"Expected 333 locked rows, found {len(paired)}")
    years = sorted(paired["year"].unique().tolist())
    if years != list(range(2016, 2026)):
        raise AssertionError(f"Unexpected locked year blocks: {years}")
    if not np.allclose(paired["baseline_prediction"], 0.0, atol=1e-12):
        raise AssertionError("Module A baseline is not the zero-residual vector")
    return paired


def block_sums(
    paired: pd.DataFrame, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = sorted(paired["year"].unique())
    y = paired["trend_residual_t_ha"].to_numpy(dtype=float)
    baseline = paired["baseline_prediction"].to_numpy(dtype=float)
    model_error_sq = (y - prediction) ** 2
    baseline_error_sq = (y - baseline) ** 2
    model_sse = []
    baseline_sse = []
    counts = []
    year_values = paired["year"].to_numpy()
    for year in years:
        mask = year_values == year
        model_sse.append(float(model_error_sq[mask].sum()))
        baseline_sse.append(float(baseline_error_sq[mask].sum()))
        counts.append(int(mask.sum()))
    return (
        np.asarray(model_sse, dtype=float),
        np.asarray(baseline_sse, dtype=float),
        np.asarray(counts, dtype=float),
    )


def main() -> None:
    paired = load_locked_pair()
    y = paired["trend_residual_t_ha"].to_numpy(dtype=float)
    original_prediction = paired["selected_prediction"].to_numpy(dtype=float)
    baseline_prediction = paired["baseline_prediction"].to_numpy(dtype=float)

    n_blocks = paired["year"].nunique()
    outer_rng = np.random.default_rng(OUTER_SEED)
    inner_rng = np.random.default_rng(INNER_SEED)
    outer_blocks = outer_rng.integers(
        0, n_blocks, size=(OUTER_REPLICATES, n_blocks), endpoint=False
    )
    inner_positions = inner_rng.integers(
        0,
        n_blocks,
        size=(OUTER_REPLICATES, INNER_REPLICATES, n_blocks),
        endpoint=False,
    )
    outer_broadcast = np.broadcast_to(
        outer_blocks[:, None, :],
        (OUTER_REPLICATES, INNER_REPLICATES, n_blocks),
    )
    sampled_original_blocks = np.take_along_axis(
        outer_broadcast, inner_positions, axis=2
    )

    rows: list[dict[str, float | int]] = []
    baseline_rmse = float(np.sqrt(np.mean((y - baseline_prediction) ** 2)))
    for effect_fraction in EFFECT_GRID:
        injected_prediction = original_prediction + effect_fraction * (
            y - original_prediction
        )
        model_sse, baseline_sse, counts = block_sums(paired, injected_prediction)
        sampled_n = counts[sampled_original_blocks].sum(axis=2)
        model_rmse = np.sqrt(
            model_sse[sampled_original_blocks].sum(axis=2) / sampled_n
        )
        zero_rmse = np.sqrt(
            baseline_sse[sampled_original_blocks].sum(axis=2) / sampled_n
        )
        delta_draws = model_rmse - zero_rmse
        upper_bounds = np.quantile(delta_draws, 0.975, axis=1)
        pass_probability = float(np.mean(upper_bounds < 0.0))

        point_model_rmse = float(np.sqrt(np.mean((y - injected_prediction) ** 2)))
        point_delta = point_model_rmse - baseline_rmse
        rows.append(
            {
                "effect_fraction": float(effect_fraction),
                "model_rmse_t_ha": point_model_rmse,
                "baseline_rmse_t_ha": baseline_rmse,
                "delta_rmse_t_ha": point_delta,
                "absolute_improvement_t_ha": -point_delta,
                "estimated_power": pass_probability,
                "outer_replicates": OUTER_REPLICATES,
                "inner_replicates": INNER_REPLICATES,
                "year_blocks": n_blocks,
            }
        )

    curve = pd.DataFrame(rows)
    qualifying = curve[curve["estimated_power"] >= TARGET_POWER]
    if qualifying.empty:
        raise RuntimeError("Effect grid did not reach 80% estimated power")
    mde = qualifying.iloc[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT_DIR / "module_a_mde_power_curve.csv", index=False)
    paired.to_csv(OUT_DIR / "module_a_locked_pair_used.csv", index=False)
    summary = {
        "analysis": "Module A nested year-block bootstrap power calibration",
        "status": "retrospective sensitivity; not a predictive result",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": file_sha256(SOURCE),
        "selected_config": SELECTED_CONFIG,
        "selected_family": SELECTED_FAMILY,
        "baseline": BASELINE_CONFIG,
        "n_rows": int(len(paired)),
        "year_blocks": int(n_blocks),
        "outer_replicates": OUTER_REPLICATES,
        "inner_replicates": INNER_REPLICATES,
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "target_power": TARGET_POWER,
        "injection_rule": "p_lambda = p + lambda * (y - p)",
        "mde_effect_fraction": float(mde["effect_fraction"]),
        "mde_delta_rmse_t_ha": float(mde["delta_rmse_t_ha"]),
        "mde_absolute_improvement_t_ha": float(mde["absolute_improvement_t_ha"]),
        "estimated_power_at_mde": float(mde["estimated_power"]),
    }
    (OUT_DIR / "module_a_mde_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
