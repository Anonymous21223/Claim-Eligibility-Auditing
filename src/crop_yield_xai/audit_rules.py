"""Config-driven decision rules used by the fidelity-gated audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_gate_config(path: Path) -> dict[str, object]:
    """Read JSON-compatible YAML to avoid a nonstandard parser dependency."""
    return json.loads(path.read_text(encoding="utf-8"))


def paired_error_pass(ci95_high: float) -> bool:
    """A model-minus-baseline error difference passes only below zero."""
    return bool(np.isfinite(ci95_high) and ci95_high < 0.0)


def tail_component_pass(row: dict[str, object], config: dict[str, object]) -> bool:
    """Evaluate the primary tail without invalid negative-only sign accuracy.

    The caller supplies null-aware rank and top-k component outcomes.  Severe
    tails are sensitivity rows unless selected as ``tail_policy.primary_tail``.
    """
    primary = str(config["tail_policy"]["primary_tail"])
    if str(row["threshold"]) != primary:
        return False
    return bool(
        paired_error_pass(float(row["paired_delta_rmse_ci95_high"]))
        and paired_error_pass(float(row["paired_delta_mae_ci95_high"]))
        and str(row.get("rank_recovery_status", "FAIL")) == "PASS"
        and str(row.get("topk_recovery_status", "FAIL")) == "PASS"
    )


def final_gate_status(components: dict[str, bool]) -> str:
    return "PASS" if all(components.values()) else "FAIL"


def top_k_recovery(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Overlap of the k most-negative observed and predicted residuals.

    Stable mergesort provides deterministic index-order handling for ties.
    """
    if k <= 0 or k > len(observed) or len(observed) != len(predicted):
        raise ValueError("k must be in [1, n] and vectors must have equal length")
    observed_top = set(np.argsort(observed, kind="mergesort")[:k])
    predicted_top = set(np.argsort(predicted, kind="mergesort")[:k])
    return len(observed_top & predicted_top) / k
