"""Canonical ICTAI2026 audit runner.

It reuses the locked-split prediction pipeline and adds the null-aware Gate A/B
experiments required for the final paper.  No test-period model selection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_yield_xai.null_audit import rank_null_audit, topk_null_audit  # noqa: E402
from run_main8_audit import GATE_CONFIG, main as locked_pipeline, paired_delta  # noqa: E402
from run_expanded_models import main as expanded_models  # noqa: E402
from run_extended_audits import main as extended_audits  # noqa: E402


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def topk_pass(result: dict[str, float], alpha: float) -> bool:
    """The pre-specified conjunction for chance-adjusted top-k recovery."""
    return bool(result["lift"] > 1 and result["lift_ci95_low"] > 1 and result["hypergeometric_pvalue"] < alpha and result["permutation_pvalue"] < alpha)


def event_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, float]:
    labels = (predictions.trend_residual_z < threshold).astype(int).to_numpy()
    scores = -predictions.prediction.to_numpy()
    # The operational score threshold is locked at zero residual before final evaluation.
    decisions = (scores >= 0.0).astype(int)
    tp = int(((labels == 1) & (decisions == 1)).sum())
    fp = int(((labels == 0) & (decisions == 1)).sum())
    tn = int(((labels == 0) & (decisions == 0)).sum())
    fn = int(((labels == 1) & (decisions == 0)).sum())
    return {
        "n": len(predictions), "prevalence": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
        "precision": float(precision_score(labels, decisions, zero_division=0)),
        "recall": float(recall_score(labels, decisions, zero_division=0)),
        "f1": float(f1_score(labels, decisions, zero_division=0)),
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "decision_score_threshold": 0.0,
    }


def run_null_experiments() -> None:
    audit = ROOT / "artifacts" / "audit"
    records = ROOT / "artifacts" / "audit_records"
    tables = ROOT / "artifacts" / "tables"
    figures = ROOT / "artifacts" / "figures"
    aggregate = pd.read_csv(audit / "final_test" / "seed_aggregated_predictions.csv")
    selected = json.loads((audit / "selection" / "selected_config.json").read_text(encoding="utf-8"))["selected_config"]
    model = aggregate[(aggregate.config_id == selected["config_id"]) & (aggregate.feature_family == selected["feature_family"])].copy()
    paired = pd.read_csv(audit / "bootstrap" / "paired_feature_family_ci.csv")
    draws = pd.read_csv(audit / "bootstrap" / "year_block_draws.csv")
    tail_rows, rank_rows, event_rows = [], [], []
    permutations = int(GATE_CONFIG["topk_recovery"]["permutations"])
    n_boot = int(GATE_CONFIG["bootstrap"]["replicates"])
    alpha = float(GATE_CONFIG["topk_recovery"]["alpha"])
    for threshold in (-1.0, -1.5, -2.0):
        label = f"z<{threshold:g}"
        tail = model[model.trend_residual_z < threshold].copy()
        sensitivity_k = [max(1, int(np.ceil(fraction * len(tail)))) for fraction in GATE_CONFIG["topk_recovery"]["fraction_values"]]
        for k in sorted({*GATE_CONFIG["topk_recovery"]["k_values"], *sensitivity_k}):
            result = topk_null_audit(tail.trend_residual_t_ha.to_numpy(), tail.prediction.to_numpy(), tail.year.to_numpy(), int(k), n_boot, permutations, 20260718 + int(k * 10 + abs(threshold) * 100))
            result.update({"threshold": label, "definition": f"k={k}"})
            result["status"] = status(topk_pass(result, alpha))
            tail_rows.append(result)
        rank = rank_null_audit(tail.trend_residual_t_ha.to_numpy(), tail.prediction.to_numpy(), tail.year.to_numpy(), n_boot, permutations, 20260718 + int(abs(threshold) * 1000))
        rank.update({"threshold": label})
        rank["status"] = status(rank["spearman_ci95_low"] > 0 and rank["permutation_pvalue"] < alpha)
        rank_rows.append(rank)
        row = event_metrics(model, threshold)
        row.update({"threshold": label, "scope": "locked_final_test", "status": "DIAGNOSTIC"})
        event_rows.append(row)
    topk = pd.DataFrame(tail_rows)
    rank = pd.DataFrame(rank_rows)
    events = pd.DataFrame(event_rows)
    write(topk, records / "topk_null_audit.csv")
    write(rank, records / "rank_null_audit.csv")
    write(events, records / "event_detection_metrics.csv")
    # The top-level artifact is intentionally easy to inspect outside the paper build.
    write(events, ROOT / "artifacts" / "event_detection_metrics.csv")

    primary = str(GATE_CONFIG["tail_policy"]["primary_tail"])
    primary_topk = topk[(topk.threshold == primary) & (topk.definition == "k=10")].iloc[0]
    primary_rank = rank[rank.threshold == primary].iloc[0]
    tail_metric = pd.read_csv(audit / "tail" / "tail_metrics_by_threshold.csv").query("threshold == @primary").iloc[0]
    selected_delta = paired[(paired.comparison == f"{selected['config_id']}_{selected['feature_family']}_vs_zero") & (paired.metric == "rmse_t_ha")].iloc[0]
    overall = float(selected_delta.ci95_high) < 0
    rmse = float(tail_metric.paired_delta_rmse_ci95_high) < 0
    mae = float(tail_metric.paired_delta_mae_ci95_high) < 0
    rank_gate_pass, topk_gate_pass = primary_rank.status == "PASS", primary_topk.status == "PASS"
    gate_a = overall and rmse and mae and rank_gate_pass and topk_gate_pass
    gate_rows = [
        {"gate": "Gate A", "component": "overall paired RMSE", "scope": "locked final test", "status": status(overall)},
        {"gate": "Gate A", "component": "primary-tail paired RMSE", "scope": primary, "status": status(rmse)},
        {"gate": "Gate A", "component": "primary-tail paired MAE", "scope": primary, "status": status(mae)},
        {"gate": "Gate A", "component": "rank recovery", "scope": primary, "status": status(rank_gate_pass)},
        {"gate": "Gate A", "component": "chance-adjusted top-k recovery", "scope": primary, "status": status(topk_gate_pass)},
    ]
    metadata = aggregate[(aggregate.config_id == selected["config_id"]) & (aggregate.feature_family == "metadata_only")]
    full = aggregate[(aggregate.config_id == selected["config_id"]) & (aggregate.feature_family == "full")]
    weather = aggregate[(aggregate.config_id == selected["config_id"]) & (aggregate.feature_family == "weather_only")]
    full_vs_metadata = paired_delta(full, metadata, draws, f"{selected['config_id']}_full_vs_metadata_only")
    weather_vs_metadata = paired_delta(weather, metadata, draws, f"{selected['config_id']}_weather_only_vs_metadata_only")
    refreshed = {f"{selected['config_id']}_full_vs_metadata_only", f"{selected['config_id']}_weather_only_vs_metadata_only"}
    paired = pd.concat([paired[~paired.comparison.isin(refreshed)], full_vs_metadata, weather_vs_metadata], ignore_index=True)
    write(paired, records / "paired_comparisons.csv")
    primary_b = f"{selected['config_id']}_full_vs_metadata_only"
    diagnostic_b = f"{selected['config_id']}_weather_only_vs_metadata_only"
    decisions = {}
    for comparison, gate_name, role in ((primary_b, "Gate B1", "primary"), (diagnostic_b, "Gate B2", "diagnostic")):
        item = paired[(paired.comparison == comparison) & (paired.metric == "rmse_t_ha")]
        if not item.empty:
            passed = float(item.iloc[0].ci95_high) < 0
            decisions[gate_name] = {"comparison": comparison, "role": role, "n": int(item.iloc[0].n), "delta_left_minus_right": float(item.iloc[0].delta_left_minus_right), "ci95_low": float(item.iloc[0].ci95_low), "ci95_high": float(item.iloc[0].ci95_high), "status": status(passed)}
            gate_rows.append({"gate": gate_name, "component": comparison, "scope": "locked final test", "status": status(passed)})
    component = pd.DataFrame(gate_rows)
    gate_b = decisions.get("Gate B1", {}).get("status") == "PASS"
    gate_dir = ROOT / "artifacts" / "gates"; gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "gate_b_decision.json").write_text(json.dumps({"final_weather_specific_claim_requires": ["Gate A", "Gate B1"], "gate_b1": decisions.get("Gate B1"), "gate_b2_diagnostic": decisions.get("Gate B2")}, indent=2) + "\n", encoding="utf-8")
    component = pd.concat([component, pd.DataFrame([
        {"gate": "Gate A", "component": "FINAL GATE A", "scope": primary, "status": status(gate_a)},
        {"gate": "Gate B1", "component": "FINAL GATE B1", "scope": "weather-specific claim", "status": status(gate_b)},
    ])], ignore_index=True)
    write(component, records / "fidelity_gate_components.csv")
    write(component, tables / "fidelity_gate_components.csv")
    matrix = pd.DataFrame([
        {"gate_a": "PASS", "gate_b": "PASS", "allowed_claim": "event-level and weather-specific claims"},
        {"gate_a": "PASS", "gate_b": "FAIL", "allowed_claim": "event-level predictive claim only; no weather-specific claim"},
        {"gate_a": "FAIL", "gate_b": "PASS", "allowed_claim": "no substantive observed-event claim"},
        {"gate_a": "FAIL", "gate_b": "FAIL", "allowed_claim": "no substantive observed-event or weather-specific claim"},
    ])
    write(matrix, tables / "gate_decision_matrix.csv")
    snapshot = json.dumps(GATE_CONFIG, sort_keys=True).encode("utf-8")
    (records / "gate_config_sha256.txt").write_text(hashlib.sha256(snapshot).hexdigest() + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    primary_rows = topk[topk.threshold == primary]
    ax.bar(primary_rows.definition, primary_rows.lift, color="#4b6e9b", hatch="//")
    ax.axhline(1, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("Top-k lift over random expectation")
    ax.set_title("Primary-tail chance-adjusted recovery")
    fig.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "topk_null_distribution.pdf")
    plt.close(fig)
    print(f"Null-aware audit complete: Gate A={status(gate_a)}, Gate B={status(gate_b)}")


def run_selection_and_baseline_records() -> None:
    """E05/E06 records from the already locked validation and baseline vectors."""
    audit = ROOT / "artifacts" / "audit"
    records = ROOT / "artifacts" / "audit_records"
    seed = pd.read_csv(audit / "selection" / "validation_seed_metrics.csv")
    ranked = seed.copy()
    ranked["seed_rank"] = ranked.groupby("seed")["rmse_t_ha"].rank(method="min")
    stability = ranked.groupby(["config_id", "model", "feature_family"], as_index=False).agg(
        seed_count=("seed", "nunique"), rmse_mean=("rmse_t_ha", "mean"), rmse_sd=("rmse_t_ha", "std"),
        first_rank_count=("seed_rank", lambda values: int((values == 1).sum())),
    )
    stability["rank_first_probability"] = stability.first_rank_count / stability.seed_count
    stability["one_standard_error_eligible"] = stability.rmse_mean <= stability.rmse_mean.min() + stability.rmse_sd.min()
    write(stability.sort_values(["rmse_mean", "config_id"]), records / "validation_stability.csv")
    baseline = pd.read_csv(audit / "final_test" / "baseline_prediction_audit.csv")
    write(baseline, records / "baseline_vector_hashes.csv")
    selected = json.loads((audit / "selection" / "selected_config.json").read_text(encoding="utf-8"))["selected_config"]
    locked = {
        "selected_config": selected["config_id"], "feature_family": selected["feature_family"],
        "validation_only": True, "locked_final_test_access_for_selection": False,
        "validation_config_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    (records / "locked_test_access.json").write_text(json.dumps(locked, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fidelity_gate.yaml")
    parser.add_argument("--stage", default="all", choices=("all", "core"))
    parser.parse_args()
    locked_pipeline()
    run_null_experiments()
    run_selection_and_baseline_records()
    expanded_models()
    extended_audits()


if __name__ == "__main__":
    main()
