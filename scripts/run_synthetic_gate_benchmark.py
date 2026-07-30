"""Synthetic temporal benchmark for observable explanation-permission policies."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "experiments" / "synthetic-gate-benchmark"
REPORT = ROOT / "reports" / "experiments"
GENERATED = ROOT / "paper" / "generated"

REPEATS = 30
BASE_SEED = 4300
REQUESTED_CLAIM_LEVEL = "event_recovery"
WEATHER_COLUMNS = [0, 1, 2, 3, 7]
METADATA_COLUMNS = [4, 5, 6]
SCENARIOS = {
    "no_signal": 0.0,
    "weak_signal": 0.25,
    "moderate_signal": 0.7,
    "strong_signal": 1.4,
    "correlated_features": 0.7,
    "omitted_confounder": 0.0,
    "temporal_drift": 0.7,
    "geographic_shift": 0.7,
    "leakage": 0.0,
    "train_only_detrending": 0.7,
    "small_sample": 0.7,
    "imbalanced_tail": 0.7,
    "measurement_error": 0.7,
    "spatial_resolution_mismatch": 0.7,
}
GROUND_TRUTH_PERMISSION = {
    "moderate_signal",
    "strong_signal",
    "train_only_detrending",
    "small_sample",
    "imbalanced_tail",
    "spatial_resolution_mismatch",
}
NULL_OR_AMBIGUOUS_SCENARIOS = {
    "correlated_features",
    "omitted_confounder",
    "temporal_drift",
    "geographic_shift",
    "leakage",
    "measurement_error",
}
WEAK_OR_NULL_SIGNAL_SCENARIOS = {"no_signal", "weak_signal"}
REQUIRED_MODULES = {
    "overall": ("module_a_pass",),
    "weather_reliance": ("module_a_pass", "module_b_pass"),
    "event_recovery": ("module_a_pass", "module_b_pass", "module_e_pass"),
}


def decide_policy(observed_module_results: Mapping[str, bool], requested_claim_level: str) -> bool:
    """Return the observable policy decision from module outputs only."""
    if requested_claim_level not in REQUIRED_MODULES:
        raise ValueError(f"Unknown requested claim level: {requested_claim_level}")
    return all(bool(observed_module_results[module]) for module in REQUIRED_MODULES[requested_claim_level])


def sha_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def panel(name: str, effect: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 800 if name != "small_sample" else 180
    x = rng.normal(size=(n, 8))
    years = np.repeat(np.arange(20), n // 20)[:n]
    y = effect * x[:, 0] + rng.normal(size=n)
    if name == "correlated_features":
        x[:, 1] = x[:, 0] + rng.normal(scale=0.08, size=n)
    if name == "omitted_confounder":
        conf = rng.normal(size=n)
        x[:, 2] = conf + rng.normal(scale=0.1, size=n)
        y = 0.9 * conf + rng.normal(size=n)
    if name == "temporal_drift":
        y += (years >= 15) * 1.1 * x[:, 1]
    if name == "geographic_shift":
        shifted = np.arange(n) % 4 == 0
        x[:, 3] += shifted * 2
        y += shifted * 0.8 * x[:, 3]
    if name == "leakage":
        x[:, 7] = y + rng.normal(scale=0.01, size=n)
    if name == "imbalanced_tail":
        y += (x[:, 0] < -1.8) * (-2.5)
    if name == "measurement_error":
        x[:, 0] += rng.normal(scale=1.3, size=n)
    if name == "spatial_resolution_mismatch":
        x[:, 0] = np.repeat(x.reshape(-1, 4, 8).mean(axis=1)[:, 0], 4)[:n]
    return x, y, years


def evaluate_scenario(name: str, effect: float, repeat: int) -> dict[str, object]:
    seed = BASE_SEED + 1000 * repeat + list(SCENARIOS).index(name)
    x, y, _years = panel(name, effect, seed)
    cut = int(len(y) * 0.7)
    validation_cut = int(cut * 0.7)
    validation_train, validation = np.arange(validation_cut), np.arange(validation_cut, cut)
    train, test = np.arange(cut), np.arange(cut, len(y))
    model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, random_state=19 + repeat, n_jobs=-1)
    model.fit(x[train][:, WEATHER_COLUMNS], y[train])
    prediction = model.predict(x[test][:, WEATHER_COLUMNS])
    baseline_prediction = np.repeat(float(y[train].mean()), len(test))
    r2 = float(r2_score(y[test], prediction))
    selected_rmse = rmse(y[test], prediction)
    baseline_rmse = rmse(y[test], baseline_prediction)

    metadata_model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, random_state=1019 + repeat, n_jobs=-1)
    metadata_model.fit(x[train][:, METADATA_COLUMNS], y[train])
    metadata_prediction = metadata_model.predict(x[test][:, METADATA_COLUMNS])
    metadata_rmse = rmse(y[test], metadata_prediction)

    validation_model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, random_state=2019 + repeat, n_jobs=-1)
    validation_model.fit(x[validation_train][:, WEATHER_COLUMNS], y[validation_train])
    validation_prediction = validation_model.predict(x[validation][:, WEATHER_COLUMNS])
    validation_r2 = float(r2_score(y[validation], validation_prediction))

    tail_threshold = float(np.quantile(y[train], 0.25))
    tail_mask = y[test] <= tail_threshold
    n_tail = int(tail_mask.sum())
    if n_tail:
        tail_rmse_delta = rmse(y[test][tail_mask], prediction[tail_mask]) - rmse(y[test][tail_mask], baseline_prediction[tail_mask])
    else:
        tail_rmse_delta = float("nan")
    k = max(1, min(10, n_tail))
    predicted_tail_idx = np.argsort(prediction)[:k]
    observed_tail_idx = set(np.flatnonzero(tail_mask).tolist())
    topk_overlap = int(sum(int(idx in observed_tail_idx) for idx in predicted_tail_idx))
    topk_expected = float(k * n_tail / len(test)) if len(test) else 0.0
    topk_lift = float(topk_overlap / topk_expected) if topk_expected > 0 else 0.0

    module_a = r2 > 0.05
    module_b = (selected_rmse - metadata_rmse) < 0.0
    module_e = n_tail >= 5 and tail_rmse_delta < 0.0 and topk_lift > 1.0
    observed_modules = {
        "module_a_pass": module_a,
        "module_b_pass": module_b,
        "module_e_pass": module_e,
    }
    policy_permit = decide_policy(observed_modules, REQUESTED_CLAIM_LEVEL)
    ground_truth_permission = name in GROUND_TRUTH_PERMISSION
    failed_modules = [module for module in REQUIRED_MODULES[REQUESTED_CLAIM_LEVEL] if not observed_modules[module]]
    return {
        "scenario": name,
        "repeat": repeat,
        "seed": seed,
        "requested_claim_level": REQUESTED_CLAIM_LEVEL,
        "holdout_r2": r2,
        "validation_r2": validation_r2,
        "selected_rmse": selected_rmse,
        "baseline_rmse": baseline_rmse,
        "metadata_rmse": metadata_rmse,
        "module_b_delta_rmse": selected_rmse - metadata_rmse,
        "tail_threshold": tail_threshold,
        "n_tail": n_tail,
        "tail_rmse_delta": tail_rmse_delta,
        "topk_overlap": topk_overlap,
        "topk_expected": topk_expected,
        "topk_lift": topk_lift,
        "target_sha256": sha_array(y[test]),
        "prediction_sha256": sha_array(prediction),
        "metadata_prediction_sha256": sha_array(metadata_prediction),
        "baseline_prediction_sha256": sha_array(baseline_prediction),
        "ground_truth_permission": ground_truth_permission,
        "module_a_pass": module_a,
        "module_b_pass": module_b,
        "module_e_pass": module_e,
        "policy_permit": policy_permit,
        "abstention_reason": "permit" if policy_permit else ";".join(failed_modules),
        "true_permission": bool(policy_permit and ground_truth_permission),
        "false_permission": bool(policy_permit and not ground_truth_permission),
        "true_abstention": bool((not policy_permit) and not ground_truth_permission),
        "false_abstention": bool((not policy_permit) and ground_truth_permission),
        "validation_only_permission": validation_r2 > 0.05,
        "module_a_only_permission": module_a,
        "module_a_b_permission": module_a and module_b,
        "module_a_e_permission": module_a and module_e,
        "module_a_b_e_permission": module_a and module_b and module_e,
        "ungated_permission": True,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def summarize_rule(frame: pd.DataFrame, rule: str, permission_col: str) -> dict[str, object]:
    invalid = ~frame.ground_truth_permission
    valid = frame.ground_truth_permission
    false_permission = frame[invalid & frame[permission_col]]
    false_abstention = frame[valid & ~frame[permission_col]]
    true_permission = frame[valid & frame[permission_col]]
    true_abstention = frame[invalid & ~frame[permission_col]]
    fp_low, fp_high = wilson(len(false_permission), int(invalid.sum()))
    fa_low, fa_high = wilson(len(false_abstention), int(valid.sum()))
    return {
        "rule": rule,
        "permission_column": permission_col,
        "n_invalid": int(invalid.sum()),
        "n_valid": int(valid.sum()),
        "false_permission_count": int(len(false_permission)),
        "false_permission_rate": float(len(false_permission) / invalid.sum()),
        "false_permission_ci95_low": fp_low,
        "false_permission_ci95_high": fp_high,
        "false_abstention_count": int(len(false_abstention)),
        "false_abstention_rate": float(len(false_abstention) / valid.sum()),
        "false_abstention_ci95_low": fa_low,
        "false_abstention_ci95_high": fa_high,
        "sensitivity": float(len(true_permission) / valid.sum()),
        "specificity": float(len(true_abstention) / invalid.sum()),
        "permission_rate": float(frame[permission_col].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(
        evaluate_scenario(name, effect, repeat)
        for repeat in range(REPEATS)
        for name, effect in SCENARIOS.items()
    )
    runs["correct_policy_decision"] = runs.policy_permit == runs.ground_truth_permission

    ground_truth = pd.DataFrame(
        [
            {
                "scenario": name,
                "effect": effect,
                "ground_truth_permission": name in GROUND_TRUTH_PERMISSION,
                "requested_claim_level": REQUESTED_CLAIM_LEVEL,
                "reason": "valid signal scenario"
                if name in GROUND_TRUTH_PERMISSION
                else "invalid, too weak, ambiguous, or not interpretable",
            }
            for name, effect in SCENARIOS.items()
        ]
    )
    rules = [
        ("ungated", "ungated_permission"),
        ("Validation-only policy", "validation_only_permission"),
        ("Module A only", "module_a_only_permission"),
        ("Module A + Module B", "module_a_b_permission"),
        ("Module A + Module E", "module_a_e_permission"),
        ("Module A + Module B + Module E", "module_a_b_e_permission"),
        ("Observable policy", "policy_permit"),
    ]
    summary = pd.DataFrame(summarize_rule(runs, rule, column) for rule, column in rules)
    scenario_summary = (
        runs.groupby("scenario", as_index=False)
        .agg(
            ground_truth_permission=("ground_truth_permission", "first"),
            requested_claim_level=("requested_claim_level", "first"),
            module_a_pass_rate=("module_a_pass", "mean"),
            module_b_pass_rate=("module_b_pass", "mean"),
            module_e_pass_rate=("module_e_pass", "mean"),
            policy_permit_rate=("policy_permit", "mean"),
            correct_policy_decision_rate=("correct_policy_decision", "mean"),
            false_permission_count=("false_permission", "sum"),
            false_abstention_count=("false_abstention", "sum"),
            holdout_r2_mean=("holdout_r2", "mean"),
            holdout_r2_sd=("holdout_r2", "std"),
        )
        .merge(ground_truth[["scenario", "reason"]], on="scenario", how="left")
    )

    scenario_yaml = [
        "# Synthetic scenarios for the fidelity-gated explanation benchmark.",
        f"repeats_per_scenario: {REPEATS}",
        "scenarios:",
    ]
    for _, row in ground_truth.iterrows():
        scenario_yaml.extend(
            [
                f"  - scenario: {row['scenario']}",
                f"    effect: {row['effect']}",
                f"    ground_truth_permission: {str(bool(row['ground_truth_permission'])).lower()}",
                f"    requested_claim_level: {row['requested_claim_level']}",
                f"    reason: {row['reason']}",
            ]
        )
    (OUT / "synthetic_scenarios.yaml").write_text("\n".join(scenario_yaml) + "\n", encoding="utf-8")

    runs.to_csv(OUT / "synthetic_runs_long.csv", index=False)
    ground_truth.to_csv(OUT / "synthetic_ground_truth.csv", index=False)
    summary.to_csv(OUT / "gate_component_ablation.csv", index=False)
    summary.to_csv(OUT / "synthetic_component_ablation.csv", index=False)
    summary.to_csv(OUT / "synthetic_summary.csv", index=False)
    summary.to_csv(OUT / "synthetic_summary_ci.csv", index=False)
    scenario_summary.to_csv(OUT / "scenario_level_decisions.csv", index=False)
    scenario_summary.to_csv(OUT / "scenario_results.csv", index=False)
    scenario_table = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Scenario & GT permit & Claim & A & B & E & Policy permit \\",
        r"\midrule",
    ]
    for _, row in scenario_summary.iterrows():
        scenario_table.append(
            f"{latex_escape(row['scenario'])} & "
            f"{'yes' if row['ground_truth_permission'] else 'no'} & "
            f"{latex_escape(row['requested_claim_level'])} & "
            f"{100 * row['module_a_pass_rate']:.0f}\\% & "
            f"{100 * row['module_b_pass_rate']:.0f}\\% & "
            f"{100 * row['module_e_pass_rate']:.0f}\\% & "
            f"{100 * row['policy_permit_rate']:.0f}\\% \\\\"
        )
    scenario_table.extend([r"\bottomrule", r"\end{tabular}"])
    (GENERATED / "table_synthetic_scenario_decisions.tex").write_text(
        "\n".join(scenario_table) + "\n", encoding="utf-8"
    )

    full = summary[summary.rule == "Observable policy"].iloc[0]
    ungated = summary[summary.rule == "ungated"].iloc[0]
    confusion = {
        "tp": int((runs["policy_permit"] & runs["ground_truth_permission"]).sum()),
        "fp": int((runs["policy_permit"] & ~runs["ground_truth_permission"]).sum()),
        "tn": int((~runs["policy_permit"] & ~runs["ground_truth_permission"]).sum()),
        "fn": int((~runs["policy_permit"] & runs["ground_truth_permission"]).sum()),
    }
    payload = {
        "status": "PASS",
        "scenarios": len(SCENARIOS),
        "repeats_per_scenario": REPEATS,
        "runs": int(len(runs)),
        "requested_claim_level": REQUESTED_CLAIM_LEVEL,
        "valid_ground_truth_runs": int(full.n_valid),
        "invalid_ground_truth_runs": int(full.n_invalid),
        "policy_formula": "module_a_pass AND module_b_pass AND module_e_pass",
        "policy_uses_gt_or_oracle": False,
        "tp": confusion["tp"],
        "fp": confusion["fp"],
        "tn": confusion["tn"],
        "fn": confusion["fn"],
        "ungated_false_permission_rate": float(ungated.false_permission_rate),
        "ungated_false_permission_ci95": [float(ungated.false_permission_ci95_low), float(ungated.false_permission_ci95_high)],
        "observable_policy_false_permission_rate": float(full.false_permission_rate),
        "observable_policy_false_permission_ci95": [float(full.false_permission_ci95_low), float(full.false_permission_ci95_high)],
        "observable_policy_false_abstention_rate": float(full.false_abstention_rate),
        "observable_policy_false_abstention_ci95": [float(full.false_abstention_ci95_low), float(full.false_abstention_ci95_high)],
        "observable_policy_permission_rate": float(full.permission_rate),
        "observable_policy_sensitivity": float(full.sensitivity),
        "observable_policy_specificity": float(full.specificity),
        "criterion": "observable policy is scored against GT labels without using GT, scenario name, hidden DGP parameters, or oracle invalidity in the policy decision",
    }
    payload["status"] = "PASS" if (
        payload["runs"] == len(SCENARIOS) * REPEATS
        and payload["valid_ground_truth_runs"] == len(GROUND_TRUTH_PERMISSION) * REPEATS
        and payload["invalid_ground_truth_runs"] == (len(SCENARIOS) - len(GROUND_TRUTH_PERMISSION)) * REPEATS
        and confusion["tp"] + confusion["fp"] + confusion["tn"] + confusion["fn"] == len(runs)
    ) else "FAIL"
    policy_schema = {
        "decision_function": "decide_policy(observed_module_results, requested_claim_level)",
        "allowed_inputs": ["module_a_pass", "module_b_pass", "module_e_pass", "requested_claim_level"],
        "forbidden_inputs": ["ground_truth_permission", "scenario", "effect", "hidden_dgp", "admissibility_label", "driver", "valid"],
        "required_modules": REQUIRED_MODULES,
    }
    (OUT / "observable_policy_schema.json").write_text(json.dumps(policy_schema, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (REPORT / "synthetic-gate-benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = ["# Synthetic Gate Benchmark", ""]
    report.extend(f"- {key}: `{value}`" for key, value in payload.items())
    (REPORT / "synthetic-gate-benchmark.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if payload["status"] != "PASS":
        raise SystemExit("Synthetic gate benchmark failed")


if __name__ == "__main__":
    main()
