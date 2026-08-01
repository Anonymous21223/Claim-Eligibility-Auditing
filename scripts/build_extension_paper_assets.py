"""Generate V6.1 anonymous-paper LaTeX assets from locked artifacts."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "artifacts" / "extensions" / "v1" / "core"
EXT = CORE.parent
DEST = ROOT / "paper_versions" / "v6_1_extension_factorial_anonymous" / "source" / "generated"
AUDIT = ROOT / "reports" / "extension_factorial_v6_1_release" / "artifact_audit.json"


def write(name: str, text: str) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def esc(value: object) -> str:
    return str(value).replace("_", r"\_")


def latex_table(columns: str, header: str, rows: list[str]) -> str:
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *[row + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def main() -> None:
    lock = json.loads((CORE / "winner_lock.json").read_text())
    leaderboard = pd.read_csv(CORE / "validation_leaderboard.csv")
    folds = pd.read_csv(CORE / "outer_fold_metrics.csv")
    modules = pd.read_csv(CORE / "module_decisions.csv")
    semi = pd.read_csv(EXT / "semi" / "scenario_results.csv")
    pjm = pd.read_csv(EXT / "pjm" / "degradation_curve.csv")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["decision"] != "NO_RERUN":
        raise ValueError("Submission assets require a passing stored-artifact audit")
    counts = audit["checks"]["candidate_wide_counts"]["evidence"]
    event_full = audit["checks"]["winner_modules_reproduced"]["evidence"]["event_full"]
    winner = lock["winner"]
    mask = (
        (leaderboard.arm_id == winner["arm_id"])
        & (leaderboard.model_id == winner["model_id"])
        & (leaderboard.feature_family == winner["feature_family"])
    )
    row = leaderboard[mask].iloc[0]
    module_mask = (
        (modules.arm_id == winner["arm_id"])
        & (modules.model_id == winner["model_id"])
        & (modules.feature_family == winner["feature_family"])
    )
    if mask.sum() != 1 or module_mask.sum() != 1:
        raise ValueError("Locked winner must resolve to exactly one leaderboard and module row")
    module = modules[module_mask].iloc[0]

    macros = {
        "ExtWinnerArm": winner["arm_id"],
        "ExtWinnerModel": "CatBoost",
        "ExtWinnerFamily": "Full",
        "ExtRows": f"{int(row['n']):,}",
        "ExtRMSE": f"{row['pooled_rmse']:.3f}",
        "ExtMAE": f"{row['pooled_mae']:.3f}",
        "ExtRtwo": f"{row['pooled_r2']:.3f}",
        "ExtPositiveFolds": int(row["positive_r2_folds"]),
        "ExtGateADelta": f"{module['module_a_delta_rmse']:.3f}",
        "ExtGateALow": f"{module['module_a_ci95_low']:.3f}",
        "ExtGateAHigh": f"{module['module_a_ci95_high']:.3f}",
        "ExtGateBDelta": f"{module['module_b_delta_rmse']:.3f}",
        "ExtGateBLow": f"{module['module_b_ci95_low']:.3f}",
        "ExtGateBHigh": f"{module['module_b_ci95_high']:.3f}",
        "ExtEventRows": int(module["event_n"]),
        "ExtEventRho": f"{module['event_spearman']:.3f}",
        "ExtEventTopKLift": f"{module['event_topk_lift']:.2f}",
        "ExtTailRMSEDelta": f"{event_full['tail_rmse']['estimate']:.3f}",
        "ExtTailRMSELow": f"{event_full['tail_rmse']['ci95_low']:.3f}",
        "ExtTailRMSEHigh": f"{event_full['tail_rmse']['ci95_high']:.3f}",
        "ExtTailMAEDelta": f"{event_full['tail_mae']['estimate']:.3f}",
        "ExtTailMAELow": f"{event_full['tail_mae']['ci95_low']:.3f}",
        "ExtTailMAEHigh": f"{event_full['tail_mae']['ci95_high']:.3f}",
        "ExtPassRone": f"{counts['pass_r1']}/{counts['pipelines']}",
        "ExtPassA": f"{counts['pass_a']}/{counts['pipelines']}",
        "ExtPassB": f"{counts['pass_b_full']}/{counts['full_pipelines']}",
        "ExtPassAB": f"{counts['pass_a_b_full']}/{counts['full_pipelines']}",
        "ExtPassE": f"{counts['pass_e']}/{counts['evaluated_e']}",
        "ExtSemiRows": f"{len(semi):,}",
        "ExtPJMRows": f"{int(pjm['seeds'].sum()):,}",
    }
    write(
        "extension_numbers.tex",
        "\n".join(
            f"\\newcommand{{\\{key}}}{{{value}}}"
            for key, value in macros.items()
        ),
    )

    design_rows = [
        r"BASE & 4 & 35 full-season & Reference",
        r"STAGE-CAL & 4 & 15 calendar-stage & Fixed progress dates",
        r"STAGE-GDD & 4 & 15 GDD-stage & Heat accumulation",
        r"STAGE-HYBRID & 4 & 65 combined & BASE + CAL + GDD",
    ]
    write(
        "table_design.tex",
        latex_table(
            "lrrl",
            r"Representation & Arms & Weather vars. & Role",
            design_rows,
        ),
    )

    ranked = leaderboard[leaderboard.winner_rank.notna()].sort_values("winner_rank")
    ranking_rows = []
    for value in ranked.itertuples(index=False):
        ranking_rows.append(
            f"{int(value.winner_rank)} & {value.arm_id} & {esc(value.model_id)} & "
            f"{esc(value.feature_family)} & {value.pooled_rmse:.3f} & "
            f"{value.pooled_r2:.3f} & {int(value.positive_r2_folds)}/6 & "
            f"{value.claim_tier}"
        )
    ranking_rows.extend(
        [
            r"\midrule\multicolumn{8}{l}{Candidate-wide passes: R1 38/240; A 122/240; B 46/80 full;}",
            r"\multicolumn{8}{l}{A+B 46/80 full; E 0/46 evaluated.}",
        ]
    )
    write(
        "table_locked_ranking.tex",
        latex_table(
            "clllrrrr",
            r"Rank & Arm & Model & Family & RMSE & $R^2$ & $R^2{>}0$ & Tier",
            ranking_rows,
        ),
    )

    winner_folds = folds[
        (folds.arm_id == winner["arm_id"])
        & (folds.model_id == winner["model_id"])
        & (folds.feature_family == winner["feature_family"])
    ]
    intervals = {
        "O1": "2008--11",
        "O2": "2012--14",
        "O3": "2015--17",
        "O4": "2018--20",
        "O5": "2021--23",
        "O6": "2024--25",
    }
    fold_rows = [
        f"{value.outer_fold} & {intervals[value.outer_fold]} & "
        f"{value.rmse:.3f} & {value.mae:.3f} & {value.r2:.3f}"
        for value in winner_folds.itertuples(index=False)
    ]
    fold_rows += [
        f"Pooled & 2008--25 & {row.pooled_rmse:.3f} & "
        f"{row.pooled_mae:.3f} & {row.pooled_r2:.3f}"
    ]
    write(
        "table_fold_metrics.tex",
        latex_table("lrrrr", r"Fold & Test years & RMSE & MAE & $R^2$", fold_rows),
    )

    module_rows = [
        f"A & Full $-$ zero & {module.module_a_delta_rmse:.3f} & "
        f"{module.module_a_ci95_low:.3f} & {module.module_a_ci95_high:.3f}",
        f"B & Full $-$ metadata & {module.module_b_delta_rmse:.3f} & "
        f"{module.module_b_ci95_low:.3f} & {module.module_b_ci95_high:.3f}",
        f"E & Tail RMSE & {event_full['tail_rmse']['estimate']:.3f} & "
        f"{event_full['tail_rmse']['ci95_low']:.3f} & {event_full['tail_rmse']['ci95_high']:.3f}",
        f"E & Tail MAE & {event_full['tail_mae']['estimate']:.3f} & "
        f"{event_full['tail_mae']['ci95_low']:.3f} & {event_full['tail_mae']['ci95_high']:.3f}",
    ]
    write(
        "table_modules.tex",
        latex_table(
            "llrrr",
            r"Module & Comparison & Estimate & 95\% low & 95\% high",
            module_rows,
        ),
    )

    base = semi[semi.event_prevalence == "base"]
    summary = (
        base.groupby(["candidate_id", "signal_strength_mde"], as_index=False)
        .policy_permission.mean()
    )
    semi_rows = []
    for candidate in ["winner", "runner_up_1", "runner_up_2", "kse_primary"]:
        scope = (
            summary[summary.candidate_id == candidate]
            .set_index("signal_strength_mde")
            .policy_permission
        )
        semi_rows.append(
            f"{esc(candidate)} & {100 * scope[0]:.1f}\\% & "
            f"{100 * scope[1]:.1f}\\% & {100 * scope[1.5]:.1f}\\%"
        )
    write(
        "table_semi_power.tex",
        latex_table(
            "lrrr",
            r"Candidate & 0 MDE & 1 MDE & 1.5 MDE",
            semi_rows,
        ),
    )

    pjm_rows = []
    for mechanism, group in pjm.groupby("mechanism", sort=True):
        group = group.sort_values("level")
        pjm_rows.append(
            f"{esc(mechanism)} & {group.level.iloc[-1]:g} & "
            f"{group.module_b_pass_probability.iloc[0]:.2f} & "
            f"{group.module_b_pass_probability.iloc[-1]:.2f}"
        )
    write(
        "table_pjm.tex",
        latex_table(
            "lrrr",
            r"Mechanism & Max. level & $P(B)$ at 0 & $P(B)$ at max.",
            pjm_rows,
        ),
    )

    files = sorted(DEST.glob("*.tex"))
    manifest = {
        "source": "artifacts/extensions/v1",
        "winner_lock_sha256": sha256(
            (CORE / "winner_lock.json").read_bytes()
        ).hexdigest(),
        "artifact_audit_sha256": sha256(AUDIT.read_bytes()).hexdigest(),
        "files": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (DEST / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
