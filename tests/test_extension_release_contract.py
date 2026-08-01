from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "extensions" / "v1"
CORE = OUT / "core"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_core_completion_and_winner_lock_contract():
    matrix = pd.read_csv(OUT / "COMPLETION_MATRIX.csv")
    assert matrix.groupby("phase").size().to_dict() == {"inner": 34560, "outer": 4896}
    assert matrix["status"].eq("PASS").all()
    assert matrix["completion_fraction"].eq(1).all()
    leaderboard = pd.read_csv(CORE / "validation_leaderboard.csv")
    assert len(leaderboard) == 240
    assert leaderboard["arm_id"].nunique() == 16
    assert leaderboard[["arm_id", "model_id"]].drop_duplicates().shape[0] == 80
    assert leaderboard["winner_rank"].notna().sum() == 3
    lock = json.loads((CORE / "winner_lock.json").read_text())
    assert lock["status"] == "LOCKED"
    assert lock["winner"]["arm_id"] == "C310"
    assert lock["winner"]["claim_tier"] == "A+B"
    assert lock["selection_used_external_results"] is False
    assert lock["leaderboard_sha256"] == digest(CORE / "validation_leaderboard.csv")


def test_predictions_and_external_suites_contract():
    predictions = pd.read_parquet(CORE / "outer_predictions.parquet")
    required = {
        "arm_id", "model_id", "feature_family", "outer_fold", "seed",
        "y_true", "y_pred", "baseline_pred", "row_hash",
    }
    assert required.issubset(predictions.columns)
    assert predictions["outer_fold"].nunique() == 6
    assert not predictions[["y_true", "y_pred", "baseline_pred"]].isna().any().any()
    semi = pd.read_csv(OUT / "semi" / "scenario_results.csv")
    assert len(semi) == 28800
    assert semi["base_id"].nunique() == 9600
    assert semi["seed"].nunique() == 30
    assert not semi["ground_truth_used_as_model_input"].any()
    pjm = pd.read_csv(OUT / "pjm" / "scenario_results.csv")
    assert len(pjm) == 600
    zero = pjm[pjm["level"] == 0].groupby("mechanism")["delta_rmse_b"].mean()
    assert zero.nunique() == 1


def test_manifest_hashes_and_qa_contract():
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert manifest["external_results_used_for_winner_selection"] is False
    for relative, expected in manifest["files"].items():
        assert digest(ROOT / relative) == expected
    qa = json.loads((OUT / "QA_REPORT.json").read_text())
    assert qa["status"] == "PASS"
    assert all(qa["checks"].values())
