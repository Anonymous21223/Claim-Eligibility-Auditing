from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "derived" / "stage_features" / "cutoff_registry.json"


def test_stage_registry_has_every_frozen_cutoff():
    protocol = json.loads(
        (ROOT / "configs" / "extension_factorial_v1.yaml").read_text()
    )
    expected = {
        int(fold["train_end"]) for fold in protocol["outer_folds"]
    } | {
        int(inner["train_end"])
        for fold in protocol["outer_folds"]
        for inner in fold["inner"]
    }
    registry = json.loads(REGISTRY.read_text())
    assert {int(row["cutoff_year"]) for row in registry} == expected
    assert all(row["status"] == "PASS" for row in registry)


def test_stage_features_are_complete_and_have_unique_keys():
    for manifest in json.loads(REGISTRY.read_text()):
        folder = REGISTRY.parent / f"cutoff_{manifest['cutoff_year']}"
        frame = pd.read_csv(folder / "stage_features.csv")
        assert len(frame) == 1257
        assert frame[["crop", "region", "year", "window"]].drop_duplicates().shape[0] == 1257
        assert not frame.isna().any().any()
        stage_columns = [
            column
            for column in frame
            if column not in {"country", "crop", "region", "year", "window"}
        ]
        assert len(stage_columns) == 30


def test_boundaries_are_ordered_and_source_covered():
    for manifest in json.loads(REGISTRY.read_text()):
        folder = REGISTRY.parent / f"cutoff_{manifest['cutoff_year']}"
        calendar = pd.read_csv(folder / "calendar_definitions.csv")
        gdd = pd.read_csv(folder / "gdd_definitions.csv")
        coverage = pd.read_csv(folder / "source_coverage.csv")
        assert (calendar["source_complete_years"] >= 3).all()
        assert (
            pd.to_datetime(
                "2001-"
                + calendar["boundary1_month"].astype(str)
                + "-"
                + calendar["boundary1_day"].astype(str)
            )
            < pd.to_datetime(
                "2001-"
                + calendar["boundary2_month"].astype(str)
                + "-"
                + calendar["boundary2_day"].astype(str)
            )
        ).all()
        assert (gdd["boundary1_gdd"] > 0).all()
        assert (gdd["boundary2_gdd"] > gdd["boundary1_gdd"]).all()
        assert set(coverage["calendar_status"]) == {"PASS"}
