from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.extension_factorial import build_inner_jobs  # noqa: E402


def test_locked_factorial_counts_and_no_locked_test_selection():
    protocol = json.loads(
        (ROOT / "configs" / "extension_factorial_v1.yaml").read_text()
    )
    jobs = build_inner_jobs(protocol)
    assert jobs["arm_id"].nunique() == 16
    assert jobs[["arm_id", "model_id"]].drop_duplicates().shape[0] == 80
    assert (
        jobs[["arm_id", "model_id", "feature_family"]]
        .drop_duplicates()
        .shape[0]
        == 240
    )
    assert len(jobs) == 34560
    assert jobs["job_id"].nunique() == 34560
    assert (jobs["evaluation_end"] <= jobs["train_end"] + 3).all()


def test_each_grid_has_exact_locked_config_count():
    protocol = json.loads(
        (ROOT / "configs" / "extension_factorial_v1.yaml").read_text()
    )
    for model in protocol["models"]:
        assert len(protocol["model_configs"][model]["COMPACT"]) == 4
        assert len(protocol["model_configs"][model]["EXPANDED"]) == 12
