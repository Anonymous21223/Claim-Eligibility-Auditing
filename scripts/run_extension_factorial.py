"""Execute and checkpoint extension-factorial-v1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.extension_factorial import (  # noqa: E402
    build_inner_jobs,
    build_outer_jobs,
    file_sha256,
    fit_predict_job,
    load_protocol,
    select_inner_configs,
)


OUT = ROOT / "artifacts" / "extensions" / "v1"
CORE = OUT / "core"
MATRIX = OUT / "COMPLETION_MATRIX.csv"


def canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_matrix(inner: pd.DataFrame, outer: pd.DataFrame | None = None) -> None:
    parts = [
        inner
        if "status" in inner.columns
        else inner.assign(status="PENDING")
    ]
    if outer is not None:
        parts.append(
            outer
            if "status" in outer.columns
            else outer.assign(status="PENDING")
        )
    matrix = pd.concat(parts, ignore_index=True)
    keep = [
        "job_id",
        "phase",
        "arm_id",
        "representation",
        "hierarchy",
        "grid",
        "model_id",
        "feature_family",
        "outer_fold",
        "inner_fold",
        "train_end",
        "evaluation_start",
        "evaluation_end",
        "seed",
        "config_id",
        "status",
    ]
    matrix[keep].to_csv(MATRIX, index=False)


def setup_matrix() -> None:
    protocol = load_protocol(str(ROOT))
    CORE.mkdir(parents=True, exist_ok=True)
    jobs = build_inner_jobs(protocol)
    expected = int(protocol["expected_counts"]["inner_fit_calls"])
    if len(jobs) != expected or jobs["job_id"].nunique() != expected:
        raise AssertionError(f"Expected {expected} unique inner jobs, found {len(jobs)}")
    jobs.to_parquet(CORE / "inner_jobs.parquet", index=False)
    identities = jobs[
        ["arm_id", "model_id", "feature_family"]
    ].drop_duplicates()
    if (
        jobs["arm_id"].nunique() != 16
        or jobs[["arm_id", "model_id"]].drop_duplicates().shape[0] != 80
        or len(identities) != 240
    ):
        raise AssertionError("Factorial identity counts do not match protocol")
    write_matrix(jobs)
    print(
        json.dumps(
            {
                "arms": 16,
                "model_cells": 80,
                "pipelines": 240,
                "inner_jobs": len(jobs),
            },
            indent=2,
        )
    )


def execute(
    jobs: pd.DataFrame,
    output: Path,
    workers: int,
    include_predictions: bool = False,
    executor_kind: str = "process",
) -> pd.DataFrame:
    done = pd.DataFrame()
    if output.exists():
        done = pd.read_parquet(output)
    complete = set(done["job_id"]) if len(done) else set()
    pending = jobs[~jobs["job_id"].isin(complete)].copy()
    records = done.to_dict("records")
    prediction_dir = CORE / "outer_prediction_parts"
    if include_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    executor = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
    with executor(max_workers=workers) as pool:
        iterator = pool.map(
            fit_predict_job,
            [str(ROOT)] * len(pending),
            pending.to_dict("records"),
            chunksize=1,
        )
        for index, result in enumerate(iterator, 1):
            prediction = result.pop("predictions", None)
            if prediction is not None:
                prediction.to_parquet(
                    prediction_dir / f"{result['job_id']}.parquet",
                    index=False,
                )
            records.append(result)
            if index % 100 == 0 or index == len(pending):
                pd.DataFrame(records).to_parquet(output, index=False)
                elapsed = time.perf_counter() - started
                print(
                    f"{output.stem}: {len(complete) + index}/{len(jobs)} "
                    f"({elapsed:.1f}s this run)",
                    flush=True,
                )
    return pd.DataFrame(records)


def benchmark(workers: int, executor_kind: str) -> None:
    protocol = load_protocol(str(ROOT))
    jobs = pd.read_parquet(CORE / "inner_jobs.parquet")
    count = int(protocol["expected_counts"]["benchmark_inner_fit_calls"])
    jobs = jobs.assign(
        sample_key=jobs["job_id"].map(lambda value: sha256(value.encode()).hexdigest())
    ).sort_values("sample_key").head(count).drop(columns="sample_key")
    started = time.perf_counter()
    pending_before = count - (
        len(pd.read_parquet(CORE / "benchmark_results.parquet"))
        if (CORE / "benchmark_results.parquet").exists()
        else 0
    )
    results = execute(
        jobs,
        CORE / "benchmark_results.parquet",
        workers,
        executor_kind=executor_kind,
    )
    elapsed = time.perf_counter() - started
    if len(results) != count:
        raise AssertionError("Benchmark did not complete exactly 1,728 jobs")
    report = {
        "status": "PASS",
        "jobs": len(results),
        "workers": workers,
        "jobs_executed_this_attempt": pending_before,
        "elapsed_seconds_this_attempt": elapsed,
        "seconds_per_new_job_this_attempt": (
            elapsed / pending_before if pending_before else None
        ),
        "checkpoint_span_note": "The first benchmark included failed high-memory process attempts; use the per-new-job value from a nonzero stable attempt for scheduling.",
        "grid_changed_after_benchmark": False,
    }
    (CORE / "benchmark_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def run_inner(workers: int, executor_kind: str) -> None:
    protocol = load_protocol(str(ROOT))
    jobs = pd.read_parquet(CORE / "inner_jobs.parquet")
    results = execute(
        jobs,
        CORE / "inner_results.parquet",
        workers,
        executor_kind=executor_kind,
    )
    if len(results) != int(protocol["expected_counts"]["inner_fit_calls"]):
        raise AssertionError("Inner search incomplete")
    if results["future_access"].any():
        raise AssertionError("Future access detected in inner jobs")
    selected = select_inner_configs(jobs, results)
    selected.to_csv(CORE / "selected_inner_configs.csv", index=False)
    outer = build_outer_jobs(protocol, selected)
    expected = int(protocol["expected_counts"]["outer_seeded_fits"])
    if len(outer) != expected or outer["job_id"].nunique() != expected:
        raise AssertionError(f"Expected {expected} outer jobs, found {len(outer)}")
    outer.to_parquet(CORE / "outer_jobs.parquet", index=False)
    write_matrix(jobs.assign(status="PASS"), outer)


def run_outer(workers: int, executor_kind: str) -> None:
    protocol = load_protocol(str(ROOT))
    outer = pd.read_parquet(CORE / "outer_jobs.parquet")
    results = execute(
        outer,
        CORE / "outer_results.parquet",
        workers,
        include_predictions=True,
        executor_kind=executor_kind,
    )
    expected = int(protocol["expected_counts"]["outer_seeded_fits"])
    if len(results) != expected or results["future_access"].any():
        raise AssertionError("Outer evaluation incomplete or leaked")
    parts = sorted((CORE / "outer_prediction_parts").glob("O-*.parquet"))
    if len(parts) != expected:
        raise AssertionError("Outer prediction part count mismatch")
    predictions = pd.concat(
        [pd.read_parquet(path) for path in parts],
        ignore_index=True,
    )
    predictions.to_parquet(CORE / "outer_predictions.parquet", index=False)
    inner = pd.read_parquet(CORE / "inner_jobs.parquet").assign(status="PASS")
    write_matrix(inner, outer.assign(status="PASS"))


def manifest() -> None:
    protocol = load_protocol(str(ROOT))
    required = [
        ROOT / "configs" / "extension_factorial_v1.yaml",
        CORE / "inner_jobs.parquet",
        CORE / "inner_results.parquet",
        CORE / "selected_inner_configs.csv",
        CORE / "outer_jobs.parquet",
        CORE / "outer_results.parquet",
        CORE / "outer_predictions.parquet",
        MATRIX,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Cannot finalize; missing: {missing}")
    matrix = pd.read_csv(MATRIX)
    if not matrix["status"].eq("PASS").all():
        raise AssertionError("Completion matrix is not 100% PASS")
    payload = {
        "protocol_id": protocol["protocol_id"],
        "status": "CORE_COMPLETE",
        "protocol_sha256": file_sha256(required[0]),
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "counts": matrix.groupby("phase").size().to_dict(),
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in required
        },
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    (OUT / "RUN_MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=["matrix", "benchmark", "inner", "outer", "manifest"],
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
    )
    args = parser.parse_args()
    if args.phase == "matrix":
        setup_matrix()
    elif args.phase == "benchmark":
        benchmark(args.workers, args.executor)
    elif args.phase == "inner":
        run_inner(args.workers, args.executor)
    elif args.phase == "outer":
        run_outer(args.workers, args.executor)
    else:
        manifest()


if __name__ == "__main__":
    main()
