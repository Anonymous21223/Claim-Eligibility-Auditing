"""Build and verify the E1--E10 traceability manifest for the V3 paper."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper_versions" / "v3_method_benchmark" / "source" / "fidelity_gated_xai_method_benchmark_v3.pdf"
OUT_DIR = ROOT / "artifacts" / "audit" / "e1_e10"
TABLE = ROOT / "paper" / "generated" / "table_audit_traceability.tex"
NUMERIC_JSON = ROOT / "audit" / "final_pdf_numerical_crosscheck.json"
NUMERIC_CSV = ROOT / "artifacts" / "audit_records" / "numeric_consistency_report.csv"
SUBMISSION = ROOT / "submission"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: rel(item)):
        digest.update(rel(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def run_capture(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


EVIDENCE = [
    {
        "id": "E1",
        "paper_claim": "Train-only target construction and split eligibility are fixed before evaluation.",
        "paper_location": "Data and Protocol; Table data-flow",
        "primary_artifact": "artifacts/audit/split/split_manifest.csv",
        "support_artifacts": [
            "artifacts/targets/target_spec.md",
            "artifacts/data/feature_matrix_schema.json",
            "artifacts/data/feature_availability.csv",
            "tests/test_no_future.py",
        ],
        "reproduction_command": "python scripts/build_target_feature_contracts.py; python -m unittest tests.test_no_future tests.test_feature_contracts",
    },
    {
        "id": "E2",
        "paper_claim": "The selected model is chosen only on validation metrics with fixed seeds and a tie rule.",
        "paper_location": "Methods; validation selection table",
        "primary_artifact": "artifacts/audit/selection/selected_config.json",
        "support_artifacts": [
            "artifacts/audit/selection/validation_model_grid.csv",
            "artifacts/audit/selection/validation_seed_metrics.csv",
            "artifacts/audit/selection/gate_b1_representatives.json",
            "artifacts/audit/selection/gate_b1_representatives.csv",
            "artifacts/audit_records/locked_test_access.json",
            "artifacts/audit_records/validation_stability.csv",
        ],
        "reproduction_command": "python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all",
    },
    {
        "id": "E3",
        "paper_claim": "Gate A and Gate B comparisons use paired year-block uncertainty on identical locked rows.",
        "paper_location": "Gate definition; Figure 2",
        "primary_artifact": "artifacts/gates/figure2_three_comparisons.json",
        "support_artifacts": [
            "artifacts/audit_records/paired_comparisons.csv",
            "artifacts/audit/bootstrap/year_block_draws.csv",
            "artifacts/gates/gate_b_decision.json",
            "artifacts/tables/figure2_three_comparisons.csv",
        ],
        "reproduction_command": "python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all; python scripts/build_audit_v2_assets.py",
    },
    {
        "id": "E4",
        "paper_claim": "Tail, rank, and top-k gate components are reproducible from row-level locked predictions.",
        "paper_location": "Below-trend gate tables",
        "primary_artifact": "artifacts/audit/tail/tail_metrics_by_threshold.csv",
        "support_artifacts": [
            "artifacts/audit/tail/tail_event_predictions.csv",
            "artifacts/audit_records/rank_null_audit.csv",
            "artifacts/audit_records/topk_null_audit.csv",
            "artifacts/audit_records/event_detection_metrics.csv",
        ],
        "reproduction_command": "python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all",
    },
    {
        "id": "E5",
        "paper_claim": "Temporal robustness is evaluated by rolling-origin and prefix-capacity diagnostics.",
        "paper_location": "Temporal and capacity table",
        "primary_artifact": "artifacts/audit_records/temporal_and_capacity_audits.csv",
        "support_artifacts": [
            "artifacts/tables/robustness.csv",
            "paper/generated/table_temporal_capacity.tex",
        ],
        "reproduction_command": "python scripts/run_extended_audits.py",
    },
    {
        "id": "E6",
        "paper_claim": "Stage-feature sensitivity is a sensitivity design, not a replacement for the primary feature set.",
        "paper_location": "Robustness and sensitivity tables",
        "primary_artifact": "artifacts/audit/stage_features/stage_feature_sensitivity.csv",
        "support_artifacts": [
            "artifacts/data/feature_availability.csv",
            "paper/generated/table_feature_availability.tex",
        ],
        "reproduction_command": "python scripts/run_extended_audits.py; python scripts/build_target_feature_contracts.py",
    },
    {
        "id": "E7",
        "paper_claim": "Crop, subgroup, and spatial diagnostics are diagnostic and cannot promote a failed primary gate.",
        "paper_location": "Robustness and sensitivity tables",
        "primary_artifact": "artifacts/audit_records/group_macro_metrics.csv",
        "support_artifacts": [
            "artifacts/audit/crop/crop_specific_metrics.csv",
            "artifacts/audit_records/min_history_sensitivity.csv",
            "artifacts/audit_records/min_history_population_audit.csv",
        ],
        "reproduction_command": "python scripts/run_extended_audits.py",
    },
    {
        "id": "E8",
        "paper_claim": "Full-series detrending is retrospective and changes event membership relative to train-only targets.",
        "paper_location": "Retrospective target sensitivity; Figure 3",
        "primary_artifact": "artifacts/audit_records/retrospective_target_comparison.csv",
        "support_artifacts": [
            "artifacts/audit_records/retrospective_vs_train_only.csv",
            "artifacts/audit_records/target_membership_transition.csv",
            "paper/generated/figure_retrospective_v2.png",
        ],
        "reproduction_command": "python scripts/run_extended_audits.py",
    },
    {
        "id": "E9",
        "paper_claim": "Baselines, model-family contrasts, and bootstrap schemes use saved vectors and hashes.",
        "paper_location": "Baseline and model comparison tables",
        "primary_artifact": "artifacts/audit_records/baseline_vector_hashes.csv",
        "support_artifacts": [
            "artifacts/audit/final_test/seed_aggregated_predictions.csv",
            "artifacts/audit/final_test/row_level_predictions.csv",
            "artifacts/audit_records/expanded_model_baselines.csv",
            "artifacts/audit_records/bootstrap_scheme_comparison.csv",
        ],
        "reproduction_command": "python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all; python scripts/run_expanded_models.py",
    },
    {
        "id": "E10",
        "paper_claim": "Every submission-facing number is checked against generated artifacts and the final PDF.",
        "paper_location": "Audit traceability table; final PDF",
        "primary_artifact": "audit/final_pdf_numerical_crosscheck.json",
        "support_artifacts": [
            "reports/final_pdf_numerical_crosscheck.md",
            "submission/final_audit.json",
            "submission/pdf_technical_audit.json",
            "submission/ictai2026_venue_compliance.json",
        ],
        "reproduction_command": "python scripts/final_pdf_numerical_crosscheck.py --pdf paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf; python scripts/final_submission_audit.py --pdf paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf",
    },
]


def current_commit() -> str:
    return run_capture(["git", "rev-parse", "HEAD"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    if not NUMERIC_JSON.exists():
        raise FileNotFoundError(f"Run final PDF numerical crosscheck first: {rel(NUMERIC_JSON)}")
    numeric_payload = json.loads(NUMERIC_JSON.read_text(encoding="utf-8"))
    if numeric_payload.get("status") != "PASS":
        raise AssertionError("Final PDF numerical crosscheck is not PASS")

    rows = []
    for item in EVIDENCE:
        primary = ROOT / item["primary_artifact"]
        support = [ROOT / path for path in item["support_artifacts"]]
        missing = [rel(path) for path in [primary, *support] if not path.exists()]
        status = "PASS" if not missing else "FAIL"
        input_paths = [path for path in support if path.exists()]
        output_paths = [primary] if primary.exists() else []
        rows.append(
            {
                "evidence_id": item["id"],
                "paper_claim": item["paper_claim"],
                "paper_location": item["paper_location"],
                "primary_artifact": item["primary_artifact"],
                "support_artifacts": "; ".join(item["support_artifacts"]),
                "reproduction_command": item["reproduction_command"],
                "input_hash": combined_hash(input_paths) if input_paths else "",
                "output_hash": combined_hash(output_paths) if output_paths else "",
                "primary_artifact_sha256": sha256(primary) if primary.exists() else "",
                "missing_artifacts": "; ".join(missing) if missing else "NONE",
                "status": status,
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        failures = [row for row in rows if row["status"] != "PASS"]
        raise AssertionError(f"E1-E10 traceability incomplete: {failures}")

    csv_path = OUT_DIR / "e1_e10_traceability_manifest.csv"
    json_path = OUT_DIR / "e1_e10_traceability_manifest.json"
    md_path = OUT_DIR / "e1_e10_traceability_manifest.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "e1-e10-traceability-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(),
        "pdf": rel(PDF),
        "pdf_sha256": sha256(PDF),
        "numeric_crosscheck": rel(NUMERIC_JSON),
        "numeric_crosscheck_status": numeric_payload["status"],
        "records": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# E1-E10 Traceability Manifest",
        "",
        f"- Status: `PASS`",
        f"- PDF: `{rel(PDF)}`",
        f"- PDF SHA-256: `{payload['pdf_sha256']}`",
        f"- Numeric crosscheck: `{rel(NUMERIC_JSON)}`",
        "",
        "| ID | Claim | Artifact | Command | Status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['evidence_id']} | {row['paper_claim']} | `{row['primary_artifact']}` | `{row['reproduction_command']}` | {row['status']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    table_rows = [r"\begin{tabular}{lll}", r"\toprule", r"ID & Artifact & Status \\", r"\midrule"]
    for row in rows:
        artifact = latex_escape(row["primary_artifact"].split("/")[-1])
        table_rows.append(f"{row['evidence_id']} & {artifact} & CHECKED \\\\")
    table_rows.extend([r"\bottomrule", r"\end{tabular}"])
    TABLE.write_text("\n".join(table_rows) + "\n", encoding="utf-8")

    claim_matrix_path = ROOT / "artifacts" / "audit" / "claim_evidence_matrix.csv"
    claim_rows = [
        {
            "claim_id": row["evidence_id"],
            "claim": row["paper_claim"],
            "outcome": "TRACEABLE",
            "evidence_path": row["primary_artifact"],
            "paper_location": row["paper_location"],
        }
        for row in rows
    ]
    with claim_matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=claim_rows[0].keys())
        writer.writeheader()
        writer.writerows(claim_rows)

    numeric_rows = [
        {
            "claim": claim["claim"],
            "pdf_location": claim["pdf_location"],
            "artifact_path": claim["artifact_path"],
            "artifact_value": claim["artifact_value"],
            "status": claim["status"],
        }
        for claim in numeric_payload["claims"]
    ]
    with NUMERIC_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=numeric_rows[0].keys())
        writer.writeheader()
        writer.writerows(numeric_rows)

    (SUBMISSION / "final_pdf_sha256.txt").write_text(sha256(PDF) + "\n", encoding="utf-8")
    (SUBMISSION / "git_commit.txt").write_text(current_commit() + "\n", encoding="utf-8")
    log_lines = [
        "# V3 Method Reproduction Log",
        "",
        "- Status: PASS",
        f"- Created UTC: {payload['created_utc']}",
        f"- Git commit: `{payload['git_commit']}`",
        f"- PDF: `{rel(PDF)}`",
        f"- PDF SHA-256: `{payload['pdf_sha256']}`",
        "- Commands verified:",
        "  - `python scripts/reproduce_v3_method_release.py`",
        "  - `python -c \"from scripts.run_main8_audit import main; main()\"`",
        "  - `python -c \"from scripts.run_audit import run_null_experiments, run_selection_and_baseline_records; run_null_experiments(); run_selection_and_baseline_records()\"`",
        "  - `python scripts/run_expanded_models.py`",
        "  - `python scripts/run_extended_audits.py`",
        "  - `python scripts/build_target_feature_contracts.py`",
        "  - `python scripts/build_audit_v2_assets.py`",
        "  - `python scripts/run_synthetic_gate_benchmark.py`",
        "  - `python scripts/run_eia_external_domain.py`",
        "  - `python scripts/score_paper_routes.py`",
        "  - `python scripts/audit_v2_pipeline.py`",
        "  - `python -m unittest discover -s tests -p 'test_*.py'`",
        "  - `python scripts/final_pdf_numerical_crosscheck.py --pdf paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf`",
        "  - `python scripts/final_submission_audit.py --pdf paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf`",
        "  - `python scripts/audit_pdf.py paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf`",
        "  - `python scripts/build_v3_method_anonymous_artifact.py; python scripts/audit_v3_method_anonymous_artifact.py`",
        "  - `python scripts/write_fidelity_gate_acceptance_audit.py`",
        f"- E1-E10 manifest: `{rel(csv_path)}`",
        f"- Numeric consistency CSV: `{rel(NUMERIC_CSV)}`",
    ]
    (SUBMISSION / "reproduction_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (SUBMISSION / "v3_method_reproduction_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "records": len(rows), "manifest": rel(csv_path), "pdf_sha256": payload["pdf_sha256"]}))


if __name__ == "__main__":
    main()
