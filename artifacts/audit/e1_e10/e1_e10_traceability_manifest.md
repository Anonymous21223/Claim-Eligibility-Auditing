# E1-E10 Traceability Manifest

- Status: `PASS`
- PDF: `paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf`
- PDF SHA-256: `e880593988046b3f6112511657a7c6e76fdb62e17d1a1bd5e54f083832df96fe`
- Numeric crosscheck: `audit/final_pdf_numerical_crosscheck.json`

| ID | Claim | Artifact | Command | Status |
|---|---|---|---|---|
| E1 | Train-only target construction and split eligibility are fixed before evaluation. | `artifacts/audit/split/split_manifest.csv` | `python scripts/build_target_feature_contracts.py; python -m unittest tests.test_no_future tests.test_feature_contracts` | PASS |
| E2 | The selected model is chosen only on validation metrics with fixed seeds and a tie rule. | `artifacts/audit/selection/selected_config.json` | `python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all` | PASS |
| E3 | Gate A and Gate B comparisons use paired year-block uncertainty on identical locked rows. | `artifacts/gates/figure2_three_comparisons.json` | `python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all; python scripts/build_audit_v2_assets.py` | PASS |
| E4 | Tail, rank, and top-k gate components are reproducible from row-level locked predictions. | `artifacts/audit/tail/tail_metrics_by_threshold.csv` | `python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all` | PASS |
| E5 | Temporal robustness is evaluated by rolling-origin and prefix-capacity diagnostics. | `artifacts/audit_records/temporal_and_capacity_audits.csv` | `python scripts/run_extended_audits.py` | PASS |
| E6 | Stage-feature sensitivity is a sensitivity design, not a replacement for the primary feature set. | `artifacts/audit/stage_features/stage_feature_sensitivity.csv` | `python scripts/run_extended_audits.py; python scripts/build_target_feature_contracts.py` | PASS |
| E7 | Crop, subgroup, and spatial diagnostics are diagnostic and cannot promote a failed primary gate. | `artifacts/audit_records/group_macro_metrics.csv` | `python scripts/run_extended_audits.py` | PASS |
| E8 | Full-series detrending is retrospective and changes event membership relative to train-only targets. | `artifacts/audit_records/retrospective_target_comparison.csv` | `python scripts/run_extended_audits.py` | PASS |
| E9 | Baselines, model-family contrasts, and bootstrap schemes use saved vectors and hashes. | `artifacts/audit_records/baseline_vector_hashes.csv` | `python scripts/run_audit.py --config configs/fidelity_gate.yaml --stage all; python scripts/run_expanded_models.py` | PASS |
| E10 | Every submission-facing number is checked against generated artifacts and the final PDF. | `audit/final_pdf_numerical_crosscheck.json` | `python scripts/final_pdf_numerical_crosscheck.py --pdf paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf; python scripts/final_submission_audit.py --pdf paper_versions/v3_method_benchmark/source/fidelity_gated_xai_method_benchmark_v3.pdf` | PASS |
