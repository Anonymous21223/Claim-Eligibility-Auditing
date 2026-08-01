# Extension Factorial V6.1 Artifact Audit

- Status: **PASS**
- Decision: **NO_RERUN**
- Model fitting performed by this audit: **No**

| Check | Status |
|---|---|
| `authored_v6_frozen` | PASS |
| `frozen_kse_unchanged` | PASS |
| `inner_jobs_unique_complete` | PASS |
| `outer_jobs_unique_complete` | PASS |
| `factorial_matrix_complete` | PASS |
| `configured_seeds_complete` | PASS |
| `train_only_execution_fields` | PASS |
| `stage_manifest_provenance` | PASS |
| `outer_prediction_row_integrity` | PASS |
| `seed_mean_predictions_reproduced` | PASS |
| `all_outer_fold_metrics_reproduced` | PASS |
| `all_pooled_metrics_reproduced` | PASS |
| `all_core_tables_and_selection_reproduced` | PASS |
| `winner_modules_reproduced` | PASS |
| `candidate_wide_counts` | PASS |
| `winner_lock_and_external_order` | PASS |
| `external_suite_completion` | PASS |
| `external_tables_reproduced` | PASS |

## Candidate-Wide Counts

- R1: 38/240
- Module A: 122/240
- Module B: 46/80 full pipelines
- A+B: 46/80 full pipelines
- Module E: 0/46 evaluated candidates

## Lock-Order Evidence

The final filesystem modification time is not used as proof because the lock was rewritten during finalization. The stored command order places core finalization before both external suites, the external runner requires an existing LOCKED file, the lock hashes the crop-only leaderboard and protocol, and the selection source does not read EXT-SEMI or EXT-PJM artifacts.
