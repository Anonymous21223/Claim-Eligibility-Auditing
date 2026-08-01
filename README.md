# Claim-Eligibility Factorial Audit: Anonymous Artifact

This branch supports the anonymous V6.1 factorial manuscript. It contains the
locked protocol, completion records, row-level outer-fold predictions,
candidate-wide decisions, external calibration outputs, anonymous manuscript
source, and every figure-generation script used by the paper. It contains no
authored manuscript or author identity.

## Verify stored artifacts

```bash
python -m pip install -r requirements-lock.txt
python scripts/audit_extension_v6_submission.py
python -m pytest -q tests/test_extension_factorial_contract.py tests/test_extension_release_contract.py tests/test_extension_stage_features.py
```

The audit does not fit or select a model. It checks all unique job keys, row
hashes, fold assignment, seed aggregation, train-only fields, locked metrics,
module calculations, candidate-wide counts, external completion, and lock
isolation from external suites.

## Regenerate tables and figures

```bash
python scripts/build_extension_paper_assets.py
python scripts/build_extension_coverage_map.py
python scripts/build_extension_v6_1_figure1.py
```

## Build the anonymous manuscript

Run `pdflatex`, `bibtex`, then `pdflatex` twice in
`paper_versions/v6_1_extension_factorial_anonymous/source/`. The final reviewed
PDF is also stored under `paper/final/`.

## Interpretation boundary

The locked crop winner is C310/CatBoost/full. Modules A and B pass; Module E
fails. The artifact supports a weather-specific predictive-reliance statement
for the evaluated retrospective population, not event recovery or causality.
The winner intervals are post-selection developmental evidence.
