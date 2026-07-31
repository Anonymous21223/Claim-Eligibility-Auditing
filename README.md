# Claim-eligibility audit: anonymous supplementary artifact

This package supports the KSE 2026 v5.9 model-first revision. It contains
fixed inputs, derived records, source code, generated tables, and generated
figures. It intentionally excludes the authored manuscript, author names,
affiliations, email addresses, local paths, credentials, and repository history.

## Environment

- Python 3.11 or later
- Packages pinned in `requirements-kse.txt`
- A writable Matplotlib cache

```bash
export MPLCONFIGDIR=/tmp/kse-matplotlib
python -m pip install -r requirements-kse.txt
```

## Rebuild the V5.9 display assets

Run from the package root:

```bash
python scripts/build_kse_paper_assets.py
python scripts/build_claim_eligibility_workflow_kse.py
python scripts/build_kse_study_map.py
python scripts/build_kse_figure3.py
```

These commands read locked inputs and regenerate tables/figures only. They do
not fit or select a model. Outputs appear under:

- `paper_versions/v5_9_kse_model_first_6pages/source/generated/`
- `paper_versions/v5_9_kse_model_first_6pages/source/figures/`

The map script reads the bundled generalized contiguous-state GeoJSON from the
U.S. Census Bureau TIGERweb State layer, January 1, 2025 vintage. Source URL,
access date, input hash, package versions, state list, and output hashes are in
`generated/study_map_provenance_kse.json`.

## Existing sensitivity artifacts

The MDE and fixed LightGBM folders are retained as already-computed post-hoc
sensitivity evidence:

- `artifacts/experiments/kse-v5-8/mde/`
- `artifacts/experiments/kse-v5-8/lightgbm/`

The V5.9 revision did not rerun an experiment or use locked-test performance to
replace the validation-selected primary model. Legacy reproduction scripts are
included for provenance, but running them is not required to rebuild V5.9
display assets.

## Integrity

Verify the allowlisted package files with:

```bash
sha256sum -c SHA256SUMS.txt
```

`MANIFEST.json` records every allowlisted file size and SHA-256 digest.
`ANONYMITY_AUDIT.json` records the package identity scan.
