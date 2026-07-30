# Claim-eligibility audit: anonymous supplementary artifact

This package supports the KSE 2026 v5.8 claim-eligibility paper. It contains
the fixed inputs, scripts, derived records, and code-generated figures needed
to inspect the reported MDE, LightGBM sensitivity, tables, and diagrams. It
does not contain author names, affiliations, email addresses, repository
history, or a paper manuscript.

## Claim-eligibility framework

The framework checks claim eligibility before post-hoc explanations are read.
Module A tests overall predictive adequacy, and Module B tests the incremental
value of the feature group named in a proposed claim. Module E is evaluated
only when the proposed interpretation is an event-level claim. Module D is not
part of the v5.8 permission path because it lies outside that path and overlaps
informationally with Module A.

## Environment

- Python 3.11 or later
- The packages pinned in `requirements-kse.txt`
- A writable Matplotlib cache, for example:

```bash
export MPLCONFIGDIR=/tmp/kse-matplotlib
python -m pip install -r requirements-kse.txt
```

## Reproduce the v5.8 additions

Run from the package root:

```bash
python scripts/run_kse_mde_analysis.py
python scripts/run_kse_lightgbm_sensitivity.py
python scripts/build_kse_paper_assets.py
python scripts/build_claim_eligibility_workflow_kse.py
python scripts/build_kse_figure2.py
```

The MDE step is a retrospective nested year-block sensitivity analysis; it is
not a new predictive result. The LightGBM step is a fixed, post-hoc
model-family sensitivity; it does not replace the validation-selected primary
model.

Generated results appear under:

- `artifacts/experiments/kse-v5-8/mde/`
- `artifacts/experiments/kse-v5-8/lightgbm/`
- `paper_versions/v5_8_kse_revision/source/generated/`
- `paper_versions/v5_8_kse_revision/source/figures/`

## Integrity

Verify the allowlisted package files with:

```bash
sha256sum -c SHA256SUMS.txt
```

`MANIFEST.json` records each file size and SHA-256 digest. Locked inputs are
read but are not rewritten by the scripts.
