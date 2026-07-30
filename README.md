# Claim-Eligibility Auditing for Post-hoc Explanations

This is an anonymous artifact repository for double-blind review.  It contains
source code, frozen artifacts, data manifests, generated paper assets, and the
anonymous manuscript source needed to inspect the claim-eligibility audit.

## Main Contents

- `src/`: Python package code.
- `scripts/`: experiment, audit, crosscheck, and figure-generation scripts.
- `configs/`: locked experiment and claim-eligibility configuration.
- `data/`: curated non-legacy data needed by the released pipeline.
- `artifacts/`: frozen non-reproduction artifacts used by the manuscript.
- `paper/generated/`: generated tables and figures.
- `paper_versions/v5_7_web_prototype_figure/source/`: anonymous manuscript source.
- `paper/final/ictai2026_claim_eligibility_audit_v5_7_web_prototype_figure.pdf`: anonymous PDF.
- `tests/`: repository contract tests.

## Reproduction

Install Python dependencies from `requirements.txt`, then inspect or run the
entry points in `scripts/`.  The manuscript can be rebuilt from the v5.7 source
with a TeX installation:

```bash
cd paper_versions/v5_7_web_prototype_figure/source
pdflatex -interaction=nonstopmode -halt-on-error fidelity_gated_xai_method_benchmark_v3.tex
bibtex fidelity_gated_xai_method_benchmark_v3
pdflatex -interaction=nonstopmode -halt-on-error fidelity_gated_xai_method_benchmark_v3.tex
pdflatex -interaction=nonstopmode -halt-on-error fidelity_gated_xai_method_benchmark_v3.tex
```

The package intentionally omits author-identifying files and local handoff
materials.  Upload this folder from an anonymous account if an online repository
is required by the venue.
