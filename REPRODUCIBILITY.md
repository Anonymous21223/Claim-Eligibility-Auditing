# Reproducibility Guide

## Environment

Use Python 3.12 with `requirements-lock.txt`. The manuscript build additionally
requires `pdflatex`, `bibtex`, `pdfinfo`, `pdffonts`, and `pdftotext`.

## Full Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python scripts/reproduce_all.py --clean
```

The command reconstructs weather features from the bundled inputs, performs
validation-only selection, evaluates the locked 2016--2025 test, regenerates
tables/figures/PDF, verifies canonical evidence hashes, scans the anonymous ZIP,
and writes upload manifests.

## Expected Headline Results

- Selected model: ExtraTrees, `min_samples_leaf=1`, Weather only.
- Validation: 2012--2015 only, fixed seeds `7, 17, 27, 37, 47`.
- Locked rows: 333. Selected-model RMSE: 0.669 t ha^-1.
- Gate A paired delta RMSE: -0.005 [-0.019, 0.009], FAIL.
- Primary Gate B1: Weather only vs. Metadata only, FAIL.
- Sensitivity Gate B2: Full vs. Metadata only, FAIL; it cannot re-select a model.

The reproducibility wrapper stores stdout, stderr, and a machine-readable
summary under `reports/`.
