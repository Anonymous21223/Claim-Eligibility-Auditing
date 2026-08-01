# Extension Factorial V1 Work Log

- Read and converted the DOCX plan into `configs/extension_factorial_v1.yaml` before model fitting.
- Added a frozen USDA QuickStats crop-progress snapshot and documented source provenance.
- Amended sparse early-cutoff source fallback before any model metric was produced; no yield outcome informed the amendment.
- Built 12 cutoff-specific CAL/GDD stage-feature manifests and passed stage integrity tests.
- Completed the locked 1,728-job scheduling benchmark; reduced concurrency only for RAM stability, without changing any arm/config/seed.
- Completed all 34,560 inner and 4,896 outer jobs with checkpoint/resume.
- Created the leaderboard only after completion reached 100%; locked C310/CatBoost/Full before external suites.
- Ran EXT-SEMI and EXT-PJM after winner lock. Neither suite was used for model selection.
- Corrected the PJM lambda-zero missing-value invariant by applying one train-median imputation before every degradation mechanism; crop artifacts were unchanged.
- The frozen KSE paper source and PDF were not edited by this extension run.
