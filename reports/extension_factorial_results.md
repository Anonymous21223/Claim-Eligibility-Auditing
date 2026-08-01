# Extension Factorial V1 Results

## Scope

This is a retrospective extension study. It does not revise the frozen KSE manuscript and it does not support causal weather claims.

## Completion

- Design: 16 arms, 80 arm-model cells, 240 feature-family pipelines.
- Inner search: 34,560/34,560 jobs complete across six outer folds and three rolling inner folds.
- Outer refit: 4,896/4,896 seeded fits complete.
- All target trends, preprocessing, hierarchy components, and source-derived stage definitions were fitted or selected without outer-test outcomes.

## Locked Crop Result

- Winner: `C310 / catboost / full`.
- Representation: STAGE-HYBRID; hierarchy: HIER-RESIDUAL; grid: COMPACT.
- Pooled outer-fold RMSE: 0.460728 t ha^-1.
- Pooled outer-fold R2: 0.110202; positive R2 in 4/6 folds.
- Claim tier: `A+B`.
- Module A upper 95% CI: -0.012891; Module B upper 95% CI: -0.012891.
- Module E: `FAIL`. Event-recovery interpretation is not permitted.

The winner was fixed before EXT-SEMI and EXT-PJM. External-suite results were not used for crop selection.

## EXT-SEMI

The suite contains 28,800 scored rows from 9,600 model/scenario base fits: four locked candidates, five signal levels, four measurement-error levels, four spatial-mismatch levels, three event-prevalence settings, and 30 seeds. Ground-truth validity labels were used only for scoring and never as model or policy inputs.

## EXT-PJM

The degradation suite contains 600 seed-level runs and 20 curve points across permutation mix, noise, dropout, and lag substitution. All lambda-zero inputs are identical after train-median imputation of the two source forecast missing values. PJM remains calibration evidence only and did not enter crop ranking.

## Interpretation Boundary

The real crop winner passes stable positive performance and Modules A+B, allowing an overall and weather-specific predictive-reliance claim in this retrospective population. Module E does not pass; observed-event recovery and causal weather effects remain unsupported.
