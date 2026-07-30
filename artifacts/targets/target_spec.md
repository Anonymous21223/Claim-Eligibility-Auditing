# Target and Feature Contract

## Intended Use

The V3 state-level experiment is a post-season scientific audit. It predicts the
raw train-only detrended yield residual after the full crop-season weather window
is available. It is not a pre-harvest forecast and does not claim causal
attribution.

## Target

For crop-state series `(c,s)` and year `t`, fit a linear trend only on training
years:

```text
y_hat_{c,s,t} = a_{c,s}^{train} + b_{c,s}^{train} t
r_{c,s,t} = y_{c,s,t} - y_hat_{c,s,t}
z_{c,s,t} = r_{c,s,t} / max(sigma_{c,s,train}, epsilon)
Event_{c,s,t}(tau) = 1[z_{c,s,t} < tau]
```

- Prediction target: `trend_residual_t_ha` (`r`), not the event indicator.
- Event threshold: `z<-1` primary; `z<-1.5` and `z<-2` sensitivity.
- Scale: training residual standard deviation with `ddof=1`; current training
  scales are audited in `artifacts/targets/train_scale_diagnostics.csv`.
- Minimum history: three prior training rows, a technical minimum for trend/scale
  fitting rather than a stable-variance guarantee.
- Evaluation years are never used to fit trend slope/intercept or residual scale.

## Feature Families

- Metadata-only: `lat`, `lon`, `region`, `crop`.
- Weather-only: 35 full-season weather features.
- Full: metadata plus weather.

Feature availability and window details are machine-readable in
`artifacts/data/feature_availability.csv`. Forbidden target-derived columns are
listed in `artifacts/data/feature_matrix_schema.json` and audited in
`artifacts/audit_records/target_feature_overlap.csv`.

## Current Frame

- Rows: 1257
- Years: 1990-2025
- Crops: Barley, Canola, Oats, Wheat
