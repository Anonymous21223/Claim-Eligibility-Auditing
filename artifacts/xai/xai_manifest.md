# XAI Provenance Manifest

| Method | Artifact | Output scale | Status |
|---|---|---|---|
| SHAP | `outputs/xai/shap_feature_ranking.csv` | predicted raw residual t/ha | DESCRIPTIVE_ONLY_GATE_FAILED |
| LIME | `outputs/xai/lime_event_explanations.csv` | predicted raw residual t/ha | DESCRIPTIVE_ONLY_GATE_FAILED |
| Group permutation | `outputs/xai/group_permutation_importance.csv` | RMSE increase on predicted raw residual t/ha | DESCRIPTIVE_ONLY_GATE_FAILED |
| Group ablation | `outputs/xai/group_ablation_importance.csv` | RMSE increase on predicted raw residual t/ha | DESCRIPTIVE_ONLY_GATE_FAILED |
| ALE response curves | `outputs/xai/ale_curves.csv` | accumulated local effect in residual t/ha | DESCRIPTIVE_ONLY_GATE_FAILED |
| PJM permutation importance | `artifacts/experiments/external-domain-eia/gated_feature_importance.csv` | predicted daily demand MWh | INTERPRET_PREDICTIVE_RELIANCE_ONLY |
