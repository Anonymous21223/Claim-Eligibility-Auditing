# KSE 2026 v5.8 Clean Reproduction

Status: PASS

The five documented commands were run from a clean v5.8 artifact tree in a
temporary Python 3.12 environment using the versions pinned in
`requirements-kse.txt`. No locked input, test split, ground-truth label, model
selection rule, or bootstrap count was changed.

## Results

- Module A MDE used 1,000 outer and 1,000 inner year-block replicates.
- Absolute improvement at the MDE: `0.019460656580 t/ha`.
- Estimated power at the MDE: `0.822`.
- LightGBM used 333 locked rows and selected the `full` feature family on the
  validation period.
- LightGBM Module A did not pass; Module B also did not pass.
- Paper tables/macros, the workflow, and Figure 2 were generated successfully.
- Seventeen declared outputs were generated twice; all 17 SHA-256 values were
  identical across the two runs.

The MDE remains a retrospective power sensitivity. The LightGBM run remains a
post-hoc model-family sensitivity and does not replace the primary
validation-selected model.
