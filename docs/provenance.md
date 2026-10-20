# Source and changes

Heart Attack Classification was refactored from the author-supplied Python notebook into a reusable module, a clean demonstration notebook, and focused validation checks. Original files are preserved locally. Embedded notebook outputs were cleared rather than presented as newly reproduced results. Student identifiers, personal phone numbers, source-machine paths, and names from illustrative records are excluded.

New commits describe the actual work done now; original analysis dates are not fabricated.

## Corrections and interpretation

Uses the supplied 10000-row sample directly rather than resampling an unavailable full file. Excludes unknown outcome labels instead of imputing them. Splits before imputation, scaling, encoding, resampling, and clustering. Removes undefined data variable and chained fillna updates. Uses SMOTENC for mixed categorical/numeric features rather than ordinary SMOTE over arbitrary label-encoded categories. Clustering excludes the target and is descriptive only. Reports a majority baseline, class prevalence, ROC-AUC, average precision, precision/recall/F1, confusion matrices, and training-only CV. Historical notebook outputs are not claimed to reproduce under this corrected pipeline. Statsmodels ordinal-code coefficient p-values are not repeated because their category-distance assumptions were unsupported. Self-reported history is not prospective cardiovascular risk; no clinical decisions or causal claims are supported. Source was supplied as a group notebook; original collaborator names and individual contributions were not supplied, so sole authorship is not claimed.

Full-data IQR capping was removed rather than carrying over data-dependent bounds learned before the split. Missing predictor values are imputed within training folds. The corrected pipeline reports new measurements; it does not reproduce the original label-encoded and full-data-preprocessed experiment.
