# Execution and validation

Executed with Python 3.12 using the versions pinned in requirements.txt. Program run completed successfully. Generated notebook cells passed Python syntax compilation; the notebook wrapper calls the same validated module. Full notebook execution in a separate Jupyter kernel was not performed.

```text
test_missing_labels_are_excluded_and_target_removed (test_analysis.HeartTest.test_missing_labels_are_excluded_and_target_removed) ... ok
test_sampling_uses_training_predictors_only (test_analysis.HeartTest.test_sampling_uses_training_predictors_only) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.016s

OK
```

Input/output data were checked for required fields, finite values, and appropriate synthetic-versus-supplied provenance. Output files contain aggregate metrics or non-personal examples; row-level clinical predictions are not published.
