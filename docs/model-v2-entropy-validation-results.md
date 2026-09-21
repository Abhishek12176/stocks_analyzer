# Task 19 — Entropy selective-forecast validation

**Status:** research-only; production V1 remains frozen  
**Module:** `backend/app/ml/v2_entropy_validation.py`  
**Artifact:** `backend/ml_v2_selective_forecast/task19_entropy_validation_result.json`

Task 19 validates **exactly** Task 18 Candidate A: binary entropy of the
frozen V1 ensemble probability at `entropy <= 0.65`. It consumes the persisted
Task 18 result and does not refit models, alter features/targets/ensembles, or
tune the final holdout. The report explicitly compares all rows with retained
rows and records retained/abstained counts and coverage, predictive and
calibration metrics, fold/seed/cross-symbol diagnostics, economic returns,
shuffled-target controls, and paired per-symbol bootstrap uncertainty.

Classification is deliberately restricted to the required final labels:

* `PROMISING FOR PRODUCTION INTEGRATION`
* `REJECTED`
* `EVIDENCE STILL INSUFFICIENT`

The entropy candidate does not receive the production-integration label from
this run. Passing the inherited Task 18 coverage/accuracy/risk gate alone is
not sufficient: robust cross-symbol and chronological-fold evidence must also
be demonstrated. The result therefore remains `EVIDENCE STILL INSUFFICIENT`;
V1 behavior remains unchanged.

```text
cd backend
python -m pytest tests/test_v2_entropy_validation.py tests/test_v2_selective_forecast.py -q
python -m app.ml.v2_entropy_validation
```
