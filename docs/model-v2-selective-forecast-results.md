# Task 18 — Selective forecasting and uncertainty-aware decisions

**Status:** research-only; production V1 remains frozen  
**Module:** `backend/app/ml/v2_selective_forecast.py`  
**Artifacts:** `backend/ml_v2_selective_forecast/task18_selective_forecast_result.json`
and `task18_result.json`

## Contract

Task 18 evaluates abstention/acceptance rules without changing the V1 forecast,
signal, calibration, thresholds, backtest, API, or frontend. It uses the
persisted Task 17 snapshot (20 symbols, 24,798 rows; fingerprint
`3359b97e9499320509199041ce3dcda4c52bd65086919b8010056af350ef1581`) and the
frozen V1 components:

`voting`, `logistic`, `ridge`, `rf`, and `xgboost`.

For every chronological fold and seed, `fair.run_v1_reference` (or its exact
V1 implementation) produces out-of-sample component probabilities and the
existing V1 ensemble. Task 18 applies fixed selectivity rules to those
probabilities only. It does **not** fit pooled logistic, random-forest, or any
other V2 candidate model.

The four expanding folds use a 20-day purge/embargo. The newest 20% is an
untouched final holdout (`2025-09-15`–`2026-09-09`) evaluated once. Conformal
calibration uses earlier out-of-sample V1 folds only; no final-holdout policy
tuning occurs.

## Fixed candidates

- Always-accept null.
- Probability margins: 0.05, 0.10, 0.15.
- **Candidate A:** entropy ≤ 0.65 (`candidate_a_entropy`, alias `entropy_065`).
- **Candidate B:** V1 component disagreement ≤ 0.05
  (`candidate_b_disagreement`).
- Additional component-disagreement control ≤ 0.10.
- Split-conformal 90% singleton prediction sets.

Entropy is binary predictive entropy of the frozen V1 ensemble probability.
Disagreement is the standard deviation of the available frozen V1 component
probabilities, never probabilities from a newly trained pooled model.

## Full real-data run (seeds 17, 31, 53)

The final holdout has 136,740 policy/seed records. Always-accept accuracy is
0.4868 and mean signed 20-day return is 0.000564.

| Policy | Coverage | Accuracy | Selective risk | Mean signed return |
|---|---:|---:|---:|---:|
| Candidate A — entropy ≤ 0.65 | 0.4566 | 0.5277 | 0.4723 | 0.004912 |
| Candidate B — disagreement ≤ 0.05 | 0.0148 | 0.6337 | 0.3663 | 0.016921 |
| disagreement ≤ 0.10 | 0.1060 | 0.5135 | 0.4865 | 0.001679 |
| margin ≥ 0.05 | 0.8276 | 0.4889 | 0.5111 | 0.001188 |
| margin ≥ 0.10 | 0.6562 | 0.5062 | 0.4938 | 0.003155 |
| margin ≥ 0.15 | 0.4364 | 0.5281 | 0.4719 | 0.005187 |
| conformal 90% | 0.1311 | 0.5206 | 0.4794 | −0.000730 |

Candidate A and the margin-0.15 rule pass the predeclared research gate, but
Candidate B does not meet the minimum coverage requirement. This does **not**
authorize a production change: the overall result remains research-only and
V1 stays frozen (`production_v1_unchanged: true`).

The JSON artifact contains per-symbol, per-fold, per-seed, regime, null-control,
candidate, and protocol diagnostics, including the V1 component list and source.

## Validation

```text
cd backend
python -m pytest tests/test_v2_selective_forecast.py tests/test_v2_v1_fair_compare.py -q
python -m app.ml.v2_selective_forecast \
  --snapshot ml_v2_fair_compare/task17_shared_snapshot.csv
```

The tests are synthetic/offline and cover fixed policy immutability, entropy,
component disagreement, conformal abstention, holdout isolation, diagnostics,
and JSON serializability.
