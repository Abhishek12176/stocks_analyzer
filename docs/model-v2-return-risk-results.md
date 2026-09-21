# Task 20 — Expected Return + Risk Forecasting Research

**Status:** research-only; production V1 remains frozen  
**Module:** `backend/app/ml/v2_return_risk_forecast.py`  
**Artifact:** `backend/ml_v2_return_risk/task20_result.json`

Task 20 investigates whether forecasting expected return magnitude and risk
contains genuine out-of-sample predictive information beyond the existing
weak V1 directional probability. This is a pure research module — no V1
features, targets, models, ensemble, calibration, thresholds, backtest,
monitor, forecast API, or frontend are modified.

## Research Design

### Two Approaches Compared
1. **Two-stage:** P(up) from logistic × E(return | up) from Ridge/RF/XGB
2. **Direct regression:** Ridge/RF/XGB predicting raw 20d return

### Feature Configurations
- **Config A:** Frozen V1-compatible features only (baseline)
- **Config B:** V1 features + Task-14 relative/cross-sectional features

### Regressors Investigated
- Ridge regression (L2-regularised linear)
- Random Forest regression (200 trees, depth 6)
- XGBoost regression (200 trees, depth 3)

### Risk Outputs
- Conditional volatility (residual RMSE)
- Residual MAD (median absolute deviation)
- Quantile prediction intervals (5th / 95th percentile)

### Evaluation Protocol
- Chronological walk-forward: test_size=60, step=60, min_train=260, embargo=20
- Seeds: 17, 31, 53
- Final holdout: newest 20% of dates (untouched during fold training)
- Training-only median imputation (sklearn-free)
- Null control: shuffled-target regression

### Entropy Interaction Diagnostic
One predeclared diagnostic: correlation between predictive entropy and
absolute regression error. If high entropy → high error, it suggests
entropy-based abstention could improve return forecasts.

## Classification

* `PROMISING`
* `REJECTED`
* `EVIDENCE STILL INSUFFICIENT`

## Result (10-Sep-2026, shared Task 17 snapshot, 20 symbols)

**Classification: `REJECTED` — expected-return regression shows NO genuine
out-of-sample predictive signal on this snapshot.**

Real-data run (seeds 17/31/53, most recent 8 walk-forward folds, holdout =
newest 20% of dates):

| Regressor | Holdout RMSE | MAE | corr(⊞,⊟) | bias | n |
|---|---|---|---|---|---|
| Ridge        | 0.0930 | 0.0744 | 0.125 | +0.024 | 13,674 |
| Random Forest| 0.0817 | 0.0652 | 0.040 | +0.007 | 13,674 |
| XGBoost      | 0.0930 | 0.0759 | 0.026 | +0.005 | 13,674 |

* Best regression RMSE (Random Forest, 0.0817) is **at the shuffled-target
  null-control level (null RMSE 0.0815)** → the null gate rejects.
* Direction correlation is small (Ridge 0.125, RF 0.040, XGB 0.026) and
  dominated by Ridge's momentum-biased shift, not reliable skill.
* Two-stage (`P(up) × E(return | up)`) lowers *absolute* MAE (0.0586) purely
  via shrinkage toward a small constant, not information; it cannot rescue
  RMSE/signal that is already at null level.
* Entropy interaction: corr(entropy, |return error|) = −0.12 → **no_signal**
  (uncertainty is not a usable abstention signal for return forecasting).
* Bootstrap CI on pooled holdout RMSE: 95% CI [0.0888, 0.0900] — prediction
  error is indistinguishable from the natural 20-day return dispersion.
* Config B had no additional relative columns in the snapshot (enriched ==
  baseline, 88 cols), consistent with Task 17; recorded as redundant.

**Conclusion:** Root mean error of expected-return forecasts is statistically
indistinguishable from predicting that the around-horizon return is zero (the
null/shrinkage benchmark). Expected-return magnitude + risk forecasts do NOT
carry genuine out-of-sample information on the frozen Task 17 snapshot, so they
are rejected for production and remain research-only. **Production V1 stays
frozen.**

## Commands

```bash
cd backend
python -m pytest tests/test_v2_return_risk_forecast.py -q   # 35 tests
python -m app.ml.v2_return_risk_forecast                     # real-data run
python -m app.ml.v2_return_risk_forecast --snapshot path/to/task17_shared_snapshot.csv
```
