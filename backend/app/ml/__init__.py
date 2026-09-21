"""ML package for the next-gen 20-day NSE forecasting system.

This package holds the modular machine-learning layer:
- versions.py             : feature/model/prediction versioning + snapshots
- features.py             : (Task 3) technical feature engine
- market_features.py      : (Task 4) market-relative + volatility/risk features
- alpha.py / beta.py      : (Task 5) rank-normalised alpha factors + exposure
- events.py               : (Task 6) PIT event layer (no-lookahead)
- fundamental_features.py : (Task 6) PIT fundamentals
- sentiment_features.py   : (Task 6) PIT news-sentiment features
- macro_features.py       : (Task 6) slow-moving macro/global features
- regime.py               : (Task 7) market regime (trend/vol/risk/breadth)
- dataset.py              : (Task 7) 20d forward target + chronological splits
- options_df.py           : (Task 7) F&O OI/PCR/IV/Greeks (graceful disable)
- models.py               : (Task 8) voting baseline + logistic/ridge/RF/XGBoost
- backtest.py             : (Task 8) walk-forward, purged CV + embargo, costs, metrics
- ablation.py             : (Task 8) leave-one-group-out feature ablation
- ensemble.py             : (Task 9) weighted P(up) averaging across models
- calibration.py          : (Task 9) Platt (IRLS) / isotonic (PAVA) calibration + ECE
- signal.py               : (Task 9) BUY/HOLD/SELL policy from validated thresholds
- explain.py              : (Task 9) permutation importances + local +/- factors
- monitor.py              : (Task 9) freshness, PSI drifts, Brier drift, degraded mode
- pipeline.py             : (Task 10) forecast orchestrator — feature assembly,
                            walk-forward OOF ensemble + calibration, latest signal,
                            explanation, monitor status, versioned snapshot
- scorecard.py            : (W9) forward (in-time) prediction scorecard — append-only
                            live-prediction log + realized-outcome scoring against
                            the T->T+horizon close + accuracy/Brier/edge-over-baseline
                            report (the missing production caller of
                            `versions.record_outcome()`).
- calibration_monitor.py  : (W10) live calibration-drift monitor — persists the
                            holdout-calibrated Brier/ECE reference at forecast time
                            and compares it against the W9 forward scorecard;
                            alarm only with a sufficient sample, UNKNOWN otherwise.
- conformal.py            : (W11) split-conformal uncertainty band around the
                            calibrated P(up) — |q-y| nonconformity scores on the
                            older 80% calibration-fit slice, conservative higher
                            quantile, holdout empirical coverage as a sanity
                            check; insufficient-sample/NaN handling is honest,
                            never fabricates an interval.
- benchmark.py            : (Task 11) multi-stock V1 benchmark & failure analysis —
                             measurement layer; re-runs the production pipeline over a
                             configurable NSE universe with per-model, ablation,
                             baseline, regime and calibration diagnostics + JSON/text
                             reports. Measurement only — never changes production.
- v2_v1_fair_compare.py   : (Task 17) rigorous frozen-V1 vs pooled model comparison
                             on shared historical snapshot; research-only.
- ...                     : (later tasks) acceptance + docs.

The rule-based voting engine in services/signal_service.py stays as a
baseline/fallback benchmark and is NOT replaced by this package.
"""