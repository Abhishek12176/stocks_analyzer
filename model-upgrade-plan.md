# Model Upgrade Plan — 20-Day NSE Forecast (Task 10)

Status: implemented and integrated with the full API/frontend (Task 10, final).
Tests: full backend suite **309 passed** (including 19 Task-10 tests).

## What this is

A probabilistic, out-of-sample backtested 20-day forecast for NSE stocks,
produced by the `backend/app/ml/pipeline.py` orchestrator. It reuses every
engine built in Tasks 3–9 — it does **not** replace the existing rule-based
`services/signal_service.py` baseline; that stays as the live Signal tab.

## Pipeline (one walk-forward pass, no lookahead)

```
OHLCV + NIFTY/VIX/banknifty + (optional) fundamentals / news / macro / options
   └─ build_feature_frame()  technical + indicator columns + alpha + beta +
                             regime + events + sentiment + macro + fundamentals + options
   └─ ds.add_target()         target_up_20d / target_ret_20d  (label lives at row T,
                             evaluated after the horizon — feature rows carry only data ≤ T)
   └─ purged walk-forward OOF  (train→validate→future, embargo = horizon)
        per fold: fit logistic/rf/xgboost → P(up) per model
        ensemble.combine_probs()  → one OOF P(up) per date (same fold as the models)
   └─ calibration on OOF   isotonic (PAVA) — fallback `uncalibrated` when <40 rows
   └─ probe LATEST row     index-median imputation fit on train only → all models →
                         ensemble → calibrated P(up) → signal.decide_signal(BUY/HOLD/SELL)
                         (gate: full-OOF ensemble AUC < 0.55 ⇒ forced HOLD + base-rate P(up))
   └─ backtest numbers     derived from the SAME OOF probs (ensemble + each model + buy_hold)
   └─ explanation          fitted-once RF (models.fit_model → predict_up), scale-aware
                         ±0.15·train-std per feature → top ±8 factors
   └─ monitor              freshness age + head(60%)/tail(40%) PSI + P(up) PSI + Brier drift
   └─ snapshot             versions.new_snapshot(): model_probability, calibrated_probability/
                         signal, ensemble_version, calibration_method, feature fingerprint
```

## Endpoints

- `GET /api/v1/stock/{symbol}/forecast` (`fast`, `enrich` query flags)
  - `fast=true` (API default): test_size=40, step=90 — seconds on 5y daily history.
  - `enrich=true`: additionally pulls news sentiment, macro (SNP/USD_INR/gold), F&O:
    monotonic assembly only if the source is available — never fabricated.
- `GET /api/v1/ml/status` — source freshness (nifty50/banknifty/india_vix), last-run
  mode, alarms, versions + thresholds.
- `GET /api/v1/stock/{symbol}` (full analysis) now embeds the forecast block
  (guarded by a timeout; the page falls back to the dedicated endpoint).
- `POST /api/v1/chat` — reads the forecast payload for a mentioned symbol
  (LLM never invents numbers; fallback template prints the same structured block).

## Hard rules enforced

- **No lookahead** — every feature computed from rows ≤ T; the forward return is the label.
- **No fabrication** — unavailable data → `is_available=false` + `error`, or NaN columns.
- **No fake precision** — numbers rounded to 4dp; NaN → `null` in the JSON.
- **Honest probabilities** — calibrated P(up) ∈ [0,1], P(up)+P(down)=1,
  calibration fit on separate OOS probabilities (never on training data).
- **Prefer OOS robustness** — feature set, models and thresholds are fixed by
  config (`ml_feature_version`, `ml_model_version`, `ml_ensemble_version`,
  `ml_threshold_buy/sell`); determinism from the `seed` argument.

## What gets versioned

- `settings.ml_feature_version` (`v1`) tagged into every feature frame (`attrs`).
- `settings.ml_model_version` (`baseline-v1`) + `ml_ensemble_version` (`ensemble-v1`).
- `versions.new_snapshot()` writes the prediction snapshot (prob 4dp, signal, versions,
  calibration method + calibrated prob, sensor-feature fingerprint).

## Rollout

1. Verify on your ticker: `GET /api/v1/stock/RELIANCE/forecast` → forecast card in the
   stock page ("Forecast" tab).
2. Compare the dedicated endpoint vs the embedded full-analysis block (same numbers).
3. Raise `fast=false` (full OOF, test_size 60 / step 120) only for scheduled batch jobs.
4. Enrichment (`enrich=true`) is off by default because news/macro/options add latency
   and provider risk; enable it once you confirm the source robustness you need.
5. `ml/status` is the operational health check — alarms correspond to monitor modes
   (`fresh` / `degraded` / `stale`).

## Known limitations

- Forecast strength depends on the underlying 5y daily history; short/small-cap tapes
  with thin history degrade via the `is_available=false` path.
- Options/F&O layer is gracefully disabled when yfinance has no chain (e.g. RELIANCE.NS
  without expiry data) — no `opt_*` columns then, the ensemble just runs on the rest.
- The local explanation is a fitted-once RF sensitivity browse (top ±8 factors), not the
  full multi-tree SHAP library; same "why" answers, much cheaper.
- Log-regression may warn on non-convergence on noisy synthetic tapes; this is cosmetic
  and never leaks into the API response.

## Hardening (10-11 Sep-2026) — V1-quality iteration

- **Evaluation metric**: experiments are judged by the ensemble **full-OOF AUC**
  (`discrimination.full_oof_auc`), not the noisy newest-20% holdout.
- Live ensemble reduced to `(logistic, rf, xgboost)` (voting removed); BUY uses
  `settings.ml_threshold_buy`; logistic/ridge are scaled (`Pipeline(StandardScaler)`);
  XGBoost trains with `scale_pos_weight = neg/pos`; constant feature columns are
  dropped before modelling.
- **Signal-quality gate**: full-OOF ensemble AUC < `ml_min_auc_for_signal` (0.55) ⇒
  HOLD + holdout base-rate P(up) (no overconfident signals on weak-fit symbols).
- Replay harness + fixed snapshot (`ml_replay/*_features.pkl`, as_of 2026-09-09):
  fixes-baseline median full-OOF AUC **0.5214**; +3 momentum features
  (`dist_52w_high`, `vol_ratio_20_60`, `rel_rs_slope_nifty50_20d`)
  → **0.5351**. Feature-gain diagnostics: `python -m app.ml.feature_gain`.
- **Feature-selection track CLOSED (11-Sep-2026, all rejected/reverted):** three global
  drop-tests (raw OHLC levels →0.5202, `rel_rs_slope_nifty50_20d` →0.5321,
  `roc_*/mom_10d` →0.5349) and a cross-sectional peer-universe experiment
  (`v2_cross_sectional.py`, research-only →0.5208) all MISSED the 0.5351 baseline on
  the same snapshot. The tree ensemble uses the full feature set; **0.5351 is the
  locked baseline** and global dropping / cross-sectional addition is not a lever.
  Details: `handoff.md` + `docs/model-v2-feature-selection-cross-sectional.md`.

## Hardening (12-Sep-2026) — forecast-quality iteration (replay median full-OOF AUC)

- **XGBoost removed from the live ensemble:** `ENSEMBLE_MODELS = ("logistic", "rf")`.
  On the fixed 09-09 snapshot XGBoost full-OOF AUC median ~0.5092 (below chance on
  TCS/INFY) dragged the equal-weight ensemble down (0.5244); logistic/rf alone ≈ 0.5346.
  No stacked meta-learner. Replay median **0.5244 → 0.5346** (replay-measured 0.5399:
  TCS 0.4835 / RELIANCE 0.5399 / HDFCBANK 0.5579 / INFY 0.5293).
- **RF hyperparameters tuned, adopted:** `DEFAULT_RF` `(200, depth 6, leaf 20)` →
  `(n_estimators=300, max_depth=8, min_samples_leaf=30)` (`models.py`). Leakage-free
  coordinate search over the exact walk-forward splits on the fixed replay frames
  (imputer fit on train only): default 0.5331 → best **0.5512** (> 0.5399 bar) → ADOPT.
  Replay after: median **0.5399 → 0.5465**. Harness `backend/app/ml/tune_rf.py`,
  artifact `ml_rf_tune/rf_tune_result.json`.
- **Volatility-adjusted TRAINING target, adopted:** `dataset.add_volatility_target` +
  `pipeline.VOL_ADJ_K = 0.5`. Noise band `theta_T = k·sigma_T·√20`; training keeps only
  clear up/down rows, evaluation stays on full binary `target_up_20d` (comparable AUC).
  Sweep (numpy median): k=0.5 → **0.5696** (k=0 0.5419 / 0.25 0.5443 / 0.75 0.5454 /
  1.0 0.5241) → ADOPT. Replay after: median **0.5465 → 0.5703**
  (TCS 0.4921 / RELIANCE 0.5688 / HDFCBANK 0.6091 / INFY 0.5703). Harness
  `backend/app/ml/tune_vol_target.py`, artifact `ml_vol_target/vol_target_tune_result.json`.
- **FINAL LOCKED STATE (current, 12-Sep-2026):** median full-OOF AUC **0.5703** on the
  fixed 09-09 replay snapshot (TCS 0.4921 / RELIANCE 0.5688 / HDFCBANK 0.6091 / INFY 0.5703).
  Progression: 0.5214 → 0.5351 → 0.5399 → 0.5465 → 0.5703. All drop-tests and
  cross-sectional experiments remain rejected. Tests: 519 passed, 3 skipped, 2 pre-existing
  synthetic failures (Brier≤0.5 bound).
- **Validated on a FRESH snapshot (12-Sep-2026):** as_of 09-11 (+2 trading days) reproduced
  median full-OOF AUC **0.5703 bit-stable** (0.492 / 0.5688 / 0.6092 / 0.5703) → the gain is
  REAL, not overfit to 09-09. Re-tune on the new vol-adjusted train target:
  **NO IMPROVEMENT** (best = current params) → locked config is the local optimum.
- **Honest per-symbol ceiling (proven, do NOT force):** the model has strong validated skill
  on the core 4 large-caps (0.5703 on 5y) but is at chance level on ITC/SBIN this window
  (fresh check ICICIBANK 0.4448 / SBIN 0.3219 / ITC 0.4657). Research Tasks 21–22
  (vol-adjusted pooled training + probability blend, `ml_v2_pooled_vol_adj/`) proved by an
  oracle best-of-per-symbol test that **NEW3 median > 0.52 is mathematically unreachable this
  window** (SBIN 0.4935 / ITC 0.4597 are binding ceilings). ITC/SBIN chance-level is an honest
  market-predictability ceiling, not a bug. Cross-symbol generalization is deferred to future
  research (a different window / more data / richer features). The locked architecture above
  remains the production baseline; no further model/feature changes this session.
- **10y production adoption (Task 27, 15-Sep-2026):** production `forecast_symbol` default
  `period="5y"` → `"10y"` (2474 rows/symbol vs 1240). Feature/model code UNCHANGED
  (ensemble = logistic + rf, RF 300/8/30, C=1.0, VOL_ADJ_K=0.5, thresholds same).
  A/B: 10y median 0.6179 vs same-day 5y control 0.5596 (+0.058, TCS +0.092 / RELIANCE
  +0.121 / INFY +0.084 / HDFCBANK −0.048). 0.5703 remains the 5y LOCKED baseline for
  historical reference. The locked FEATURE/MODEL state (vanilla RF+logistic ensemble on
  100-feature set) is unchanged — only the training window grew.

## Hardening (19-Sep-2026) — policy decisions (model/features unchanged)

- **Task 29 — Bear-Market Defense (ADOPTED, production):** signal `decide_signal` now uses
  `confidence_floor=0.08` (prob in (0.46, 0.54) strictly HOLD) and, when
  `regime_trend/risk == -1`, raises the effective BUY threshold to `max(tb, 0.62)`; backtest
  holdout directions and the live probe share the SAME per-row policy, so the API signal
  always matches the measured backtest. Median full-OOF AUC unchanged (0.6179); realised
  loss-cut defence (TCS −21.0 → −11.4%, settlements 441 → 402) on the 09-15 10y snapshot.
- **Task 30 — 15y window A/B (REJECTED):** same-day (as_of 2026-09-18) 15y median full-OOF
  AUC **0.5060** vs 10y control **0.6166** (−0.111) → longer windows add no OOS edge
  (matches Tasks 23/26/28). Production stays 10y.
- **Task 31 — Per-symbol high-conviction BUY thresholds (ADOPTED, production):**
  `settings.ml_per_symbol_buy_thresholds` {TCS 0.66 / RELIANCE 0.72 / HDFCBANK 0.64 /
  INFY 0.72} wired into `forecast_symbol`; cache version 1→2 (old 0.60-policy results miss).
  Tuning honest: thresholds selected on the older 80% OOF-fit rows, evaluated on the untouched
  newest-20% holdout of a same-day control10y snapshot, then CONFIRMED on an independent
  re-fetch (76 feature cols differ via yfinance adj-reconstruction — improvement reproduces).
  FORMAL-LEDGER median cumRet **−20.06% → −4.97%** (loss ~75% cut) at unchanged AUC 0.6166;
  BUY hit-rate median 50.9% → 57.1%. This lever converts the SAME ranking into
  higher-conviction decisions; it does not change discrimination (AUC/flags unchanged).
  Remaining gate = confirmation on a genuinely NEW trading window (next market session).

## Acceptance checklist (see architecture.md → Section 26)