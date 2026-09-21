# Feature-Selection & Cross-Sectional Experiments (11-Sep-2026)

> **Note (12-Sep-2026):** the 0.5351 baseline below has been **superseded** — the
> forecast-quality iteration (XGBoost drop + RF tuning + volatility-adjusted train
> target) raised the same-snapshot median full-OOF AUC to **0.5703**. See
> `handoff.md` and `architecture.md` (12-Sep addendum) before comparing new results.
> The experiments recorded in this document remain rejected/reverted as of 11-Sep-2026.

**Status:** research-only; production V1 remains **FROZEN**  
**Baseline locked (at the time):** ensemble median full-OOF AUC **0.5351** (fixed 09-Sep-2026 replay snapshot)  
**Module (cross-sectional):** `backend/app/ml/v2_cross_sectional.py`  
**Artifacts:** `backend/ml_cross_sectional/*_features.pkl`, `backend/ml_benchmark/benchmark_17syms_20260911T115944Z.*`

## Executive summary

Every experiment to exceed median full-OOF AUC **0.5351** failed. Global feature
dropping (3 tests) and a new-information cross-sectional peer-universe lever both
regressed or came in flat on the SAME fixed snapshot. The tree ensemble benefits
from the full feature set (including interactions); 0.5351 is the honest,
calibrated ceiling for the 20-day single-stock directional task on this data.

## Measurement protocol (shared by all experiments)

- Evaluation metric: **ensemble full-OOF ROC-AUC** (`discrimination.full_oof_auc`).
  The newest-20% holdout (~40–88 rows) is noise and was NOT the decision metric.
- Input: frozen `ml_replay/*_features.pkl` snapshot, as_of 2026-09-09, 1239 rows/symbol
  (TCS / RELIANCE / HDFCBANK / INFY), walk-forward test 60/step 60/min_train 260/embargo 20.
- A/B purity: every candidate was compared against the SAME frames. A fresh data pull
  (e.g. `pipeline._build_symbol_input`) changes `as_of` and row count, which itself
  shifts AUC — a classic confound. Cross-sectional frames were therefore built ON TOP
  of the exact replay snapshot, adding only the new `cs_*` columns.

## Experiment 1 — Raw OHLC level drop (FILE 10)

| Symbol | Baseline | Drop OHLC | Delta |
|---|---|---|---|
| TCS | 0.4796 | 0.4749 | −0.0108 |
| RELIANCE | 0.5351 | 0.5202 | −0.0149 |
| HDFCBANK | 0.5472 | 0.5543 | +0.0132 |
| INFY | 0.5214 | 0.5177 | −0.0037 |
| **Median** | **0.5351** | **0.5202** | **−0.0149** |

Removing `Open/High/Low/Close/Adj Close` from the feature matrix **regressed** the
median. Raw price levels were flagged as high-gain by `feature_gain.py` and the
economic concern (non-stationary levels, cross-stock proxy) is real — but on a
single-symbol walk-forward fit the levels still carried usable regime information.
**REJECTED, reverted** (`_default_feature_cols` restored).

## Experiment 2 — `rel_rs_slope_nifty50_20d` drop

| Symbol | Baseline | Drop rs_slope | Delta |
|---|---|---|---|
| TCS | 0.4796 | 0.4835 | +0.0039 |
| RELIANCE | 0.5351 | 0.5321 | −0.0030 |
| HDFCBANK | 0.5472 | 0.5424 | −0.0048 |
| INFY | 0.5137 | 0.5141 | +0.0004 |
| **Median** | **0.5351** | **0.5321** | **−0.0030** |

`rel_rs_slope_nifty50_20d` was the gain-analysis "weakest / never top-15 → drop
candidate". Removing it was flat-to-slightly-negative (noise-level but baseline
stronger). **REJECTED, reverted.** Gain ranking is correlation-based; it does not
prove a feature is removable — tree models use low-gain features in splits.

## Experiment 3 — `roc_1d/5d/10d/20d` + `mom_10d` drop

| Symbol | Baseline | Drop roc_/mom_ | Delta |
|---|---|---|---|
| TCS | 0.4796 | 0.4780 | −0.0016 |
| RELIANCE | 0.5351 | 0.5349 | −0.0002 |
| HDFCBANK | 0.5472 | 0.5382 | −0.0090 |
| INFY | 0.5137 | 0.5208 | +0.0071 |
| **Median** | **0.5351** | **0.5349** | **−0.0002** |

Effectively flat in aggregate but strongly mixed per symbol (INFY **+0.0071**,
HDFCBANK **−0.0090**). Interpretation: what is noise for one symbol is signal for
another; **global single-batch dropping is not viable**. **REJECTED, reverted.**
(A per-symbol feature set was deliberately NOT pursued — tuning on a 4-symbol
snapshot would overfit and not generalize.)

## Experiment 4 — Cross-sectional relative-strength (FILE 11)

### Features (all causal, data ≤ T)
- `cs_mom_rank_{10,20,60}d` — momentum percentile rank of the symbol within a
  20-name diversified NIFTY50 peer universe (0–1)
- `cs_rel_ret_vs_median_{10,20,60}d` — symbol h-day return minus universe-median
- `cs_rs_ratio_20d` — symbol Close / equal-weight universe index − 1

Peer universe: TCS, INFY, RELIANCE, HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK,
HINDUNILVR, ITC, MARUTI, TATAMOTORS, SUNPHARMA, DRREDDY, TATASTEEL, JSWSTEEL, LT,
BHARTIARTL, NTPC, BAJFINANCE (9 sectors; contains all 4 smoke symbols).

### Results

| Symbol | Baseline | +CS features | Delta |
|---|---|---|---|
| TCS | 0.4796 | 0.4903 | +0.0107 |
| RELIANCE | 0.5351 | 0.4936 | −0.0415 |
| HDFCBANK | 0.5472 | 0.5372 | −0.0100 |
| INFY | 0.5137 | 0.5208 | +0.0071 |
| **Median** | **0.5351** | **0.5208** | **−0.0143** |

**REJECTED.** The peer-context layer added no discriminative signal for the
ensemble on this snapshot (RELIANCE −0.0415 worst). Module kept as a research
artifact. Production V1 untouched — no `cs_*` features are wired into
`features.py`/`pipeline.py`.

## Why dropping keeps failing (takeaway for future agents)

1. **Info is in interactions.** `roc_*`/`mom_*`/levels/rs_slope never shine as
   top-gain features on their own, but removing them makes splits marginally worse.
2. **Per-symbol heterogeneity.** A feature that is noise for INFY is signal for
   HDFCBANK; a global exclude set hurts whoever needed it.
3. **New information ≠ better tree splits.** Cross-sectional peer features are
   economically sensible but were swamped by already-present market/relative
   features; they added 0 net discriminative value here.

## Locked baseline (final state)

- **median full-OOF AUC 0.5351** — KEPT: FILE 8 momentum features
  (`dist_52w_high`, `vol_ratio_20_60`, `rel_rs_slope_nifty50_20d`) + all signal
  vetting fixes. REJECTED/REVERTED: every drop-test + cross-sectional.
- 17-symbol benchmark (11-Sep-2026): 16/17 ok (TATAMOTORS upstream-404), full-OOF
  per-model median rf 0.5518 / logistic 0.5193 / xgboost 0.5181; regime-bucketed
  OOF AUC bull 0.537 / sideways 0.5026 / bear 0.5425.
- Tests: **519 passed, 3 skipped, 2 pre-existing synthetic failures** (Brier ≤ 0.5
  bound on random data — unrelated to feature/model state).
- Do NOT re-attempt feature dropping or cross-sectional additions without new evidence
  (different data, longer horizon, or genuinely new feature families).