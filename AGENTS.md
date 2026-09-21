# AGENTS.md — Onboarding for new AI agents / engineers

> Read this first. It tells you what the project is, what state it is in,
> where the real documentation lives, and the rules that must never be broken.
> Khud ke liye: start hamesha yahan se karo.

**Project:** AVORA / EquityLens — Indian (NSE/BSE) stock analysis platform with a
research-grade **20-trading-day probabilistic ML forecasting system**. FastAPI
backend + Next.js 15 frontend + optional LLM chat.

---

## 1. Where to look (in order)

| File | Purpose |
|---|---|
| `AGENTS.md` | **THIS FILE** — quick onboarding: state, commands, rules, conventions. |
| `handoff.md` | Project roadmap — tracked `[DONE]` / `[IN_PROGRESS]` / `[TODO]`. **Update after every change.** |
| `architecture.md` | Full architecture diagram, folder structure, data flow, the 10-task master plan, and the **locked measurement/validation layer** spec. |
| `model-upgrade-plan.md` | Model-upgrade strategy doc. |

## 2. Current state (finalised 12-Sep-2026 — PRODUCTION LOCKED)

- **All 10 master-plan tasks are DONE** (ML scaffolding → data → features → factors →
  PIT fundamentals/events/sentiment/macro → regime/dataset/options → models/backtest →
  ensemble/calibration/signal/explain/monitor → API/frontend/chat/docs).
- **Post-Task-10 chat/screening intelligence** (Hinglish/English chat, deep-dives,
  screening rules, concept KB, analyst reasoning) is DONE. Further chat polish
  (11-Sep-2026, production, no model/feature changes):
  - **FILE 12** — greetings/welcome smalltalk (context-aware EN+HI), `_who_made_you`,
    `penny/penni` → `maxPrice = 100.0`.
  - **FILE 13** — out-of-scope decline: `OUT_OF_SCOPE_PATTERNS` (13 topical groups incl.
    weather/sports/politics/math/health/jokes) + `_out_of_scope_reply` (EN+HI) checked in
    `_general_reply` before the LLM; `GENERAL_SYSTEM_PROMPT` scoped to NSE stocks +
    AVORA dashboard. Known wrinkle: OOS topic mixed with a stock keyword (e.g. `cricket
    match prediction do`) still routes to the stock pipeline (see backlog).
  - **FILE 14** — "N din se bullish" duration filter: `parse_intent` → `bullishDays` (cap
    60), `filter_predictions` keeps `retNd > 0` (positive N-day return, horizons
    3/5/7/10/15/20d) with strict-streak fallback (`upDaysConsecutive >= N`) when the exact
    horizon isn't precomputed; `_fallback_reply` HI/EN condition text; `bullishDays` added
    to `is_smalltalk` bail-out + `is_stock_query` trigger.
  - **FILE 15** — financial disclaimer appended to every NSE/stock answer (stock pipeline +
    general/NSE concept chat): "AI-generated, not financial advice — research or consult a
    SEBI-registered advisor" + "thank you for using AVORA Chatbot" (EN/HI). Skipped for
    smalltalk, who-made-you, and out-of-scope declines.
- **LLM chat (Groq) active (11-Sep-2026):** `backend/.env` (not `.env.example`) holds
  `OPENAI_API_KEY`/`OPENAI_BASE_URL=https://api.groq.com/openai/v1`/`OPENAI_MODEL=groq/compound`;
  `process_chat` returns `source: "llm"`. Verified live: HTTP 200 + end-to-end reply.
  A transient Groq 413 (Payload Too Large) on a small payload was server-side; retry succeeds.
  - Chat tests: `tests/test_chat_service.py` → **65 passed** (parse_intent, routing,
    filter_predictions incl. bullishDays positive-return/streak-fallback, concept KB,
    deep-dive, fundamentals compaction, analyst report, smalltalk, who-made-you, OOS,
    financial disclaimer).
  - Full backend suite numbers stated elsewhere in this file predate FILE 12–15; the
    15 new chat tests (`50` → `65`) are on top of those counts.
- **Measurement / validation hardening (07-Sep-2026)** — locked:
  - Real daily **marked-to-market equity-curve backtest** (no fake per-trade drawdowns).
  - **No calibration leakage**: isotonic fitted on older 80% of OOF probs, all headline
    metrics on the newest 20% **holdout only**.
  - Baselines (`always_up`, `always_down`, seeded `random`) + **shuffled-target control**
    on the same holdout → chance-level ROC-AUC/recall reference.
  - `metric_groups` provenance: per-bet vs daily-equity-curve metrics always separated.
- **Read-only audit + forensic reconciliation of the ensemble trade-count (07-Sep-2026)**:
  - Traced `calibrated probability → direction → equity_ledger → n_trades` end-to-end.
  - `n_trades` = settled ledger positions = number of `direction ≠ 0` (long) rows.
  - Reported TCS ensemble `n_trades=88, cum_return −27.6%` confirmed **correct for the
    2026-09-06 data snapshot** (all 88 holdout calibrated probs ≥ 0.5 → identical to
    `always_up`); **bit-deterministic within a fixed input frame**; the 67–69-long figures
    came from **different upstream data pulls**, not a code bug.
- **Task 11 — V1 Forecast Benchmark & Failure Analysis (07/08-Sep-2026)** — measurement only,
  no production behavior changed:
  - `backend/app/ml/benchmark.py` — repeatable multi-stock benchmark of the CURRENT V1
    20-day forecast: per-symbol holdout AUC/Brier/ECE + trading metrics, same-window
    baselines, full-OOF per-model comparison + rf feature-group ablation + failure/regime
    diagnostics, cross-stock aggregation, fingerprints/versions, JSON+txt outputs +
    CLI (`python -m app.ml.benchmark [--smoke|--symbols ...]`).
  - `pipeline._build_symbol_input` extracted (shared by live `forecast_symbol` and the
    benchmark) — behavior-preserving.
  - Tests: `tests/test_benchmark.py` (18, synthetic). Full suite **423 passed**.
  - Real runs (outputs under `backend/ml_benchmark/`): smoke (4 syms, 827s) and
    17-symbol default universe (16 ok + TATAMOTORS upstream-404 skip). Key finding:
    the ensemble is effectively always-long on this snapshot (calibrated probs pinned
    at the 0.50 boundary in down windows) and bled in the bearish holdout; only
    SUNPHARMA/SIEMENS ended positive.
- **Task 17 — Rigorous Frozen-V1 vs Pooled Model Comparison (09-Sep-2026)** — DONE, **KEEP FROZEN V1**:
  - Full 20-symbol real-data comparison on `task17-v1` snapshot (`c4ccdc68…`, 2021-09-09 → 2026-09-09).
  - Matched sample: 19 198 rows, 20 symbols, 960 dates, 16 purged walk-forward folds, seeds 17/31/53.
  - V1 global AUC: **0.5173**; pooled median AUC delta: **−0.032** (bootstrap 95% CI [−0.0498, −0.0231]).
  - Per-symbol win rate: 6/20 (30%); V1 wins 14/20; fold robustness FALSE; seed stability TRUE.
  - **Decision: KEEP FROZEN V1** — pooled learning does not beat frozen V1 on the identical snapshot.
  - Config B (relative features) adds **no signal** — all 20 symbols collapse into one `__all__` sector group.
  - Research-only module `v2_v1_fair_compare.py` does NOT modify production V1.
  - Results: `backend/ml_v2_fair_compare/task17_result.json`.
  - Full report: `docs/model-v2-v1-fair-comparison.md`.
- **Task 18 — Selective Forecasting & Uncertainty-Aware Decision Research (10-Sep-2026)** — DONE,
  research-only:
  - Applied fixed margin, entropy, V1-component disagreement, and conformal rules to
    frozen V1 out-of-sample probabilities on the Task 17 snapshot.
  - Entropy and margin-0.15 were research-promising; disagreement failed coverage/
    robustness requirements. **Production remains KEEP FROZEN V1.**
  - Report: `docs/model-v2-selective-forecast-results.md`; result:
    `backend/ml_v2_selective_forecast/task18_selective_forecast_result.json`.
- **Task 19 — Entropy Selective-Forecast Validation (10-Sep-2026)** — DONE,
  research-only:
  - Validated the frozen Task 18 entropy rule (`entropy <= 0.65`) against the
    same snapshot and evaluation protocol without retuning or production wiring.
  - Result: **EVIDENCE STILL INSUFFICIENT** for production integration.
    Retained coverage was 45.66% (6,244 retained; 7,430 abstained), with
    all-vs-selective metrics, fold/symbol/seed, calibration, economic,
    shuffled-target, and paired bootstrap diagnostics recorded.
  - Full backend suite: **486 passed, 3 skipped, 0 failed**. V1 remains frozen.
- **Task 20 — Expected Return + Risk Forecasting Research (10-Sep-2026)** — DONE,
  research-only:
  - Evaluated expected-return magnitude + risk forecasting beyond the weak V1
    directional probability: two-stage (P(up) × E(return|up)) vs direct regression
    (Ridge / Random Forest / XGBoost), risk outputs (cond vol, residual MAD,
    quantile intervals), walk-forward (test 60/step 60/min-train 260/embargo 20),
    seeds 17/31/53, newest-20% holdout, null control, bootstrap CI, and one
    entropy-interaction diagnostic on the Task 17 snapshot (20 symbols).
  - Result: **REJECTED.** Best holdout RMSE (RF 0.082) ≈ shuffled-target null
    (0.081); direction correlations small (Ridge 0.125 / RF 0.040 / XGB 0.026);
    two-stage lowers MAE only via shrinkage; entropy interaction no_signal
    (corr −0.12). Return forecasts carry no genuine OOS signal. V1 frozen.
  - Module `v2_return_risk_forecast.py` + 35 tests; artifact
    `backend/ml_v2_return_risk/task20_result.json`;
    report `docs/model-v2-return-risk-results.md`.
- **V1 forecasting-quality iteration (10-11 Sep-2026)** — live-pipeline fixes +
  measurement upgrade (baseline recorded):
  - **Stable metric locked**: `forecast_frame` result now carries
    `discrimination.full_oof_auc` (ensemble full-OOF ROC-AUC). The small newest-20%
    holdout (~88 rows) proved noise-driven (e.g. INFY holdout 0.17 ↔ full-OOF 0.52);
    full-OOF AUC is now the go/no-go measure for feature/model experiments.
  - Model/vetting fixes (production): `voting` removed from live `ENSEMBLE_MODELS`
    (`(logistic, rf, xgboost)`); BUY threshold reads `settings.ml_threshold_buy`;
    logistic/ridge wrapped in `Pipeline(StandardScaler)` (scale-sensitive);
    XGBoost gets `scale_pos_weight = neg/pos` (edge-safe); constant feature columns
    are dropped in `forecast_frame` (no NaN-argmax collapse).
  - **Signal-quality gate**: when full-OOF ensemble AUC < `ml_min_auc_for_signal`
    (0.55) ⇒ signal = HOLD with holdout base-rate `P(up)` (+ matching
    confidence/reason) instead of an overconfident probability.
  - Fixes' baseline (replay, as_of 2026-09-09, 1239 rows/symbol):
    **median full-OOF AUC = 0.5214** (TCS 0.4857 / RELIANCE 0.5087 / HDFCBANK 0.5411 /
    INFY 0.5214). Excess-return (market-relative P(up)) target experiment measured on
    full-OOF = median 0.3885 ⇒ **REJECTED, fully reverted** (no `mkt_nifty_close`/
    `market_col` left in code).
  - **FILE 8 (kept)**: 3 momentum features — `dist_52w_high`, `vol_ratio_20_60`
    (`features.py`), `rel_rs_slope_nifty50_20d` (`market_features.py`, list + engine
    synced). Replay median full-OOF AUC **0.5214 → 0.5351**.
  - **FILE 9**: `backend/app/ml/feature_gain.py` — per-symbol walk-forward XGBoost
    gain diagnostic (top-15 + new-feature share). 100 features/symbol (uniform share
    0.010). New-feature group share 3.0–3.7%; individually ~0.017–0.018 (~1.7–1.8×
    uniform): `vol_ratio_20_60` strongest, `dist_52w_high` moderate,
    `rel_rs_slope_nifty50_20d` weakest (never top-15 → drop candidate). Raw OHLC
    levels rank high-gain (robustness flag). Full detail in `handoff.md`.
  - Replay harness `backend/app/ml/replay_snapshot.py` (save/run over fixed
    `backend/ml_replay/*_features.pkl`, as_of 2026-09-09).
  - **Feature-selection track (FILE 10, CLOSED — all rejected/reverted):** 3 global
    drop-tests on the SAME fixed 09-09 snapshot all missed the 0.5351 baseline —
    raw OHLC levels 0.5351→0.5202, `rel_rs_slope_nifty50_20d` →0.5321, `roc_1d/5d/10d/20d`
    +`mom_10d` →0.5349. Lesson: the tree ensemble uses the FULL feature set (incl.
    interactions); global dropping is not a lever. `_default_feature_cols` is the
    plain baseline.
  - **FILE 11 (cross-sectional, research-only, REJECTED):** `backend/app/ml/v2_cross_sectional.py`
    adds 7 peer-universe features (`cs_mom_rank_{10,20,60}d`,
    `cs_rel_ret_vs_median_{10,20,60}d`, `cs_rs_ratio_20d` from a 20-symbol NIFTY50
    peer panel; causal, data ≤ T). Measured clean (frames built ON TOP of the exact
    09-09 replay snapshot; a fresh `--save` pull alone introduces an as_of confound).
    median full-OOF AUC 0.5351→0.5208 (RELIANCE −0.0415 worst) → **REJECTED, kept as
    research artifact only**. Production V1 is untouched.
  - **FINAL LOCKED STATE (15-Sep-2026):** **median full-OOF AUC 0.5703** was the
    feature/model baseline on the 5y frame (12-Sep-2026). **10y adoption (15-Sep-2026,
    Task 27):** `forecast_symbol(period="10y")` is now the production default (routes,
    chat). Measured: same-day 10y median 0.6179 vs same-day 5y 0.5596 (+0.058, 3/4
    symbols improved on TCS/RELIANCE/HDFCBANK/INFY). `forecast_frame`/`build_feature_frame`
    signatures unchanged; old `period="5y"` callers (benchmarks, replay baseline, v2
    research) still work as-is. Locked baseline is the 10y frame for production.
    REJECTED/REVERTED: all drop-tests, cross-sectional, macro-expansion (Task 23),
    fundamentals PIT (Task 26, flat), sector indices (Task 28, −0.011).
    `handoff.md` + `docs/` are the source of truth.
    17-symbol benchmark (11-Sep-2026): 16/17 ok (TATAMOTORS upstream-404), full-OOF
    per-model median rf 0.5518 / logistic 0.5193 / xgboost 0.5181. Tests: 519 passed,
    3 skipped, 2 pre-existing synthetic failures (Brier≤0.5 bound on random data).
    Do NOT re-attempt feature dropping or cross-sectional additions without new evidence.
    RF + logistic-C re-tune on the NEW vol-adjusted train target (`tune_retune_models.py`,
    12-Sep-2026): NO IMPROVEMENT — best = current params (0.5703, not strictly
    > 0.5703); DEFAULT_RF (300/8/30) + logistic C=1.0 unchanged.
    Fresh-snapshot validation (12-Sep-2026, as_of 09-11, +2 trading days): median full-OOF
    AUC **0.5703 reproduced bit-stable → VALIDATED (not snapshot-overfit)**; broader
    new-symbol check weaker (ICICIBANK 0.4448 / SBIN 0.3219 / ITC 0.4657 — skill is
    symbol-specific). Baseline snapshot restored (`ml_replay_20260909_backup`), locked
    baseline reproducible.
- **Task 21 — Cross-symbol generalization + vol-adjusted POOLED training probe
  (12-Sep-2026)** — research-only (`v2_pooled_vol_adj.py`, 6 tests), **NO ADOPT**:
  - Diagnostic D1: vol-adjusted coverage at k=0.5 is **uniform across the 20-symbol
    panel** (0.57–0.67; new symbols ICICIBANK 0.5975 / SBIN 0.5697 / ITC 0.5811) → the
    k=0.5 threshold DOES transfer; target coverage is NOT the new-symbol failure cause.
  - Diagnostic D2: per-symbol full-OOF AUC with vs without raw OHLC levels is **~neutral
    on the 7 focus symbols** (ICs/SBIN/ITC ±≤0.01) → raw levels are NOT the failure driver
    (per-symbol models already see one price scale).
  - Pooled leave-one-symbol-out (ONE rf, locked 300/8/30, trained on other symbols'
    rows up to each fold's embargo cut, held-out symbol never in the pool; vol-adjusted
    train label, full binary eval; same purged splits as the replay):
    median over CORE4 (drop-levels) **0.6428** (TCS 0.6428 / RELIANCE 0.4224 /
    HDFCBANK 0.6629 / INFY 0.5911) vs per-symbol median 0.5557 → **≥0.55 criterion
    PASSES**; median over NEW3 **0.4935** (ICICIBANK 0.5867 / SBIN 0.4935 / ITC 0.3779)
    vs per-symbol median 0.4307 → **>0.52 criterion FAILS narrowly** → **ADOPT=False**.
  - Honest read: pooling lifts the two WORST per-symbol symbols dramatically
    (ICICIBANK 0.43→0.59, SBIN 0.37→0.49, TCS 0.50→0.64, HDFCBANK 0.61→0.66) but
    hurts RELIANCE (0.55→0.42) and ITC (0.46→0.38) → high per-symbol variance, not a
    uniform lift. keep-vs-drop raw levels ~neutral in pooled too. Production untouched;
    result: `backend/ml_v2_pooled_vol_adj/task21_pooled_voladj_result.json`.
- **Task 22 — Per-symbol + pooled probability BLEND (12-Sep-2026)** — research-only
  (`v2_pooled_blend.py`, 4 tests), **NO ADOPT**: P_blend = w·P_pooled + (1−w)·P_per_symbol,
  w ∈ {0.3,0.5,0.7}, both full-OOF P(up) vectors re-derived on the same 440 test dates/symbol
  (perfectly date-aligned; per-symbol+pooled AUCs reproduce Task 21 to 4dp). Result: blending
  DILUTES the pooled signal — best new3 median 0.4395 (w=0.3) vs pure pooled 0.4935; core4
  0.5555–0.5835 (≥0.55 passes, but new3 never clears 0.52). Even oracle best-of-per-symbol
  caps new3 at 0.4935 (ITC 0.46 / SBIN 0.49 binding). Blend buffers RELIANCE (−0.13 pooled
  damage → 0.533 @w=0.3) and ITC (0.378→0.440), but that's small change vs the loss of pooled
  lift. No weight passes acceptance; production untouched.
  Artifact: `backend/ml_v2_pooled_vol_adj/task22_pooled_blend_result.json`.
- **Task 23 — Global/Commodity/Currency macro features (12-Sep-2026)** — research-only,
  **REJECTED, FULLY REVERTED**: fed the already-registered `nasdaq/nikkei/hangseng/vix/brent`
  into the enrich macro loop (`pipeline._build_symbol_input`, was `("snp500","usd_inr","gold")`)
  plus `banknifty` in the market dict (all in `SERIES_REGISTRY`; `add_macro_features` generic,
  no data-source/feature-engine change). Enriched replay snapshot (as_of 09-11, `--save --enrich`
  flag added to `replay_snapshot.py`): median full-OOF AUC **0.5572** vs locked **0.5703** → fails
  strict >0.5703 acceptance → **REJECTED**. Same-day (09-11) no-enrich control reproduces 0.5703
  exactly, isolating the drop to the macro features (not the +2-day as_of shift). pipeline.py
  reverted to `("snp500","usd_inr","gold")` + `("nifty50","india_vix")`; `ml_replay/` restored to
  the locked 09-09 baseline. Pre-existing latent enrich-path bug surfaced (fixed later in
  Task 25, 15-Sep-2026): `_series_close(...) or mdf["Close"]` (Series bool-eval) silently drops
  all macro series; news (NewsData.io 401) + options (`get_options_inputs(clean)` signature
  mismatch, `pipeline.py:1174`, plus nested-`metrics` not materialized) also degrade gracefully.
  Enriched snapshot artifact: `backend/ml_replay_enriched_macro/`.
  Frozen 0.5703 baseline unchanged.
- **Task 24 — GDELT historical news-sentiment A/B (13-Sep-2026)** — research-only, **NOT MEASURED
  (BLOCKED → PAUSED)**: modules written — `backend/app/ml/gdelt_sentiment.py` (GDELT DOC 2.0
  `mode=timelinetone` 5-year daily AvgTone fetcher; tone /100 to [-1,1] scale; each news day = one
  causal PIT article reused via the existing PIT-safe `sentiment_features.add_sentiment_features`)
  and `backend/app/ml/gdelt_ab.py` (clean A/B on the EXACT locked 09-09 snapshot: baseline
  `forecast_frame` vs +`sent_*`; no fresh `--save` → no as_of confound; strict acceptance > 0.5703;
  tone cached to `backend/ml_gdelt/*_tone.pkl`). Why: NewsData free tier returns only today's
  articles (sent_* non-null on 1/1241 rows → dropped as near-constant → unmeasurable), so GDELT
  (free, 2015+/2017+ full-text, no API key) was chosen to get real 5-year tone. **BLOCKED:** after
  the first artlist 200 OK, GDELT temp-banned this network — every request (even 1d artlist) 429,
  307x in ~66 min of a bounded cooldown job (per user's stop condition → killed). No tone data
  fetched, A/B not run → **sentiment branch stays untested; locked 0.5703 unchanged**. Resume only
  when GDELT serves 200 from this network; then `python -m app.ml.gdelt_ab --symbols TCS,RELIANCE,HDFCBANK,INFY`.
- **Task 26 — Fundamentals PIT history A/B (15-Sep-2026)** — research-only, **MEASURED, REJECTED**:
  - Problem: `add_fundamental_features` accepts `snapshots` (list of PIT quarterly
    snapshots) but `_build_symbol_input` always sent a single current snapshot → historical
    rows all-NaN → `fund_*` never reached training. Wired the FULL PIT path:
  - `fundamental_service.get_quarterly_fundamentals()` — yfinance `quarterly_financials` +
    `quarterly_balance_sheet` per quarter-end, `available_at = quarter_end + 45d`
    (SEBI filing-lag proxy), trailing P/E from frame quarter-end closes
    (`pipeline._quarter_end_closes`), ROCE/ROE/D/E/OPM/growth-yoy/fundamental_score computed.
  - `_build_symbol_input(..., fundamentals_pit=True)` opt-in lever (default OFF — locked
    baseline byte-identical); `replay_snapshot.py --fundamentals-pit` saves to
    `ml_replay_pit_fund/` (never clobbers baseline `ml_replay/`).
  - Measurement (fresh real-data snapshot, as_of 2026-09-15, TCS/RELIANCE/HDFCBANK/INFY):
    **median full-OOF AUC 0.5701 vs baseline 0.5703** → −0.0002, **no edge** →
    **REJECTED** (strict >0.5703 acceptance not met). Per-symbol: TCS +0.029 / RELIANCE
    +0.021 / HDFCBANK −0.012 / INFY −0.020. Root cause: yfinance quarterly exposes only
    ~5 quarter-ends → coverage just 13–20% of rows (~1.25y of a 5y frame) → too thin to
    move AUC. Production untouched; the negative-work infrastructure stays as a research
    lever for when a real multi-year PIT fundamentals feed becomes available.
  - Tests: `tests/test_enrich_fixes.py` +2 (13 total, all green): PIT path spreads
    stepwise-constant fund_* across history (>30% of a synthetic frame), baseline flag-off
    remains last-row-only.
- **Task 27 — Longer-history (10y) window A/B (15-Sep-2026)** — **MEASURED, ADOPTED**:
  - `replay_snapshot.py --period 10y` lever (label auto = `10y` sub-dir; explicit
    `--label` override added). `_build_symbol_input` was already period-parametric.
  - Exposed a **latent 10y-only crash** (bug fix): `regime.risk_regime` formed
    `on`/`off` masks that pandas aligns to the UNION of the nifty & vix calendars; with
    longer ragged history (nifty ~2466 rows vs vix ~2451) the bucket outlived the nifty
    index → `pd.Series(bucket, index=nifty_close.index)` ValueError (Length 2468 vs 2466).
    Fix: classify VIX on its own calendar first, then reindex the percentile to the
    nifty calendar (still causal). `tests/test_regime.py` +1 (14 total, regression).
  - Measurement (fresh real-data pulls, same as_of 2026-09-15, TCS/RELIANCE/HDFCBANK/INFY):
    **10y median full-OOF AUC 0.6179** (TCS 0.5774 / RELIANCE 0.6658 / HDFCBANK 0.5513 /
    INFY 0.6583) **vs same-day 5y fresh control 0.5596** (TCS 0.4856 / RELIANCE 0.5448 /
    HDFCBANK 0.5997 / INFY 0.5745) → **+0.058, 3/4 symbols up** (TCS +0.092 /
    RELIANCE +0.121 / INFY +0.084; HDFCBANK −0.048). Same-day control isolates the period
    effect from the as_of shift (fresh 5y itself reads 0.5596 vs locked 0.5703 — the locked
    snapshot's drift is why same-as_of comparison is mandatory).
  - **ADOPTED (user-approved, 15-Sep-2026):** `forecast_symbol` default `period="5y"` →
    `"10y"`; hot paths (`routes/forecast.py`, `chat_service._maybe_forecast`) call 10y.
    `build_feature_frame`/`forecast_frame` signatures unchanged — only the input window
    grew (2474 rows/symbol vs 1240). 5y callers (benchmark, replay `--period 5y`, v2
    research) pass period explicitly and still work. Cache fingerprint includes row set
    → old 5y cached results are misses (regenerated on 10y), no stale mixing. Live smoke:
    `forecast_symbol('TCS')` → HOLD P(up)=0.573, full-OOF AUC 0.5774 (matches 10y replay).
  - Tests: `test_regime.py` 14 passed, `test_enrich_fixes.py` 14 passed,
    `test_forecast.py` + market_features + data_service + benchmark + W1 passed
    (114 total in the group).
- **Task 28 — Sector-index relative features A/B (15-Sep-2026)** — research-only, **MEASURED, REJECTED**:
  - Lever: `add_sector_relative_features()` (market_features.py) — the relative
    strength/corr/vol family benchmarked against a SECTOR index instead of NIFTY 50,
    appended on top (nifty bench stays). Registered `cnxit` (`^CNXIT`, NIFTY IT) in
    SERIES_REGISTRY. Per-symbol sector map in `replay_snapshot.SECTOR_MAP`:
    TCS/INFY→cnxit, HDFCBANK→banknifty, RELIANCE→nifty50 (no tight sector fit → acts as
    its own control). CLI: `replay_snapshot --sector` → `ml_replay_sector/`.
  - Measurement (same-as_of 2026-09-15, n=4): sector median full-OOF AUC **0.5484**
    (TCS 0.4567 / RELIANCE 0.5448 / HDFCBANK 0.5520 / INFY 0.5556) vs same-day 5y
    control 0.5596 → **−0.011, no edge → REJECTED**. RELIANCE (mapped to nifty50)
    reproduces its control 0.5448 exactly, confirming the lever is additive-clean.
  - Tests: `test_enrich_fixes.py` 14 passed (sector lever appends 7 `rel_*_cnxit_*`
    columns, default path has zero sector columns; market_features/regime/data_service 54).
- **Task 25 — Enrich-path bug fixes (15-Sep-2026)** — production, no feature/model change:
  - **Options snapshot unwrap**: `get_options_inputs()` returns metrics under a NESTED
    `metrics` key, but `add_options_features()` expects metric keys at the snapshot TOP level
    (plus an `available_at`). `pipeline._build_symbol_input` now unwraps `metrics` and stamps
    `available_at = fetched_at` (PIT = fetch time), so `opt_*` columns actually materialize
    (last row only, no lookahead/backfill) when an options chain succeeds. Previously the
    entire options layer silently never produced a feature column even on a successful fetch.
    Graceful-disable payloads (`is_available: false`) still produce no `opt_*` columns.
  - **build/ copy synced**: the Task-23-flagged latent macro bug
    (`_series_close(...) or mdf["Close"]` Series-bool-eval silently dropping macro series)
    and the stale `get_options_inputs(clean)` (no spot) were BOTH present in the stale
    `backend/build/lib/app/ml/pipeline.py` setuptools output. build/ now matches app/ —
    `_series_close({sid: mdf}, sid)` returns the Close Series without truthiness eval, and
    options is fetched with the live spot.
  - Tests: `tests/test_enrich_fixes.py` gained `test_options_metrics_unwrapped_to_top_level`
    (all 8 `opt_*` columns materialize, PIT lit only on the last row, history stays NaN),
    `test_options_metrics_none_skipped` (graceful disable → zero `opt_*` columns), and
    `test_fundamentals_wired_into_feature_frame` (**verifies the fundamentals path is
    genuinely wired** — `_build_symbol_input` fetch → `build_feature_frame` →
    `add_fundamental_features`, all 8 `fund_*` columns materialize, PIT default = lit only
    on the last row, source metadata `available: true`).
    Relevant suites green (enrich-fixes 12, forecast, options_df, fundamental_features,
    sentiment_features, macro_features, ml_routes, forecast_schema_richness, dataset, models,
    ensemble, calibration, backtest, events, fingerprint). No behavior change to the locked
    `enrich=False` baseline (median full-OOF AUC 0.5703) — the enrich path is opt-in and
    off the hot path.

- **Task 29 — Bear-Market Defense (19-Sep-2026)** — **PRODUCTION** (signal policy change;
  no feature/model/calibration change):
  - `app/ml/signal.py` `decide_signal`: `confidence_floor` default 0.0 → **0.08**
    (prob in (0.46, 0.54) strictly HOLD); new `regime_trend`/`regime_risk` params; bear
    (`regime_trend == -1`) or risk-off (`regime_risk == -1`) → effective BUY threshold
    `max(tb, 0.62)`; returns `effective_threshold_buy`.
  - `app/ml/pipeline.py`: holdout directions are now per-row `decide_signal`
    (confidence_floor=0.08 + regime) in `_model_metrics`, replacing bare
    `np.where(pp >= ml_threshold_buy, 1, 0)`; NOT `hold_rows` for regime (no regime
    columns there — looked up from `out` on holdout dates); NaN probs flat. Live probe
    passes last-row regime so API/Chat signal matches the backtest policy.
  - Effect (locked 10y 09-15 snapshot, same-median check): median full-OOF AUC **0.6179
    unchanged**; cumReturn TCS −21.0→‑11.4%, HDFCBANK −31.6→‑27.7%; settlements 441→402.
    Loss-cut defence — does not create alpha.
  - Tests: signal/forecast/backtest/ensemble/calibration/schema_richness/models EXIT=0.
- **Task 31 — Per-symbol high-conviction BUY thresholds (19-Sep-2026)** — **MEASURED, ADOPTED
  (production)**:
  - `forecast_frame(..., return_oof=True)` now returns per-row OOF detail
    (`result["oof_table"]`: raw/calibrated P(up), per-model probs, label/ret/close,
    regime, holdout flag) — default OFF, production byte-identical, parity with reported
    holdout accuracy verified EXACTLY on all 4 symbols.
  - `threshold_buy_map={SYM: tb}` per-symbol BUY-threshold override, applied to BOTH the
    backtest holdout directions and the live probe (API signal stays in lockstep).
    `replay_snapshot --threshold-map "TCS=0.66,..."` CLI. None → locked 0.60 (unchanged).
  - Measurement (thresholds tuned on older 80% OOF-fit, evaluated on untouched newest-20%
    holdout of the same-day control10y snapshot, as_of 2026-09-18): per-symbol tb*
    (TCS 0.66 / RELIANCE 0.72 / HDFCBANK 0.64 / INFY 0.72) → median BUY hit-rate
    50.9%→**57.1%**, median Σret −140pts→**+7.5pts**; formal ledger cumRet median
    **−20.06%→−4.97%** (TCS −3.56 / RELIANCE −0.17 / HDFCBANK −11.16 / INFY −6.38).
    AUC unchanged. Trade count median 100→32. Regime-conditional thresholds (EXP2)
    showed no improvement → dropped.
  - **ADOPTED 19-Sep-2026** (explicit user mandate + re-check gate passed): thresholds live in
    `settings.ml_per_symbol_buy_thresholds` (`TCS .66 / RELIANCE .72 / HDFCBANK .64 / INFY .72`);
    `forecast_symbol` passes `threshold_buy_map` into `forecast_frame`; symbols absent from the
    map keep the locked 0.60. `_FORECAST_CACHE_VERSION` bumped 1→2 (old 0.60-policy cached
    results become misses). Live smoke: `forecast_symbol('TCS')` → threshold_buy=0.66, P(up)
    0.6103 HOLD, full-OOF AUC 0.5824 (matches replay). Verify stack (all real data, EXIT=0):
    (a) same-code same-window paired replay — median formal-ledger cumRet **−20.06% → −4.97%**
    deterministic, row-level counts == prober; (b) prober determinism 2× + holdout-accuracy
    parity 4/4 EXACT; (c) independently re-fetched 10y frame (`ml_replay_fresh10y/`, 76 feature
    cols differ via yfinance adj-reconstruction, prices/vol identical) → improvement REPRODUCES
    (−20.06% → −4.97%). Residual honest caveat: thresholds are this window's OOF-tuned; a
    genuinely NEW trading window (different as_of) confirmation is pending the next market
    session (Sat/window absent at adoption). Revert = empty the config dict (one line).
  - Tests: test_forecast +3 (parity + threshold map wiring), test_signal, test_backtest.
- **Task 30 — 15-year window A/B (19-Sep-2026)** — research-only, **MEASURED, REJECTED**:
  same-day (as_of 2026-09-18) 15y median full-OOF AUC **0.5060** vs 10y same-as_of
  control **0.6166** → **−0.111** (RELIANCE −0.150, INFY −0.201); longer windows add no
  OOS edge (matches Tasks 23/26/28). Production stays 10y + Task 29 (bear-defense) +
  Task 31 (per-symbol BUY thresholds) policy.
  Artifacts: `backend/ml_replay_15y/`, `backend/ml_replay_control10y/`.

## 3. Golden rules (hard constraints — never break)

1. **No lookahead.** Features at day T use only data ≤ T. Target is `Close[T+h]/Close[T]−1`.
2. **No fabricated data.** Unavailable source → graceful disable + NaN/neutral, never invent.
3. **Honest probabilities.** `P(up)+P(down)=1`, calibrated, 4-dp rounding (`_r4`/`_r4_deep`);
   never claim guaranteed accuracy or fake precision.
4. **Holdout isolation.** Newest 20% of OOF = evaluation only; never fit isotonic/scaler/
   thresholds/feature-selection on it. `final_holdout_never_used_for_fitting = True`.
5. **Measurement honesty.** `n_trades` counts actual settled long positions only; equity =
   real MTM ledger (entry `Close[T]`, exit `Close[T+horizon]`, cost once at open); per-bet
   metrics never borrow equity-curve stats (and vice-versa) — see `metric_groups`.
6. **No hidden leverage.** Per-position notional `C = capital/horizon`; hard guard rejects
   committed > capital; `max_concurrent ≤ horizon`; flat (SELL, `allow_short=False`) = no trade.
7. **Sizing invariant:** `equity(final) = capital + Σpl − costs` and `pl = C·(exit/entry − 1)`.
8. **Floats to 4dp; bools as strings** (`"yes"/"no"`) in `_r4`-intensive dicts (no 1.0/0.0
   lossy coercion).
9. **Don't modify features / models / thresholds / calibration** unless explicitly told.
10. **Never commit** unless explicitly asked.

## 4. Key files

| Area | Files |
|---|---|
| Forecast orchestrator | `backend/app/ml/pipeline.py` (`forecast_frame`, `forecast_symbol`, calibration split, baselines, shuffled control) |
| Backtest / measurement | `backend/app/ml/backtest.py` (`equity_ledger`, `equity_metrics`, `strategy_metrics`, `trade_metrics`, `buy_and_hold_metrics`, `classification_metrics`, `CostModel`, `purged_walk_forward_splits`) |
| Benchmark (measurement-only) | `backend/app/ml/benchmark.py` (multi-stock V1 benchmark: models/ablation/failure/regime diagnostics; output JSON+txt under `backend/ml_benchmark/`, gitignored) |
| Replay / feature-gain diagnostics | `backend/app/ml/replay_snapshot.py` (fixed-snapshot `--save`/`--run` harness over `backend/ml_replay/*_features.pkl`), `backend/app/ml/feature_gain.py` (per-symbol XGBoost gain), `backend/app/ml/tune_rf.py` + `tune_vol_target.py` + `tune_retune_models.py` (research harnesses for the RF / vol-adj-target / RF+logistic-C re-tune sweeps; artifacts `ml_rf_tune/`, `ml_vol_target/`, `ml_retune/`) |
| Models | `backend/app/ml/models.py` |
| Calibration | `backend/app/ml/calibration.py` (Platt / isotonic / ECE) |
| Signal policy | `backend/app/ml/signal.py` (BUY/HOLD/SELL; SELL = flat long-only) |
| Data ingestion | `backend/app/services/data_service.py` |
| Feature engine | `backend/app/ml/features.py`, `alpha.py`, `beta.py`, `regime.py`, `events.py`, `fundamental_features.py`, `sentiment_features.py`, `macro_features.py`, `options_df.py` |
| Explainability | `backend/app/ml/explain.py` (`_auc` = average-rank ROC-AUC on prob vs binary target) |
| Monitoring | `backend/app/ml/monitor.py` |
| Versioning | `backend/app/ml/versions.py` |
| Routes / schemas | `backend/app/routes/forecast.py`, `routes/ml.py`, `schemas/forecast.py` |
| Chat / AI | `backend/app/services/chat_service.py` |
| Frontend forecast | `frontend/src/components/stock/ForecastCard.tsx`, `hooks/useForecast.ts`, `types/forecast.ts` |

## 5. Commands

```bash
# Backend tests (from backend/)
cd backend
python -m pytest tests/          # full suite -> 519 passed (10-Sep-2026); targeted suites below for the new tests
python -m pytest tests/test_backtest.py -q
python -m pytest tests/test_forecast.py -q
python -m pytest tests/test_calibration.py -q
python -m pytest tests/test_benchmark.py -q   # Task 11 benchmark tests (18)
python -m pytest tests/test_chat_service.py -q  # chat intent/routing/filter/OOS/smalltalk/disclaimer (65)

# Fixed-snapshot replay + feature-gain diagnostic (from backend/)
python -m app.ml.replay_snapshot --save --symbols TCS,RELIANCE,HDFCBANK,INFY   # rebuild snapshot (features change wale steps)
python -m app.ml.replay_snapshot --run --symbols TCS,RELIANCE,HDFCBANK,INFY    # median full-OOF AUC (locked 5y baseline 0.5703; production now 10y)
python -m app.ml.replay_snapshot --save --period 10y --label ten_y  --symbols TCS,RELIANCE,HDFCBANK,INFY   # 10y snapshot (Task 27, as_of 2026-09-15)
python -m app.ml.replay_snapshot --run  --period 10y --label ten_y  --symbols TCS,RELIANCE,HDFCBANK,INFY    # 10y median full-OOF AUC 0.6179
python -m app.ml.replay_snapshot --save --sector --label sector   --symbols TCS,RELIANCE,HDFCBANK,INFY      # sector-relative features (Task 28, REJECTED 0.5484)
python -m app.ml.replay_snapshot --save --fundamentals-pit       --symbols TCS,RELIANCE,HDFCBANK,INFY      # PIT fundamentals (Task 26, REJECTED 0.5701)
python -m app.ml.feature_gain --symbols TCS,RELIANCE,HDFCBANK,INFY             # per-symbol XGBoost gain (top-15 + NEW-feature share)
python -m app.ml.tune_rf --symbols TCS,RELIANCE,HDFCBANK,INFY                  # RF hyperparameter search (400/8/30 adopted) -> ml_rf_tune/
python -m app.ml.tune_vol_target --symbols TCS,RELIANCE,HDFCBANK,INFY          # vol-adj train-target k sweep (0.5 adopted) -> ml_vol_target/
python -m app.ml.tune_retune_models --symbols TCS,RELIANCE,HDFCBANK,INFY      # RF+logistic-C re-tune on NEW train_up_20d target (12-Sep: no improvement, keep current) -> ml_retune/

# Run backend app
uvicorn app.main:app            # from backend/, http://localhost:8000

# Frontend
cd frontend
npm run dev                     # http://localhost:3000
npm run build                   # next build, EXIT=0 (verified)
npx tsc --noEmit                # typecheck

# Live forecast (from backend/)
python -c "from app.ml.pipeline import forecast_symbol; import json; print(json.dumps(forecast_symbol('RELIANCE'), indent=2))"
```

**Metrics of a correct backtest row** (every `backtest[]` entry): model, calibrated
(raw/isotonic/n-a), n_trades, win_rate, avg_return, profit_factor (per-bet) +
cum_return/max_dd/sharpe/sortino/calmar/annualized_* /n_days/max_concurrent/no_leverage_ok
(equity-curve) + accuracy/precision/recall/f1/confusion/roc_auc/calibration_brier
(classification) + `metric_groups` provenance.

## 6. Conventions

- **Windows / PowerShell:** no `&&`; chain with `;` or `if ($?)`. No `tail`.
- Floats rounded to 4dp in outputs; `NaN → None` for JSON.
- Every change: run tests → update `handoff.md` → stop for the user.
- The forecast cache `backend/ml_cache/forecasts/{SYMBOL}.json` is keyed by last-close
  date (as_of) **and input data_fingerprint** — a cached result is served only when its
  stored SHA-256 fingerprint matches the current upstream snapshot (legacy records are
  misses on fingerprint lookups). `ml_cache/` is gitignored.
- Known real-data wrinkle: the TCS ensemble backtest is **snapshot-specific** — many
  calibrated probs sit at the 0.50 boundary, so same-code runs over different
  upstream-data pulls can flip borderline signals (observed 67↔88 longs). Since the
  reproducibility layer, each run persists its input fingerprint, so such differences
  are now attributable to distinct input snapshots, not code. Always state `as_of` +
  `generated_at` + `data_fingerprint` with reported numbers.

## 7. TODO backlog (from handoff.md)

- SSE streaming for `/api/chat`
- Chat history persistence
- More chat intents (compare stocks, top gainers/losers)
- Quick-prompt chips; model-selector setting
- **GDELT sentiment A/B resume (Task 24, BLOCKED — NOT measured):** GDELT temp-banned this
  network on 13-Sep-2026 (everything 429, ~66 min bounded wait). Research modules ready
  (`gdelt_sentiment.py`, `gdelt_ab.py`); resume when GDELT answers 200 from this network:
  `python -m app.ml.gdelt_ab --symbols TCS,RELIANCE,HDFCBANK,INFY`.
  Accept only if variant median full-OOF AUC strictly > 0.5703, else REVERT + mark
  sentiment "tested, no help" (public-data space then exhausted).
- **Task 31 Monday confirmation (queued — next market session):** per-symbol BUY thresholds
  (TCS 0.66 / RELIANCE 0.72 / HDFCBANK 0.64 / INFY 0.72) ADOPTED 19-Sep-2026 in production.
  Final gate pending a genuinely NEW trading window:
  `replay_snapshot --save --period 10y --label confirm` + `--run` without/with
  `--threshold-map "TCS=0.66,RELIANCE=0.72,HDFCBANK=0.64,INFY=0.72"`.
  Pass = median cumRet −20.06% → −4.97% reproduced on the new window; then verify
  `forecast_symbol` live `threshold_buy` reads the new per-symbol values (cache v2).
  Revert = empty `settings.ml_per_symbol_buy_thresholds`.
- **Cross-symbol generalization (LONG-TERM research, do NOT force in this session):**
  the locked model is strong on the core 4 large-caps (0.5703) but ITC/SBIN sit at an
  HONEST chance-level ceiling this window — oracle best-of test proves NEW3 > 0.52 is
  unreachable now (SBIN 0.4935 / ITC 0.4597 binding). Needs a different time window /
  more data / richer features, NOT more pooling/blending/variant tuning.
  Reference: `docs/model-v2-pooled-model-results.md` (Tasks 21–22).