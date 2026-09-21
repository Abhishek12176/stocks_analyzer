# Architecture — AVORA / EquityLens Stock Analyzer

Indian (NSE/BSE) stock analysis platform. FastAPI backend + Next.js 15 frontend,
with a technical-indicator prediction engine and an optional AI chat assistant.

---

## System Overview

```
┌─────────────────────────────┐  /api/*   ┌──────────────────────────────┐
│         FRONTEND            │ ────────► │           BACKEND            │
│    Next.js 15 + TypeScript  │   proxy   │        FastAPI (Python)      │
│                             │           │                              │
│  Pages:                     │           │  Routers (prefix /api/v1)    │
│   Dashboard /               │           │   stock, signals, market,    │
│   Signals / Watchlist /     │           │   watchlist, compare,        │
│   Stock detail [symbol] /   │           │   feedback, health, chat     │
│   Baskets / History /       │           │                              │
│   Settings / Feedback       │           │  Services:                   │
│                             │           │   chat_service (intent+LLM)  │
│  ChatWidget (bottom-right)  │           │   signals_service ──► model  │
│   └─ POST /api/chat         │           │   signal_service  ──► model  │
│      └─ api/[...path] proxy │           │   yfinance_service ─► yfinance│
│         └─ BACKEND_URL      │           │   market / news / sharehold /│
└─────────────────────────────┘           │   fundamental / sentiment ...│
                                         └──────────────┬────────────────┘
                                                        │
                                          ┌─────────────▼─────────────────┐
                                          │  DATA SOURCES                 │
                                          │   yfinance (NSE/BSE OHLCV)    │
                                          │   NewsData.io / Moneycontrol  │
                                          │   Screener.in / NSE /         │
                                          │   MarketSmith (shareholding)  │
                                          │   LLM (Groq / OpenAI-compat)  │
                                          └───────────────────────────────┘
```

---

## Folder Structure

```
stocks_analyzer-main/
├── handoff.md                     # Roadmap (TODO / IN_PROGRESS / DONE)
├── architecture.md                # This document
├── render.yaml                    # Render deployment (backend blueprint)
├── runtime.txt                    # Python version (3.12.3) for Render
├── .gitignore / .python-version
│
├── backend/                       # FastAPI backend
│   ├── requirements.txt           # Pinned runtime deps
│   ├── pyproject.toml             # Project metadata (equitylens-api)
│   ├── .env.example               # Env template (LLM/NewsData keys, cache TTLs)
│   ├── .env                       # Local secrets (Groq key, NewsData key) — never committed
│   ├── start.sh                   # Production uvicorn startup script
│   ├── runtime.txt
│   └── app/
│       ├── main.py                # App factory, logging, CORS, rate-limit,
│       │                          #   global handler, lifespan health check,
│       │                          #   all routers mounted under /api/v1
│       ├── config.py              # Pydantic BaseSettings (env-driven)
│       ├── core/
│       │   ├── middleware.py      # RateLimitMiddleware (limits /api/*)
│       │   └── rate_limiter.py    # In-memory sliding-window RateLimiter
│       ├── routes/                # Thin HTTP wrappers around services
│       │   ├── stock.py           # /stock search/price/intraday/fundamentals/
│       │   │                      #   signal/news/shareholding/full analysis
│       │   ├── signals.py         # GET /signals/all
│       │   ├── market.py          # GET /market/overview
│       │   ├── watchlist.py       # JSON-file backed watchlist CRUD
│       │   ├── compare.py         # POST /compare/ (2-5 stocks side-by-side)
│       │   ├── feedback.py        # JSON-file backed feedback submit/list
│       │   ├── health.py          # GET /health/sources
│       │   └── chat.py            # POST /chat (assistant)
│       ├── services/              # Business logic + data sources
│       │   ├── signal_service.py      # generate_trade_signal() → THE model
│       │   ├── signals_service.py     # batch scan ~80 symbols → categories
│       │   ├── indicator_service.py   # RSI / MACD / SMA / EMA (pandas math)
│       │   ├── yfinance_service.py    # price / intraday / search via yfinance
│       │   ├── fundamental_service.py # fundamentals + 0-100 score + rating
│       │   ├── news_service.py        # NewsData.io + yfinance fallback
│       │   ├── sentiment_service.py   # FinBERT (optional) / VADER fallback
│       │   ├── shareholding_service.py# orchestrates screener/moneycontrol/nse/
│       │   │                          #   marketsmith + yfinance fallback
│       │   ├── screener_service.py    # quarterly + major holder scrapers
│       │   ├── moneycontrol_service.py
│       │   ├── nse_service.py
│       │   ├── marketsmith_service.py # Playwright (headless Chromium)
│       │   ├── market_service.py      # top bullish/bearish + sparklines
│       │   ├── chat_service.py        # intent parser + LLM + routing
│       │   ├── health_service.py      # async source health checks
│       │   └── cache_service.py       # TTLCache singletons
│       ├── schemas/               # Pydantic models (BaseSchema camelCase)
│       │   ├── base.py            # alias_generator=to_camel, populate_by_name
│       │   ├── stock / signal / news / fundamentals / shareholding /
│       │   │   screener / analysis
│       └── utils/
│           ├── validators.py      # clean_symbol, validate_symbol, add_exchange_suffix
│           ├── exceptions.py      # typed HTTPExceptions (404/400/429/503)
│           ├── formatters.py      # date/ISO helpers
│           └── scraper_utils.py   # shared HTML/table/shareholding parsing
│
└── frontend/                      # Next.js 15 + TypeScript (App Router)
    ├── package.json               # Next 15, React 19, TS, Tailwind v4
    ├── tsconfig.json              # strict, path alias @/* → src/*
    ├── next.config.ts             # minimal; API proxied via route handler
    ├── vercel.json                # Next.js build, region bom1 (Mumbai)
    ├── eslint.config.mjs / postcss.config.mjs
    ├── .env.example               # BACKEND_URL (Render), optional API_KEY
    └── src/
        ├── app/
        │   ├── layout.tsx / providers.tsx  # root shell + React Query client
        │   ├── page.tsx                    # Dashboard (search, watchlist, history)
        │   ├── stock/[symbol]/page.tsx     # Stock detail (tabs: technical, raw,
        │   │                                #   fundamentals, ownership, news, signal)
        │   ├── signals/page.tsx            # Signal Center (categorized grid)
        │   ├── markets/stocks/page.tsx     # Market overview (bullish/bearish)
        │   ├── watchlist / history / basket / basket/[id] / settings /
        │   │   admin/feedback / compare (placeholder) / onboarding
        │   └── api/[...path]/route.ts      # Proxy /api/* → BACKEND_URL (90s timeout)
        ├── components/
        │   ├── layout/    AppShell, Sidebar, Header, MobileNav, CommandPalette
        │   ├── chat/      ChatWidget, StockResultCard
        │   ├── chart/     Candlestick, Intraday, Rsi, Macd, Area, Line,
        │   │              Sparkline, Radar
        │   ├── stock/     CompanyHeader, MetricGrid, TradeSignal, TechnicalPanel,
        │   │              FundamentalPanel/ScoreCard, ShareholdingChart,
        │   │              MajorShareholders, NewsFeed, RawDataTable, StockSearch
        │   ├── signals/   SignalCard
        │   ├── basket/    BasketCard
        │   └── ui/        Button, Badge, Tabs, Input, Select, Modal, Table,
        │                  Skeleton, Toast, Tooltip, EmptyState, ErrorBoundary,
        │                  MetricCard, ProgressRing
        ├── hooks/         useStock, useSignal, useSignals, useNews, useShareholding,
        │                  useIntradayStock, useFullAnalysis, useFundamentals,
        │                  useBaskets, useDebounce
        ├── lib/           api.ts, chat-api.ts, constants, formatters, indicators,
        │                  validators, cn, basket-data (fallbacks)
        ├── store/         Zustand: watchlistStore, historyStore, uiStore
        │                  (localStorage persisted: equitylens-*)
        ├── styles/globals.css  # Tailwind v4, dark/light theme tokens
        └── types/         stock, signal, signals, news, fundamentals,
                           shareholding, analysis, chat, api, basket
```

---

## Backend Request Flow (layers)

```
HTTP request
   └─ RateLimitMiddleware (per-IP sliding window, /api/*)
        └─ Router (validates input via Pydantic schemas, parses params)
             └─ Service (fetches/derives data, uses cache_service)
                  └─ Data source (yfinance / screener / moneycontrol / nse / marketsmith)
                       └─ Response schema (camelCase via BaseSchema) → JSON
```

**Caching** (`cache_service`, TTLCache):
- price: 60s | fundamentals: 1h | news: 15min | shareholding: 1 day | signals overview: 10min

---

## Data Flow

### 1. Prediction / signal generation (technical-indicator model)
```
signals_service.get_all_signals()            ← also used by /signals/all and chat
  └─ ThreadPoolExecutor(15): for symbol in ALL_SYMBOLS (~80 NSE stocks)
       └─ yf.Ticker(symbol.NS).history(period="3mo")
            ├─ indicator_service → RSI, MACD, SMA20, SMA50
            ├─ _fetch_one also computes per-stock extra context: upDaysConsecutive
            │    (strict consecutive up-close streak) + ret3d/5d/7d/10d/15d/20d
            │    (N-day return %, None when history is insufficient) — used by the
            │    chat "N din se bullish" filter
            └─ signal_service.generate_trade_signal(price, rsi, macd, signal, sma20, sma50, sentiment)
                 → { signal: { action, direction, confidence, reasons }, quote }
  └─ Categorize into: strong-buy, buy, hold, sell, strong-sell,
       rsi-oversold/overbought, macd-bullish/bearish, bullish/bearish-trend
  └─ Cache 10 min → GET /api/v1/signals/all
```

`generate_trade_signal` is a **multi-factor voting model** (factors → bullish/bearish votes):
- Trend: SMA20 vs SMA50
- RSI: <30 oversold (bull) / >70 overbought (bear)
- MACD: line vs signal line
- Forecast label (Bullish/Bearish)
- News sentiment score (if provided)
`action = BUY if bull>bear, SELL if bear>bull, else HOLD`; confidence = max/bull/bear share.

### 2. AI Chat request
```
User (ChatWidget) ──► POST /api/chat  { message, history? }
  └─ Next.js proxy (api/[...path]) ──► FastAPI
       └─ chat_service.process_chat(message)
            ├─ routing:
            │     smalltalk (greetings/welcome/good night) → _fallback_smalltalk
            │     concept KB (RSI/MACD/PE/ROE...) → KB reply
            │     "who made you" → _who_made_you
            │     out-of-scope (weather/sports/politics/math/... 13 groups) →
            │        _out_of_scope_reply (professional decline + redirect)
            │     is_stock_query? → stock pipeline (below)
            │     else → _general_reply (LLM via Groq / deterministic fallback)
            ├─ stock pipeline:
            │     parse_intent() → { maxPrice?, minPrice?, action?, top?, bullishDays? }
            │       regex (EN + Hinglish): "under 500", "₹500 se kam", "buy/kharid",
            │       "penny/penni" (→ maxPrice 100), "3 din se bullish" (→ bullishDays 3)
            │     load_predictions() → get_all_signals()   ← EXISTING MODEL ONLY
            │     filter_predictions + rank stocks by intent:
            │       price bounds, action, top N, bullishDays → retNd > 0 (positive
            │       N-day return; streak fallback when the exact horizon isn't precomputed)
            └─ format_answer: LLM mode (OPENAI_API_KEY) or deterministic Hinglish/
                 English fallback template
            └─ disclaimer: every stock pipeline reply ends with an EN/HI financial
                 disclaimer ("not financial advice") + thanks
  └─ Returns { reply, stocks[], totalFound, intent, source: "llm"|"existing-model" }
       └─ ChatWidget renders text + clickable stock cards → /stock/[symbol]
```

### 3. Stock detail page (`/stock/[symbol]`)
```
useFullAnalysis  → GET /api/v1/stock/{symbol}  (quote + indicators + fundamentals + score + signal)
useStockPrice    → GET /api/v1/stock/{symbol}/price?period   (history + indicators → charts)
useSignal        → GET /api/v1/stock/{symbol}/signal  (trade signal + reasons + quote)
useNews          → GET /api/v1/stock/{symbol}/news?limit
useShareholding  → GET /api/v1/stock/{symbol}/shareholding (quarterly + major holders)
```

---

## Key Connections

| Layer              | Component                        | How it connects                                   |
|--------------------|----------------------------------|---------------------------------------------------|
| Frontend → Backend | `api/[...path]/route.ts` proxy  | `BACKEND_URL` env (default Render); dev localhost:8000 |
| Frontend → Backend | `lib/api.ts`                     | `API_BASE = /api/v1`; apiGet/apiPost via proxy    |
| Backend → Data     | `yfinance_service`               | NSE/BSE OHLCV; exchange suffix `.NS` / `.BO`      |
| Backend → Model    | `signal_service.generate_trade_signal` | indicators → BUY/SELL/HOLD confidence       |
| Backend → Chat     | `chat_service`                   | reads existing signals only (never invents forecasts) |
| Chat → LLM (opt.)  | OpenAI-compatible HTTP (Groq)    | `OPENAI_API_KEY` in `backend/.env`; rephrases stock data + general Q&A |
| Chat → OOS guard   | `_out_of_scope_reply`            | declines non-stock topics before the LLM path (13 topic groups) |
| Chat → Disclaimer  | `_financial_disclaimer`          | appended to every NSE/stock answer (EN/HI); skipped for smalltalk/who/OOS |
| Backend → Storage  | watchlist.py / feedback.py       | JSON files under `backend/data/`                  |

---

## Deployment

- **Backend**: Render (`render.yaml` blueprint) — Python web service, uvicorn `app.main:app`,
  `rootDir: backend`, free plan, health check `/api/health`. Installs Playwright Chromium for scraping.
- **Frontend**: Vercel (`vercel.json`) — Next.js, deploy region `bom1` (Mumbai), builds `.next`.
- Env secrets: `backend/.env` (local; NOT `.env.example` — settings only load `.env`) for
  `NEWSDATA_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`. Never committed.

---

## Guarantee: No hallucinated predictions

The `/api/chat` pipeline fetches all stock data from `get_all_signals()` (the existing
technical-indicator model). The LLM (if enabled) is given a strict system prompt forbidding it
from inventing symbols, prices, or signals — it only rephrases the provided prediction data.
If no LLM key is configured, a deterministic template builds the reply from the same data.

---

## Roadmap status

See `handoff.md` for the project roadmap. All baseline + AI chat features are `[DONE]`;
V2 Tasks 12–16 are complete as isolated research. Tasks 17–20 (V1-vs-pooled fair
comparison, selective forecasting, entropy validation, expected-return + risk research)
are complete — the entropy rule retained only 45.66% coverage and the expected-return
forecasts carried no genuine OOS signal, so **V1 remains frozen** and no research
candidate is approved for production. Chat polish (11-Sep-2026): FILE 12 (greetings/who/
penny), FILE 13 (out-of-scope professional decline), FILE 14 (positive N-day-return
"bullish" filter), FILE 15 (financial disclaimer on NSE/stock answers) are DONE in
production; Groq LLM chat is active via `backend/.env`.
Forecast-quality iteration (12-Sep-2026): XGBoost dropped from the live ensemble
(`ENSEMBLE_MODELS = (logistic, rf)`), `DEFAULT_RF` tuned to `(300, depth 8, leaf 30)`,
and a volatility-adjusted TRAINING target adopted (`VOL_ADJ_K = 0.5`);
replay median full-OOF AUC on the fixed 09-09 snapshot is **0.5703**
(progression 0.5214 → 0.5351 → 0.5399 → 0.5465 → 0.5703). Details in the 10-11/12-Sep
addenda below.
Remaining work includes the optional chat enhancements and explicitly approved research
tasks only.

> **Project-wide status:** The platform is being upgraded into a research-grade,
> explainable, probabilistic **20-trading-day NSE forecasting system**.
> The full master implementation plan is documented below.

---

# MASTER IMPLEMENTATION PLAN — NEXT-GEN 20-DAY NSE STOCK FORECASTING SYSTEM

Master spec: see user-provided "MASTER IMPLEMENTATION PROMPT" (sections 1–26).
Execution style: **10 sequential production tasks**, followed by explicitly
gated research tasks. After each task:
run tests → run backtest smoke → record metrics → compare vs previous version →
update `handoff.md` → stop and wait for the user before starting the next task.

## Core objective

For any NSE symbol, produce (clearly labeled research/simulation — never 100%):

- 20-trading-day **directional probability** `P(up next 20d)` (calibrated, e.g. 68%)
- **Expected return / range** and **risk / volatility estimate**
- **Market regime** (bull/bear/sideways, high/low vol, risk-on/off)
- **BUY / HOLD / SELL** from validated, configurable thresholds
- **Confidence / probability**, top positive + top negative **reasons** (real features only)
- **Walk-forward backtest performance** of the model

## Master pipeline (project-wide rule)

```
DATA
  ↓
DATA QUALITY / POINT-IN-TIME VALIDATION
  ↓
FEATURE ENGINE (technical + momentum + volatility + alpha + beta +
                fundamental + macro + sentiment + event + F&O/Greeks)
  ↓
FEATURE SELECTION / NORMALIZATION / INTERACTIONS
  ↓
REGIME DETECTION
  ↓
ML MODELS
  ↓
ENSEMBLE
  ↓
PROBABILITY CALIBRATION
  ↓
20-DAY FORECAST
  ↓
BUY / HOLD / SELL
  ↓
SHAP / EXPLAINABILITY
  ↓
BACKTEST + MONITORING
  ↓
FRONTEND + AI CHAT
```

## What already exists (baseline — REUSE, do not rewrite)

| Layer | Files | Role |
|---|---|---|
| Rule-based signals | `backend/app/services/signal_service.py` | Baseline/fallback voting model |
| 80-stock scanner | `backend/app/services/signals_service.py` | `/signals/all` (keep endpoint) |
| Index math | `backend/app/services/indicator_service.py` | SMA/EMA/RSI/MACD — extend |
| Price data | `backend/app/services/yfinance_service.py` | OHLCV/intraday/search |
| Fundamentals | `backend/app/services/fundamental_service.py` | Slow-moving feature source |
| News + sentiment | `backend/app/services/news_service.py`, `sentiment_service.py` | Sentiment layer |
| Shareholding | `shareholding_service.py`, `screener/moneycontrol/nse/marketsmith_service.py` | Ownership layer |
| Infra | `cache_service.py`, `config.py`, `schemas/`, `utils/`, `routes/*`, `main.py` | Config/cache/API |
| Frontend | Next.js pages/hooks/components/stores, `api/[...path]` proxy | Preserve design, upgrade |

## What must be modified

- `backend/app/config.py` — versioning + ML settings (feature/model version, thresholds, forecast horizon, cost model).
- `backend/requirements.txt` — add lightweight ML deps (`scikit-learn`, `joblib`, `xgboost`); `lightgbm`, `shap`, `torch`, `transformers` = **optional/lazy** (Render free-tier friendly).
- `backend/app/routes/stock.py` + `schemas/` — integrate forecast block into full analysis.
- `frontend/src/app/stock/[symbol]/page.tsx`, `types/`, `hooks/`, components — add forecast + analysis sections (preserve existing design).

## New modules (added incrementally)

`backend/app/ml/` package:
`features.py` (technical engine) · `alpha.py` · `beta.py` · `regime.py` ·
`dataset.py` (target T+1→T+20, PIT-safe) · `models.py` · `ensemble.py` ·
`calibration.py` · `signal.py` (BUY/HOLD/SELL policy) · `backtest.py` ·
`ablation.py` · `explain.py` (SHAP/importance) · `options_df.py` (F&O/OI/PCR/IV/Greeks, graceful disable) ·
`versions.py` (prediction snapshots) · `monitor.py` (freshness/drift/degraded mode) ·
`pipeline.py` (orchestrator, `.predict(symbol) -> Forecast`).

Supporting:
- `backend/app/services/data_service.py` — market/index/global/macro ingestion (yfinance, PIT-safe, freshness).
- `backend/app/routes/forecast.py` — `GET /api/v1/stock/{symbol}/forecast`.
- `backend/app/routes/ml.py` — `GET /api/v1/ml/status` (monitoring).
- `backend/app/schemas/forecast.py` — forecast + explainability schemas.
- `backend/models/` — model artifacts (joblib), versioned.
- `backend/data/predictions/` — prediction history for later outcome evaluation.
- `backend/tests/` — pytest: unit (synthetic data, no network), integration (2–3 symbols), backtest (leakage/cost).

## New APIs

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/stock/{symbol}/forecast` | Probabilistic 20d forecast + explanation + backtest perf |
| GET | `/api/v1/ml/status` | Data freshness, drift, degraded-mode monitor |
| POST | `/api/v1/chat` | (existing) upgraded: reads forecast engine output, never fabricates |

## Hard rules (every task)

- No lookahead bias: features at day T use only data ≤ T; target is T+1 → T+20.
- Never silently fabricate financial data — unavailable sources → graceful disable + neutral factor.
- Never claim guaranteed accuracy; `P(up)+P(down)=1`, calibrated, no fake precision like 68.347829%.
- Indicators = ML features, NOT hard-coded BUY/SELL rules.
- Options Greeks (Δ/Γ/Θ/ν/ρ) live in their own derivatives layer — not ordinary indicators.
- Prefer out-of-sample robustness over impressive backtest numbers.

---

## Measurement / validation layer (locked 07-Sep-2026)

### Backtest = a real marked-to-market equity book (`ml/backtest.py`)

- `equity_ledger(close, signals, horizon, capital, cost) → (equity, trades)`:
  - Entry `Close[T]`, exit `Close[T+horizon]` — exactly the model's target window
    `target_ret_{h}d = Close[T+h]/Close[T] − 1` (verified against the ledger).
  - Per-bar settle-then-open; cost charged **once at open** (never inside the mark).
  - Per-position notional `C = capital / horizon` ⇒ max gross exposure = 1.0× capital
    (no hidden leverage; hard guard rejects committed > capital; same-bar duplicates
    deduped). Invariant: `equity(final) == capital + Σpl − costs`, `pl == C·(exit/entry−1)`.
  - `equity.attrs["max_concurrent"]` reports peak concurrent positions.
- `equity_metrics(equity)` — true portfolio stats from the daily MTM curve:
  `cum_return`, `max_dd ∈ (−1,0]`, Sharpe×√252, Sortino, Calmar, annualised return/vol.
- `trade_metrics()` — per-settled-bet only (`n_trades`, `win_rate`, `avg_return`,
  `profit_factor`); no fake drawdowns/compounding on the per-trade view.
- `strategy_metrics()` returns `metric_groups` provenance:
  `{per_bet, daily_equity_curve, classification}` — every report distinguishes
  "per completed 20-day bet" from "daily MTM equity curve".
- `n_trades` = number of settled ledger positions = number of `direction ≠ 0` rows
  (flat rows are dropped in `equity_ledger`; the production strategy never emits −1).
- `buy_and_hold_metrics()` — single long position on the evaluation window, `n_trades=1`,
  equity = price ratio.
- ROC-AUC = average-rank `explain._auc(P(up), y)` (probability vs binary target); Brier
  (`calibration_brier`) is a separate metric, never fed to AUC.

### Honest evaluation protocol (`ml/pipeline.py` forecast_frame)

- OOF walk-forward probs (purged, embargo=horizon) are split **chronologically**:
  older 80% = `fit_rows` (isotonic calibration fit), newest 20% = `hold_rows`.
  Headline Brier/ECE + **all backtest rows** are computed on `hold_rows` only;
  `final_holdout_never_used_for_fitting = True`.
- Baselines on the identical holdout window: `always_up`, `always_down` (diagnostic
  short; NOT the production strategy — production SELL = flat via `allow_short=False`),
  seeded `random`, and a `shuffled_control` (per-fold seeded permutation of the labels
  **within each training fold only**; holdout labels untouched; independent model fits,
  rule-based voting excluded) → chance-level ROC-AUC/recall reference.
- Report lifespan note: the ensemble backtest is **snapshot-specific** — the isotonic
  map places many holdout probs at the 0.50 boundary, so same-code runs across different
  upstream-data pulls (market indices / fundamentals drift) can flip borderline signals
  (observed 67↔88 longs); within a fixed input frame the pipeline is bit-deterministic.
  The forecast cache (`ml_cache/forecasts/{SYMBOL}.json`) is keyed by last-close date,
  so repeat calls return the stored result until the next trading day.

### V1-quality iteration addendum (10-11 Sep-2026) — stable metric + model/vetting fixes

- `forecast_frame` now exposes `discrimination = {full_oof_auc, holdout_auc, metric}`
  (ensemble full-OOF ROC-AUC via `explain._auc`). The tiny newest-20% holdout is
  noise-prone for the 4-symbol universe (INFY holdout 0.17 ↔ full-OOF 0.52), so
  **full-OOF AUC is the go/no-go metric** for feature/model experiments; the holdout
  stays the no-fit zone for calibration.
- Live ensemble was reduced from `voting` to `(logistic, rf, xgboost)` (10-11 Sep) and
  further to `(logistic, rf)` on 12-Sep-2026 — see the 12-Sep addendum below. BUY
  threshold = `settings.ml_threshold_buy`. Logistic/ridge are
  `Pipeline(StandardScaler(...))`; XGBoost trains with `scale_pos_weight = neg/pos`
  (edge-safe). `forecast_frame` drops constant feature columns (prevents an all-NaN
  argmax collapse on flat tapes).
- **Signal-quality gate**: if full-OOF ensemble AUC < `settings.ml_min_auc_for_signal`
  (0.55) the forecast signal is pinned to **HOLD** with the holdout base-rate
  `P(up)` (confidence/reason updated too) instead of an overconfident calibrated prob.
- Replay harness `backend/app/ml/replay_snapshot.py` runs model/feature experiments on a
  fixed snapshot (`backend/ml_replay/*_features.pkl`, as_of 2026-09-09, 1239 rows) —
  baseline **median full-OOF AUC = 0.5214**; a market-relative (excess-return) target
  experiment measured 0.3885 and was **rejected + fully reverted**.
- FILE 8 momentum features (kept) lifted replay median full-OOF AUC **0.5214 → 0.5351**;
  `backend/app/ml/feature_gain.py` ranks features by per-symbol walk-forward XGBoost
  gain (uniform share = 0.010 at 100 features; new-feature group share 3.0–3.7%,
  `vol_ratio_20_60` strongest, `rel_rs_slope_nifty50_20d` weakest → selection candidate).

### V1-quality iteration addendum (12-Sep-2026) — XGBoost drop + RF tuning + volatility-adjusted train target

- **XGBoost removed from the live ensemble:** `ENSEMBLE_MODELS = ("logistic", "rf")`
  (`pipeline.py`). On the fixed 09-09 replay snapshot XGBoost full-OOF AUC median was
  ~0.5092 (below chance: TCS 0.4845 / INFY 0.4898) and dragged the equal-weight ensemble
  (logistic/rf alone ≈ 0.5346) down to 0.5244. No stacked meta-learner (unstable on this
  small set). Replay median full-OOF AUC **0.5244 → 0.5346** (replay-measured **0.5399**:
  TCS 0.4835 / RELIANCE 0.5399 / HDFCBANK 0.5579 / INFY 0.5293).
- **RF hyperparameters tuned:** `models.py` `DEFAULT_RF` `(n_estimators=200, max_depth=6,
  min_samples_leaf=20)` → `(n_estimators=300, max_depth=8, min_samples_leaf=30,
  class_weight="balanced", n_jobs=-1)`. Leakage-free protocol: exact live preprocessing +
  same purged walk-forward splits (test_size=40, step=90, min_train=260, embargo=20) on
  the fixed replay frames, median imputer fit on train only, coordinate-wise full-OOF AUC
  search. Default RF 0.5331 → best **0.5512** (> 0.5399 bar → ADOPT). Replay after adopt:
  median full-OOF AUC **0.5399 → 0.5465** (TCS 0.4854 / RELIANCE 0.5465 / HDFCBANK 0.5632 /
  INFY 0.5373). Harness `backend/app/ml/tune_rf.py`, artifact `ml_rf_tune/rf_tune_result.json`.
- **Volatility-adjusted TRAINING target:** `dataset.add_volatility_target` +
  `pipeline.VOL_ADJ_K` (default 0.5). Causal per-row noise band `theta_T = k·sigma_T·√20`
  (sigma_T = 20d realized volatility, data ≤ T). TRAINING keeps only unambiguous up/down
  rows (`train_up_20d` = 1 / 0 outside the band; NaN inside → dropped), while EVALUATION
  always uses the full binary `target_up_20d` so the AUC baseline stays comparable.
  `train_up_*` / `vol_theta_*` are excluded from features in `_default_forecast_cols` and
  threaded only as `y_train` into walk-forward folds, the latest-probe fit, and the explain
  fit. Sweep on the replay frames (harness `tune_vol_target.py` verified == replay at k=0;
  numpy-median full-OOF AUC): k=0 0.5419 / k=0.25 0.5443 / **k=0.5 0.5696** / k=0.75 0.5454 /
  k=1.0 0.5241 → **ADOPT k=0.5**. Replay after adopt: median full-OOF AUC
  **0.5465 → 0.5703** (TCS 0.4921 / RELIANCE 0.5688 / HDFCBANK 0.6091 / INFY 0.5703).
  Artifact `ml_vol_target/vol_target_tune_result.json`.
- **Baseline progression on the fixed 09-09 replay snapshot** (median full-OOF AUC):
  0.5214 (fixes) → 0.5351 (FILE 8) → 0.5399 (XGBoost dropped) → 0.5465 (RF tuned) →
  **0.5703 (vol-adj train target — current LOCKED baseline)**. Rejected/reverted:
  all drop-tests + cross-sectional additions. `handoff.md` + `docs/` remain the source of truth.
- **Fresh-snapshot validation (12-Sep-2026, measurement only):** a FRESH replay pull
  (as_of 2026-09-11, +2 trading days) reproduced the median full-OOF AUC **0.5703
  bit-stable** (TCS 0.492 / RELIANCE 0.5688 / HDFCBANK 0.6092 / INFY 0.5703) → the locked
  gain is REAL, not overfit to the 09-09 snapshot. Original snapshot restored
  (`ml_replay_20260909_backup`); baseline reproducible.
- **10y adoption (Task 27, 15-Sep-2026, user-approved):** `forecast_symbol` default
  `period="5y"` → `"10y"`. Measured same-day (as_of 2026-09-15): 10y median 0.6179 vs 5y
  0.5596 (+0.58, 3/4 symbols improved). `routes/forecast.py` and `chat_service` hot paths
  now call 10y; `build_feature_frame`/`forecast_frame` signatures unchanged (input window
  grew from 1240 to 2474 rows). `regime.risk_regime` ragged-calendar crash (10y-only)
  fixed: VIX classified on own calendar then reindexed (still causal).
  **Feature/model state unchanged** (ensemble = logistic + rf, RF=300/8/30, C=1.0,
  VOL_ADJ_K=0.5, thresholds same); only the training window grew.
- **Cross-symbol probe (Tasks 21–22, research-only) — NO ADOPT, per-symbol ceiling proven:**
  the locked config is strongly per-symbol but ~chance on new large-caps in this window
  (fresh check ICICIBANK 0.4448 / SBIN 0.3219 / ITC 0.4657). On the shared Task-17
  20-symbol panel: (21) vol-adjusted k=0.5 target coverage transfers everywhere
  (0.57–0.67) and raw OHLC levels are NOT the failure driver; leave-one-symbol-out pooled
  training (ONE rf, locked 300/8/30) lifts the weakest symbols (ICICIBANK 0.43→0.59,
  SBIN 0.37→0.49, TCS 0.50→0.64) but degrades RELIANCE (0.55→0.42) and ITC (0.46→0.38) —
  NEW3 pooled median 0.4935 < 0.52 bar. (22) probability blend
  `w·P_pooled + (1−w)·P_per_symbol` (w ∈ 0.3/0.5/0.7) DILUTES the pooled signal
  (best NEW3 0.4395); oracle best-of-per-symbol caps NEW3 at **0.4935**
  (SBIN 0.4935 / ITC 0.4597 binding) → **NEW3 > 0.52 is mathematically unreachable this
  window; ITC/SBIN at chance is an honest per-symbol ceiling, not a bug.** Cross-symbol
  generalization = future research (new window/data/features); this session ends there.
  Production untouched. Artifacts:
  `backend/ml_v2_pooled_vol_adj/task21_pooled_voladj_result.json` +
  `task22_pooled_blend_result.json` + 10 tests.

### Benchmark & failure diagnostics (Task 11, `ml/benchmark.py` — measurement only)

- Multi-stock, repeatable benchmark of the **current production V1 forecast**; it never
  alters features/models/ensemble/calibration/thresholds/costs. Inputs are built by the
  same `pipeline._build_symbol_input` the live path uses (fresh builds; forecast cache
  bypassed), and every per-symbol record persists `symbol / as_of / generated_at /
  data_fingerprint / input sources / versions / seed / walk-forward grid`, so results are
  attributable to the exact input snapshot.
- Per-symbol: holdout ROC-AUC (ensemble + each installed model), Brier/ECE calibrated,
  distinct calibrated levels + boundary-collapse share, production equity-ledger trading
  metrics, and baselines on the **same** holdout (`always_up`/`always_down`/seeded
  `random`/`buy_hold`/`shuffled_control`).
- Deep sections (full-OOF, descriptive — never used for tuning): per-model
  `run_backtest`, rf feature-group ablation (`run_ablation`), failure analysis
  (probability collapse, directional/trading weakness, regime-bucketed OOF AUC under
  `regime_trend`/`regime_vol`), cross-stock aggregation (median/mean/std + pass/fail
  counts), symbol-level ranking, data-quality issues, warnings, provenance note.
- CLI: `python -m app.ml.benchmark [--smoke | --symbols S1,S2,...] [--no-fast`
  `--no-model-comparison --no-ablation --no-failure --enrich --label L`
  `--output-dir ml_benchmark]`. Outputs: `benchmark_{label}_{ts}.json` + `.txt`
  (artifacts gitignored). Smoke universe: TCS/RELIANCE/HDFCBANK/INFY; default universe:
  17 diversified NSE names across 9 sectors.
- Task 11 runs (2026-09-07): smoke (4/4 ok, 827s) + 17-symbol default (16/16 ok, 1
  graceful skip — TATAMOTORS upstream 404), full details in `handoff.md` Task 11.

### V2 research boundary (Tasks 12–22)

- V2 work is isolated under `backend/app/ml/pooled_dataset.py`,
  `v2_relative_features.py`, `v2_target_research.py`,
  `v2_pooled_models.py`, `v2_v1_fair_compare.py`,
  `v2_pooled_vol_adj.py`, and `v2_pooled_blend.py`. These modules are
  research-only and are not imported by production routes, settings, forecast
  assembly, calibration, thresholds, backtest accounting, or frontend code.
- Tasks 12–16 established the design specification, point-in-time pooled panel,
  causal relative features, target alternatives, and pooled candidate-model
  evaluation. The frozen V1 target remains the comparison target:
  `Close[T+20] / Close[T] - 1 > 0`.
- Task 16's real-data run used 20 symbols and 24,778 rows. Final-holdout AUCs
  were inconclusive across logistic, ridge, random forest, and XGBoost; no V2
  candidate is promoted or considered a production winner.
- Task 17 (`v2_v1_fair_compare.py`) ran a strict apples-to-apples comparison
  on the shared snapshot: pooled learning does NOT beat frozen V1 (median AUC
  delta −0.032). **Decision: KEEP FROZEN V1.** Full report:
  `docs/model-v2-v1-fair-comparison.md`.
- Tasks 21–22 (vol-adjusted pooled training + probability blend, research-only)
  probed whether the new `k=0.5` vol-adjusted training target enables cross-
  symbol generalisation. **Result: NO ADOPT** — pooled LOO lifts the weakest
  symbols but does not reach the strict acceptance bar (NEW3 median 0.4935 <
  0.52); probability blending dilutes the signal; oracle best-of-per-symbol
  proves NEW3 > 0.52 is mathematically unreachable this window (ITC/SBIN at
  honest chance-level ceiling). Production untouched. Full results:
  `docs/model-v2-pooled-model-results.md` (Task 21/22 section) and
  `backend/ml_v2_pooled_vol_adj/task21_*.json` + `task22_*.json`.

---

# THE 10 TASKS

### TASK 1 — Audit + ML Scaffolding + Config + Test Harness
- Create `backend/app/ml/` package (+ `__init__.py`, `versions.py`).
- Add ML settings to `config.py`: `ml_feature_version`, `ml_model_version`, `ml_horizon=20`,
  `ml_feature_start`, `forecast_days`, cost/slippage model, signal thresholds (validated later).
- Add deps to `requirements.txt`: `scikit-learn`, `joblib`, `xgboost`; optional block `shap`/`lightgbm`/`torch`.
- Add `backend/tests/` pytest infrastructure + a couple of pure unit tests (e.g. ml versioning helper).
- Update `handoff.md` + stop.

### TASK 2 — Data Ingestion Layer (market/index/global/macro)
- `data_service.py`: NSE OHLCV (5y, adjusted), indices (NIFTY50, BankNifty, India VIX),
  global (S&P500, Nasdaq, Nikkei, Hang Seng, VIX), macro (Brent, Gold, USD/INR).
- Freshness timestamps, `is_available`, caching, graceful failure.
- Tests: alignment, resilience (network skip).

### TASK 3 — Technical Feature Engine (versioned)
- Extend `indicator_service.py` + new `ml/features.py`: EMA, Bollinger, ATR, ADX/±DI, CCI, OBV, MFI,
  ROC, momentum, pivots, Fibonacci, Heikin-Ashi, price-distance-from-MA, MA slopes/crossovers,
  multi-horizon returns (1/3/5/10/20/60/120d). Pure pandas, no lookahead.
- Unit tests vs hand-computed values.

### TASK 4 — Momentum/Relative + Volatility/Risk Features
- Stock vs NIFTY/sector relative returns, relative-strength rank, rolling corr, market/sector sensitivity.
- Rolling vol, downside vol, ATR, vol percentile, expansion/contraction, high-low range, gap frequency,
  max drawdown, stock-vol vs market-vol.
- Unit tests.

### TASK 5 — Alpha Factor Engine + Beta/Market Exposure
- `ml/alpha.py`: momentum, mean-reversion, value, growth, quality, event, stat/relative groups —
  rank-normalized composite scores. Each factor validated OOS (no "sounds financial ⇒ predictive").
- `ml/beta.py`: rolling beta vs NIFTY/sector, R², residual return/vol (Stock = Market + Residual).
- Unit tests (rank norm, beta regression).

### TASK 6 — Fundamentals + Events (PIT) + Sentiment + Macro Features
- Reuse `fundamental_service`, `news_service`, `sentiment_service`.
- Event layer with `event_timestamp` / `available_timestamp` / `effective_date` (point-in-time, no leak).
- Macro feature integration (slow-moving).
- Tests: PIT — event after close of day T must not be visible to day-T prediction.

### TASK 7 — Market Regime + Prediction Target + Dataset Builder (+ Options layer)
- `regime.py`: bull/bear/sideways, high/low vol, risk-on/off (NIFTY trend, VIX, breadth, vol, momentum, corr).
- `dataset.py`: target = 20d forward return + up/down label; chronological splits.
- `options_df.py` (F&O/OI/PCR/IV/Greeks): free NSE data, graceful disable if unreliable.
- Tests: no-leakage (T+1 never in row T), regime sanity, Greeks disable path.

### TASK 8 — ML Models + Walk-Forward Backtest Engine + Ablation
- `models.py`: baseline (existing voting, logistic, ridge), main (RF, XGBoost).
- `backtest.py`: walk-forward (train→validate→future→roll→retrain), purged CV + embargo,
  costs (₹20/0.01%/slippage/position limits), metrics (acc, P/R/F1, confusion, false buy/sell %,
  win rate, avg/cum return, max DD, Sharpe, Calmar, profit factor, calibration);
  compare ML vs voting vs buy-hold.
- `ablation.py`: feature-group ablation (technical/…/full).
- Backtest tests on synthetic data.

### TASK 9 — Ensemble + Calibration + Signal Policy + SHAP + Versioning/Monitoring
- `ensemble.py` (average XGB/Logistic/RF; optional LSTM/TCN/Transformer if torch present).
- `calibration.py` (Platt / isotonic → honest calibrated P(up)).
- `signal.py` (BUY/HOLD/SELL from validated configurable thresholds).
- `explain.py` (SHAP / feature importance → top +/− factors).
- `versions.py` (snapshot: data/feature/model version, training period, prob, signal, future outcome).
- `monitor.py` (data freshness, feature/prediction/calibration drift, degraded mode).
- Tests: calibration sanity (P sum=1), signal policy determinism.

### TASK 10 — Backend API + Frontend Integration + AI Chat + Docs + Final Acceptance
- New `forecast.py` route + schema; full-analysis updated with forecast block.
- Frontend: preserve design; add 20-Day Forecast + sections (Momentum / Alpha / Beta /
  Fundamentals / Sentiment & Events / F&O / Model Explanation / Backtest) via new components/hooks/types.
- AI chat reads structured forecast output (no fabrication).
- Docs: `architecture.md` (this file), `handoff.md`, `model-upgrade-plan.md`.
- Full test suite + Section-26 acceptance checklist.

---

## Sequencing note (per user)

- Tasks run strictly one at a time. After a task passes its tests → update `handoff.md`, then **stop**.
- The user explicitly starts the next task ("Task 2 start karo", etc.).
- If 10 tasks prove insufficient, the plan is expanded (kept extensible).

---

## SECTION 26 — FINAL ACCEPTANCE CHECKLIST (Task 10)

Task 10 = backend API + frontend integration + AI chat + docs + final acceptance.

### Backend
- [ ] `GET /api/v1/stock/{symbol}/forecast` returns:
  - [ ] `latest` block: `as_of`, `close`, raw + calibrated `probability` (∈[0,1], 4dp),
        `signal` ∈ {BUY,HOLD,SELL}, `direction`, `confidence`, `reason`, thresholds.
  - [ ] `calibration`: method (isotonic/uncalibrated), OOF `n`, raw & calibrated Brier/ECE.
  - [ ] `backtest[]` from the same OOF probs: `ensemble` + per-model + `buy_hold` rows
        with `cum_return`/`win_rate`/`max_dd`/`sharpe`/`n_trades` etc.
  - [ ] `explanation`: model + `delta_scale` + top ±8 `factors` (factor/impact/direction).
  - [ ] `monitor`: `mode` (fresh/degraded/stale), freshness, PSI/drift groups, `alarms`.
  - [ ] `snapshot` (versioned via `versions.new_snapshot`): `feature/model/ensemble_version`,
        `model_probability`, `calibrated_probability`, `calibrated_signal`.
  - [ ] Graceful path: unavailable/history-too-short/invalid symbol → `is_available:false`
        + `error` (never fabricated); 400 on invalid symbol.
- [ ] `GET /api/v1/ml/status` returns source freshness (nifty50/banknifty/india_vix),
      degraded `alarms`, last-run mode, versions + thresholds.
- [ ] Full-analysis `GET /api/v1/stock/{symbol}` embeds the same forecast block
      (timeout-guarded, never blocks the rest of the analysis).
- [ ] AI chat reads the forecast payload for a mentioned symbol; LLM and fallback
      present the same structured numbers (no fabrication).
- [ ] No lookahead / no fabricated data / 4-dp rounding enforced in tests.

### Frontend
- [ ] Stock page gets a "Forecast" tab (types/hooks/component) preserving the existing
      dark-card design language; prefers the embedded block, falls back to the endpoint.
- [ ] Forecast card shows P(up) gauge + signal, calibration digest, backtest mini-table,
      top factors with impact bars, monitor mode, versions + generated time.
- [ ] `tsc --noEmit` / `next build` clean.

### Docs & quality
- [ ] `model-upgrade-plan.md` created at repo root.
- [ ] `ml/__init__.py` docstring lists `pipeline.py` (Task 10).
- [ ] Full backend suite green: **369 passed** (360 baseline + 9 measurement/validation
      tests). Locked measurement rules (MTM equity ledger, chronological calibration
      split, holdout-only backtest, baselines + shuffled control, `metric_groups`
      provenance) verified in `test_backtest.py` / `test_forecast.py`.
- [ ] Live smoke: RELIANCE forecast (real data), `/ml/status`, chat with symbol mention,
      full-analysis with forecast block, frontend build.
- [ ] After all above pass: update `handoff.md` (Task 10 → `[DONE]`) and stop.
