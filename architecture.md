# Architecture — AVORA Stock Analyzer

## System Overview

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│         FRONTEND            │  /api/* │           BACKEND            │
│      Next.js 15 + TS        │ ──────► │        FastAPI (Python)      │
│                             │  proxy  │                              │
│  Pages:                     │         │  Routers (prefix /api/v1)    │
│   Dashboard /               │         │   stock, signals, market,    │
│   Signals / Watchlist /     │         │   watchlist, compare,        │
│   Stock detail [symbol]     │         │   feedback, health, CHAT(*)  │
│   ...                       │         │                              │
│  ChatWidget (bottom-right)  │         │  Services:                   │
│   └─ POST /api/chat         │         │   chat_service (*new)        │
│      └─ api/[...path] proxy │         │   ├─ intent parser           │
│         └─ BACKEND_URL      │         │   └─ LLM client (optional)   │
└─────────────────────────────┘         │   signals_service ──► model  │
                                       │   signal_service  ──► model  │
                                       │   yfinance_service ─► yfinance│
                                       │   market_service / news / ... │
                                       └──────────────┬────────────────┘
                                                      │
                                        ┌─────────────▼─────────────────┐
                                        │  DATA SOURCES                 │
                                        │   yfinance (NSE/BSE OHLCV)    │
                                        │   NewsData.io / MoneyControl  │
                                        │   Marketsmith / NSE           │
                                        │   LLM (OpenAI-compatible) (*) │
                                        └───────────────────────────────┘
```

(*) = new AI Chat layer added by this feature.

---

## Folder Structure

```
stocks_analyzer-main/
├── handoff.md                     # Project roadmap (TODO / IN_PROGRESS / DONE)
├── architecture.md                # This document
├── render.yaml                    # Render deployment (backend)
├── runtime.txt                    # Python version for Render
│
├── backend/                       # FastAPI backend
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py                # App factory, CORS, rate-limit, router mounts
│       ├── config.py              # Pydantic settings (+ LLM keys *)
│       ├── core/                  # middleware (rate limit), __init__
│       ├── routes/                # API route modules
│       │   ├── stock.py           # /stock/{symbol}/... (price, signal, news, ...)
│       │   ├── signals.py         # /signals/all
│       │   ├── market.py          # /market/overview
│       │   ├── watchlist.py, compare.py, feedback.py, health.py
│       │   └── chat.py            # (*NEW*) POST /api/chat
│       ├── services/              # business logic
│       │   ├── chat_service.py    # (*NEW*) intent parse + LLM + model data
│       │   ├── signals_service.py # batch scan of ALL_SYMBOLS → categories
│       │   ├── signal_service.py  # generate_trade_signal() → THE prediction model
│       │   ├── market_service.py  # top bullish/bearish
│       │   ├── yfinance_service.py# price/intraday/search via yfinance
│       │   ├── indicator_service.py # RSI / MACD / SMA / EMA
│       │   ├── fundamental_service.py, news_service.py, shareholding_service.py,
│       │   │   sentiment_service.py, screener_service.py, cache_service.py, ...
│       ├── schemas/               # Pydantic response models (BaseSchema camelCase)
│       └── utils/                 # validators, exceptions, formatters, scraper_utils
│
└── frontend/                      # Next.js 15 + TypeScript
    ├── package.json
    ├── .env.example               # BACKEND_URL for the API proxy
    ├── next.config.ts
    └── src/
        ├── app/
        │   ├── layout.tsx / providers.tsx   # Root layout + React Query
        │   ├── page.tsx                     # Dashboard
        │   ├── stock/[symbol]/page.tsx      # Stock detail (chart, signal, ...)
        │   ├── signals/page.tsx             # Signal Center
        │   └── api/[...path]/route.ts       # Proxy /api/* → BACKEND_URL
        ├── components/
        │   ├── layout/AppShell.tsx          # Sidebar + Header + (*NEW*) ChatWidget
        │   ├── chat/                        # (*NEW*) ChatWidget, ChatMessage, StockResultCard
        │   ├── chart/ ...                   # Charts (candlestick, area, sparkline, ...)
        │   ├── stock/ ...                   # CompanyHeader, TradeSignal, TechnicalPanel, ...
        │   └── ui/ ...                      # Button, Badge, Tabs, Modal, ...
        ├── hooks/                           # TanStack Query hooks (useSignals, useSignal, ...)
        ├── lib/
        │   ├── api.ts                       # apiGet / apiPost (base /api/v1)
        │   ├── chat-api.ts                  # (*NEW*) chatApi.send() → /api/chat
        │   ├── constants.ts / formatters.ts / cn.ts / indicators.ts
        ├── store/                           # Zustand stores (watchlist, history, ui)
        ├── styles/globals.css               # Tailwind v4 + dark/light theme
        └── types/                           # TS types (signal, signals, stock, chat (*NEW*), ...)
```

---

## Data Flow

### 1. Prediction generation (existing model)
```
signals_service.get_all_signals()
  └─ ThreadPool: for each symbol in ALL_SYMBOLS (~80 NSE stocks)
       └─ yfinance_service / yf.Ticker(symbol.NS).history(period="3mo")
            └─ indicator_service → RSI, MACD, SMA20, SMA50
            └─ signal_service.generate_trade_signal(price, rsi, macd, signal, sma20, sma50)
                 → { signal: { action, direction, confidence, reasons }, quote }
  └─ Categorize (strong-buy, buy, hold, sell, strong-sell, rsi-*, macd-*, trend)
  └─ Cache 10 min (cache_service) → GET /api/v1/signals/all
```

### 2. AI Chat request (new feature)
```
User (ChatWidget) ──► POST /api/chat        { message, history? }
  └─ Next.js proxy (api/[...path]/route.ts) ──► FastAPI
       └─ chat_service.process_chat(message)
            ├─ 1. parse_intent()  → { maxPrice?, minPrice?, action?, bullishOnly? }
            │       regex rules (EN + Hinglish): "under 500", "₹500 se kam", "buy", ...
            ├─ 2. load_predictions() → get_all_signals()   ← EXISTING MODEL ONLY
            │       flatten + dedupe across categories (symbol, currentPrice, signal, confidence, reasons, trend)
            ├─ 3. filter + rank stocks by intent (price, action, confidence)
            └─ 4. format_answer(stocks, intent)
                 ├─ LLM mode  (OPENAI_API_KEY set)
                 │   system prompt: use ONLY provided prediction data, never invent
                 │   → natural-language reply (HTML-friendly markdown)
                 └─ Fallback mode (no key)
                     deterministic template: "X stocks under ₹500: INFY (BUY 80%) ..."
  └─ Returns { reply, stocks[], intent, source: "existing-model" }
       └─ ChatWidget renders reply text + clickable stock cards
            └─ click stock → /stock/[symbol]  (full detail: chart, 20-day signal, reasons)
```

### 3. Stock detail page
```
/stock/[symbol]
  └─ useFullAnalysis → GET /api/v1/stock/{symbol}  (price + fundamentals + signal)
  └─ useStockPrice   → GET /api/v1/stock/{symbol}/price   (history + indicators → charts)
  └─ useSignal       → GET /api/v1/stock/{symbol}/signal  (trade signal + reasons)
  └─ useNews / useShareholding → news, ownership
```

---

## Key Connections

| Layer            | Component                        | How it connects                                        |
|------------------|----------------------------------|--------------------------------------------------------|
| Frontend → Backend | `api/[...path]/route.ts` proxy  | `BACKEND_URL` env (default Render URL); dev: localhost:8000 |
| Frontend → Backend | `lib/api.ts`                    | `API_BASE = /api/v1`; `apiGet`/`apiPost` fetch via proxy |
| Backend → Data    | `yfinance_service`              | yfinance → NSE/BSE OHLCV; exchange suffix `.NS`/`.BO`   |
| Backend → Model   | `signal_service.generate_trade_signal` | technical indicators → BUY/SELL/HOLD prediction    |
| Backend → Chat    | `chat_service`                  | reads existing signals (never generates its own forecast) |
| Chat → LLM (opt.) | OpenAI-compatible HTTP call     | `OPENAI_API_KEY`; only formats data, does not source data |

## Guarantee: No hallucinated predictions
The `/api/chat` pipeline fetches all stock data from `get_all_signals()` (the existing
technical-indicator model). The LLM (if enabled) is given a strict system prompt that forbids
inventing symbols, prices, or signals — it only rephrases the provided prediction data.
If no LLM key is configured, a deterministic template builds the reply from the same data.
