# Project Handoff — AVORA Stock Analyzer (stocks_analyzer-main)

> Senior Fullstack AI Engineer handoff. This file tracks the project roadmap.
> Update it after every change: move completed work to `[DONE]`, add new work to `[TODO]`.

---

## [DONE]

### Existing Platform (baseline)
- [x] Backend FastAPI app under `backend/` — routers under `/api/v1`
  - `stock`, `watchlist`, `compare`, `market`, `feedback`, `health`, `signals`
- [x] Existing prediction engine (technical-indicator model) —
  - `backend/app/services/signal_service.py` → `generate_trade_signal()` (Trend/SMA, RSI, MACD, sentiment → BUY/SELL/HOLD + confidence + reasons)
  - `backend/app/services/signals_service.py` → scans `ALL_SYMBOLS` (~80 NSE stocks) → `get_all_signals()`
  - `backend/app/services/market_service.py` → top bullish/bearish from `STOCK_WATCH`
- [x] Data layer — `yfinance_service.py` (price/intraday/search), `fundamental_service.py`, `news_service.py`, `shareholding_service.py`, `sentiment_service.py`
- [x] Frontend Next.js + TypeScript under `frontend/`
  - Dashboard, Signals Center, Watchlist, Compare, Basket, History, Settings, Stock detail page (`/stock/[symbol]`)
  - API proxy: `src/app/api/[...path]/route.ts` forwards `/api/*` → `BACKEND_URL`
  - API client helpers: `src/lib/api.ts` (`apiGet` / `apiPost`), base `/api/v1`

### AI Chat Assistant + Stock Prediction feature
- [x] Exploration & understanding of the full codebase completed
- [x] `architecture.md` created (architecture diagram + folder structure + data flow)
- [x] Backend: added LLM configuration to `config.py` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, etc.) + `.env.example`
- [x] Backend: created `app/services/chat_service.py`
  - Intent parser (max/min price, action, sentiment filters) — supports Hinglish + English
  - Fetches predictions **only** from existing model (`get_all_signals`)
  - LLM formatting via OpenAI-compatible API when key is set; deterministic fallback template otherwise (never fabricates data)
- [x] Backend: created `app/routes/chat.py` — `POST /api/chat` (+ alias `/api/v1/chat`) and wired into `main.py`
- [x] Frontend: created chat types (`src/types/chat.ts`) + chat API helper in `src/lib/chat-api.ts`
- [x] Frontend: built `ChatWidget` component (bottom-right floating) + stock result cards (`StockResultCard`) — stock names link to `/stock/[symbol]`
- [x] Frontend: mounted `ChatWidget` inside `AppShell.tsx`
- [x] **Groq integration** — chatbot ab live Groq LLM use karta hai:
  - `backend/.env`: `OPENAI_API_KEY` (Groq key), `OPENAI_BASE_URL=https://api.groq.com/openai/v1`, `OPENAI_MODEL=groq/compound`
  - `groq` python package installed; `.env` gitignored (backend + root)
  - Verified live: `/api/chat` returns `source: "llm"` with Hindi replies from `groq/compound`, using only existing-model predictions
- [x] **Small-talk / greeting detection** — `hlo`, `good evening`, `thanks`, `namaste`, `how are you` ab friendly reply dete hain (0 stocks), stock query hone pe hi predictions aati hain (`is_smalltalk()` in `chat_service.py`)
- [x] **ChatGPT-like general conversation** — non-stock questions (`what is RSI`, `joke sunao`, `stock market kaise kaam karta hai`) ab LLM se free-form answers dete hain via `_general_reply()` (3-way routing: smalltalk → general chat → stock predictions). Stock queries (`price/signal/buy/sell/₹ filters`) pe hi existing model ke predictions aate hain.
- [x] **Multi-turn context** — frontend ab poora conversation history (user + assistant) backend ko bhejta hai (`ChatWidget` history fix)
- [x] **Ownership (shareholding) tab fix** — "ownership kam ni kr rha" root cause mila aur fix kiya:
  - **Root cause:** `ShareholdingChart`/`MajorShareholders` frontend theek the aur backend data bhi theek de raha tha (screener.in se 12 quarters + 10 shareholders, 200 OK in 2-8s). Problem sirf **slow stocks** pe thi — ZOMATO/TATAMOTORS/NTPC jaise (jahan screener quarterly table milta nahi) endpoint **20-31s** leta tha (yfinance fallback), jo Next.js proxy ke 30s timeout se zyada tha → proxy `503 "Backend service unavailable"` → ownership tab "No data" dikhata tha.
  - **Fixes applied:**
    1. `frontend/src/app/api/[...path]/route.ts`: `FETCH_TIMEOUT` 30s → **90s**
    2. `backend/app/services/marketsmith_service.py`: playwright waits kam — goto 60s→30s, `wait_for_timeout` 12s→4s, selector 30s→15s (har stock pe pehle minimum 12s lagta tha, ab ~4s me fail-fast)
    3. `backend/app/services/shareholding_service.py`: major-shareholder source order — **screener.in pehle**, phir moneycontrol, nse, marketsmith **last** (screener sabse reliable + fast hai)
    4. `backend/app/routes/stock.py`: shareholding route pe hard timeout — `asyncio.wait_for(asyncio.to_thread(fetch_shareholding_data), timeout=50)` taki sabse slow case me bhi 50s ke andar response aa jaye
  - Verified: RELIANCE route test 200 OK in 5.8s (quarterlyData/latest/majorShareholders sahi); `tsc --noEmit` clean
- [x] Verification:
  - Backend: syntax + full app import OK (temp venv), API smoke test via TestClient passed, real-data test (live yfinance) passed — intent "500 se kam" correctly filtered to ₹<500 stocks with model predictions + Hinglish reply
  - Frontend: `tsc --noEmit` clean, `next lint` no errors, `next build` successful

---

## [IN_PROGRESS]

- (none)

---

## [TODO]

- [ ] Test the full user flow end-to-end (local backend + frontend)
- [ ] Add streaming/SSE responses from `/api/chat` for a typing effect (optional, nice-to-have)
- [ ] Add conversation history persistence (e.g., backend JSON file like `feedback.json`)
- [ ] Add more intent capabilities (compare stocks, top gainers/losers, specific stock deep-dive)
- [ ] Add suggested quick-prompt chips in the chat widget
- [ ] Add a "Select model" setting in the UI (Groq models list is dynamic per API key)
