# FinAlly India — AI Trading Workstation for Indian Markets

## 1. Vision

FinAlly India is a visually stunning AI-powered trading workstation for **Indian cash equity** (Phase 1: **equity-only, no F&O**). It streams live or simulated prices for NSE tickers, lets users trade a **virtual INR portfolio**, and integrates an LLM chat assistant that can analyze positions and execute trades on the user's behalf.

It should feel like a **modern Bloomberg / broker terminal for India**, with an AI copilot that understands:
- Indian tickers (e.g. `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `ICICIBANK`)
- Indices such as **NIFTY 50, NIFTY BANK, SENSEX**
- Indian market hours, holidays, and basic regulatory context at a high level.

FinAlly India is tuned to:
- Use **INR** as the base currency
- Seed with **Indian watchlists and portfolios**
- Use **Yahoo Finance (yfinance)** for real-time Indian stock prices when available — free, no API key — with an in-process simulator as fallback.

## 2. User Experience

### First Launch

The user runs a single Docker command (or a provided start script). A browser opens to `http://localhost:8000`. No login, no signup. They immediately see:

- A watchlist of **10 default Indian tickers** with live-updating prices in a grid
- **₹10,00,00** (one lakh) in virtual cash
- A dark, data-rich trading terminal aesthetic
- An AI chat panel ready to assist with India-specific queries

### Default Watchlist

- `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, `ICICIBANK`,
  `SBIN`, `LT`, `ASIANPAINT`, `BAJFINANCE`, `MARUTI`

Use **liquid, large-cap NSE names**. When calling Yahoo Finance, append `.NS` (e.g. `TCS.NS`, `RELIANCE.NS`).

### What the User Can Do

- **Watch prices stream** — prices flash green (uptick) or red (downtick) with subtle CSS animations that fade
- **View sparkline mini-charts** — price action beside each ticker, accumulated from the SSE stream since page load
- **Click a ticker** to see a larger detailed chart in the main chart area
- **Buy and sell shares** — market orders only, instant fill at current price, no fees, no confirmation dialog
- **Monitor their portfolio** — heatmap (treemap) by weight and P&L, plus a P&L chart of total portfolio value over time
- **View a positions table** — ticker, quantity, average cost, current price, unrealized P&L, % change
- **Chat with the AI assistant** — ask about the portfolio, get analysis, and have the AI execute trades and manage the watchlist in natural language
- **Manage the watchlist** — add/remove tickers manually or via the AI chat

All money values are displayed in **INR** (e.g. `₹10,00,00`).

### Visual Design

- **Dark theme**: backgrounds around `#0d1117` or `#1a1a2e`, muted gray borders, no pure black
- **Price flash animations**: brief green/red highlight on price change, fading over ~500ms
- **Connection status indicator**: green = connected, yellow = reconnecting, red = disconnected
- **Professional, data-dense layout**: Bloomberg/trading-terminal inspired; desktop-first, functional on tablet

### Color Scheme

- Accent Yellow: `#ecad0a`
- Blue Primary: `#209dd7`
- Purple Secondary: `#753991` (submit buttons)

## 3. Architecture Overview

### Single Container, Single Port

```
┌─────────────────────────────────────────────────┐
│  Docker Container (port 8000)                   │
│                                                 │
│  FastAPI (Python/uv)                            │
│  ├── /api/*          REST endpoints             │
│  ├── /api/stream/*   SSE streaming              │
│  └── /*              Static file serving       │
│                      (Next.js export)            │
│                                                 │
│  SQLite database (volume-mounted)               │
│  Background task: yfinance poll or simulator    │
└─────────────────────────────────────────────────┘
```

- **Frontend**: Next.js with TypeScript, static export, served by FastAPI
- **Backend**: FastAPI (Python), `uv` project
- **Database**: SQLite at `db/finally_india.db`, volume-mounted for persistence
- **Real-time data**: Server-Sent Events (SSE), one-way server→client
- **AI**: LiteLLM → OpenRouter (e.g. Cerebras / Nemotron), structured outputs for trade execution
- **Market data**:
  - **Primary**: Yahoo Finance via **yfinance** — free, no API key; use `yf.Ticker("TCS.NS").info` (e.g. `regularMarketPrice`, `currency`) for NSE symbols with `.NS` suffix
  - **Fallback**: In-process simulator when yfinance is unavailable or for testing

### Why These Choices

| Decision | Rationale |
|----------|-----------|
| SSE over WebSockets | One-way push is enough; simpler, universal browser support |
| Static Next.js export | Single origin, one port, one container |
| SQLite | No auth, single-user; self-contained, zero config |
| Single Docker container | One command to run |
| uv for Python | Fast, reproducible dependency management |
| Market orders only | No order book or limit logic; simpler portfolio math |
| yfinance for real prices | Free, unlimited; no API key; NSE symbols as `SYMBOL.NS` |

## 4. Directory Structure

```
FinalIndia/   (or finally/ with India-only app)
├── frontend/           # Next.js TypeScript (static export)
├── backend/            # FastAPI uv project (Python)
│   └── db/             # Schema, seed data, migrations
├── FinalIndia/         # Planning and config (this doc)
│   └── PLAN.md
├── scripts/            # Docker start/stop
├── test/               # E2E (India scenarios)
├── db/                 # SQLite volume mount
├── Dockerfile
├── docker-compose.yml
├── .env
└── .gitignore
```

This project runs **India-only**; no region toggle.

### India Defaults

- **Starting cash**: ₹10,00,000 (ten lakh)
- **Default index in header**: NIFTY 50 level and today’s % change (from yfinance or simulator)
- **Default watchlist**: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, LT, ASIANPAINT, BAJFINANCE, MARUTI (use `.NS` for yfinance)
- **Simulator (fallback)**: GBM with Indian large-cap–style drift/volatility, correlated sectors, occasional 2–5% event moves

## 5. Environment Variables

```bash
# Required: OpenRouter API key for LLM chat
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Optional: use simulator only (e.g. for tests or offline)
# If set to "true", backend uses simulator instead of yfinance
USE_MARKET_SIMULATOR=false

# Optional: deterministic mock LLM responses (testing)
LLM_MOCK=false
```

### Behavior

- **Default**: Backend uses **yfinance** to fetch real NSE prices for watchlist symbols (e.g. `TCS.NS`, `RELIANCE.NS`); no API key required.
- If `USE_MARKET_SIMULATOR=true`: backend uses the in-process simulator only.
- If `LLM_MOCK=true`: backend returns deterministic mock LLM responses for tests.

## 6. Market Data

### Two Modes, One Interface

- **yfinance (default)**: Python library `yfinance`; install with `pip install yfinance`. Query e.g. `yf.Ticker("TCS.NS").info` for `regularMarketPrice`, `currency` (INR). Poll at a reasonable interval (e.g. 15–60 seconds) and write to the shared price cache. No API key.
- **Simulator (fallback)**: In-process GBM-based prices for the same watchlist; used when `USE_MARKET_SIMULATOR=true` or when yfinance is unavailable.

Both implement the same internal interface. SSE and frontend are agnostic to the source.

### yfinance Integration

- NSE symbols in Yahoo Finance use the **.NS** suffix (e.g. `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`).
- Example: `t = yf.Ticker("TCS.NS"); info = t.info; price = info.get("regularMarketPrice")` — returns INR price.
- Backend maps display tickers (e.g. `TCS`) to Yahoo symbols (`TCS.NS`) when calling yfinance; store and display the short form in the UI.

### Simulator (Fallback)

- GBM with drift/volatility tuned to Indian large caps
- Updates at ~500ms intervals
- Correlated moves (e.g. IT, banks); occasional 2–5% event moves
- Seed prices in INR (e.g. RELIANCE ~₹2,500, TCS ~₹4,000, HDFCBANK ~₹1,500)

### Shared Price Cache and SSE

- One background task (yfinance poller or simulator) writes to an in-memory price cache
- SSE endpoint `GET /api/stream/prices` pushes: ticker, price, previous price, timestamp, direction
- Client uses `EventSource` and reconnects automatically

## 7. Database

Schema is unchanged; only seed data and currency are India-specific:

- Default user: `cash_balance=1000000.0` (ten lakh INR)
- Default watchlist: Indian tickers listed above

Tables: `users_profile`, `watchlist`, `positions`, `trades`, `portfolio_snapshots`, `chat_messages`

## 8. API Endpoints

- **Market**: `GET /api/stream/prices` — SSE price stream
- **Portfolio**: `GET /api/portfolio`, `POST /api/portfolio/trade`, `GET /api/portfolio/history`
- **Watchlist**: `GET /api/watchlist`, `POST /api/watchlist`, `DELETE /api/watchlist/{ticker}`
- **Chat**: `POST /api/chat`
- **System**: `GET /api/health`

All monetary values in responses and frontend are **INR** with Indian number formatting.

## 9. LLM Integration

- System prompts and examples use **NSE/BSE, NIFTY, Indian tickers, INR P&L**.
- Assistant behaviour:
  - Trading is simulated; no real brokerage connection.
  - No specific legal/tax advice; can discuss risk, diversification, sizing.
- Analyse sector exposure (e.g. financials), suggest trades with reasoning, execute trades and watchlist changes via structured JSON (same schema as core).
- Use OPENROUTER_API_KEY from `.env`; see `FinalIndia/.claude/skills/cerebras/SKILL.md` for LiteLLM/Cerebras usage.

## 10. Frontend Design

- Watchlist: default Indian tickers; prices with flash and sparklines
- Main chart: selected ticker, NSE/BSE context in title
- Header: portfolio value, cash, optional NIFTY 50 level and % change
- Portfolio heatmap (treemap), P&L chart, positions table, trade bar, AI chat panel
- Use `EventSource` for `/api/stream/prices`; Tailwind dark theme

## 11. Docker & Deployment

- Multi-stage Dockerfile: Node build frontend, then Python image with FastAPI and static files
- Volume for `db/finally_india.db`
- Single container, port 8000; start/stop scripts as provided

## 12. Testing Strategy

- **Unit**: Backend — yfinance parsing, simulator, trade/portfolio logic, LLM structured output. Frontend — components, price flash, watchlist, portfolio display.
- **E2E (India)**: Default watchlist shows Indian tickers; buy/sell RELIANCE or TCS updates INR balance; P&L and charts use Indian data; chat (e.g. “How concentrated am I in financials?”, “Suggest 2–3 NIFTY stocks to diversify”) works with LLM mock and real model.
- Run E2E with `USE_MARKET_SIMULATOR=true` and `LLM_MOCK=true` for speed and determinism.
