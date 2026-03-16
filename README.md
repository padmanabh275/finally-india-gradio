# FinAlly India — AI Trading Workstation

FinAlly India is an AI‑powered trading workstation for **Indian NSE cash equities**. It streams live prices from Yahoo Finance, simulates a virtual INR portfolio, and includes an LLM copilot that can analyse positions and place trades from natural‑language commands.

This repo contains both the original FastAPI/Next.js implementation and a new, self‑contained **Gradio terminal** for quick local use and Hugging Face Spaces.

## Key Features

- **Indian watchlist by default** — RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, LT, ASIANPAINT, BAJFINANCE, MARUTI
- **Virtual portfolio** — ₹10,00,000 starting cash, market orders, instant fills
- **Positions & P&L** — positions table, per‑ticker P&L, portfolio P&L over time
- **Advanced visuals (Gradio)**:
  - Per‑ticker **main price chart** (5d / 15m)
  - **Portfolio P&L** line chart
  - **Portfolio by ticker** exposure bar chart
  - **Ticker sparkline** (1d / 5m) under the watchlist
- **AI copilot** — local Hugging Face chat model (TinyLlama by default) that:
  - Explains the portfolio in natural language
  - Understands commands like “buy 10 RELIANCE shares” and executes trades
- **Watchlist management** — add any NSE ticker and trade it immediately

## Quick Start – Gradio Terminal (Recommended)

From the repo root:

```bash
conda activate torch_env  # or your Python env
pip install -r FinalIndia/requirements_gradio.txt
python FinalIndia/gradio_app_clean.py
```

Then open the URL shown in the terminal (default `http://localhost:7860`).

### LLM Settings

The Gradio app uses a local Hugging Face model. Configure via `.env`:

```bash
LOCAL_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0  # or another small chat model
HF_TOKEN=your_hf_token_if_needed
```

If `LOCAL_LLM_MODEL` is omitted, the TinyLlama chat model is used by default. The app runs on **CPU only** by design.

## Original FastAPI / Next.js Stack

The original architecture (see `planning/`, `backend/`, `frontend/`) uses:

- **Frontend**: Next.js (static export) with TypeScript and Tailwind CSS
- **Backend**: FastAPI (Python/uv) with SSE price streaming
- **Database**: SQLite
- **AI**: LiteLLM → OpenRouter with structured outputs
- **Market data**: GBM simulator by default, Massive/Polygon optional

Refer to `planning/PLAN.md` for the full India‑specific design and API contract.

## Project Structure (excerpt)

```text
finally/
├── FinalIndia/
│   ├── PLAN.md               # India-specific product plan
│   ├── gradio_app_clean.py   # Clean Gradio trading terminal (current focus)
│   ├── gradio_app.py         # Earlier Gradio prototype
│   ├── streamlit_app.py      # Streamlit prototype terminal
│   ├── requirements_gradio.txt
│   ├── requirements_hf.txt
│   └── requirements.txt
├── backend/                  # FastAPI uv project (original stack)
├── frontend/                 # Next.js static export (original stack)
├── planning/                 # Project documentation and agent contracts
└── README.md
```

## License

See [LICENSE](LICENSE).
#
