import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

# Force transformers to use PyTorch backend and keep TensorFlow quiet
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import streamlit as st
import torch
import yfinance as yf
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()


APP_TITLE = "FinAlly India – Streamlit Prototype"
STARTING_CASH = 1_000_00.0  # ₹10,00,000

DEFAULT_TICKERS: List[str] = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ASIANPAINT",
    "BAJFINANCE",
    "MARUTI",
]

# Simple sector tagging for the default top 10 (can be extended)
DEFAULT_SECTOR_BY_TICKER: Dict[str, str] = {
    "RELIANCE": "Energy / Conglomerate",
    "TCS": "IT Services",
    "INFY": "IT Services",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "LT": "Capital Goods / Infra",
    "ASIANPAINT": "FMCG / Paints",
    "BAJFINANCE": "NBFC / Financials",
    "MARUTI": "Auto",
}


def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"


@st.cache_data(ttl=30)
def fetch_prices(tickers: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    if not tickers:
        return prices

    yahoo_symbols = [f"{t}.NS" for t in tickers]
    data = yf.download(
        " ".join(yahoo_symbols),
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
    )

    # Guard against empty DataFrames to avoid .iloc[-1] index errors
    if len(data) > 0 and "Close" in data:
        last_row = data["Close"].iloc[-1]
        for t, ysym in zip(tickers, yahoo_symbols):
            sym = ysym
            if sym in last_row.index:
                prices[t] = float(last_row[sym])
    else:
        for t, ysym in zip(tickers, yahoo_symbols):
            tkr = yf.Ticker(ysym)
            info = tkr.info or {}
            price = info.get("regularMarketPrice")
            if price is not None:
                prices[t] = float(price)

    return prices


def init_state() -> None:
    if "cash" not in st.session_state:
        st.session_state.cash = STARTING_CASH
    if "positions" not in st.session_state:
        # positions: {ticker: {"qty": float, "avg_cost": float}}
        st.session_state.positions = {}
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = list(DEFAULT_TICKERS)
    if "sectors" not in st.session_state:
        # sectors: {ticker: sector_name}
        st.session_state.sectors = dict(DEFAULT_SECTOR_BY_TICKER)
    if "trades" not in st.session_state:
        st.session_state.trades = []  # list of dicts
    if "prev_prices" not in st.session_state:
        st.session_state.prev_prices: Dict[str, float] = {}
    if "portfolio_history" not in st.session_state:
        # list of {"time": datetime, "value": float}
        st.session_state.portfolio_history = []
    if "chat_messages" not in st.session_state:
        # list of {"role": "user"|"assistant", "content": str}
        st.session_state.chat_messages = []
    if "hf_pipeline" not in st.session_state:
        st.session_state.hf_pipeline = None


def compute_portfolio_value(prices: Dict[str, float]) -> float:
    total = st.session_state.cash
    for ticker, pos in st.session_state.positions.items():
        qty = pos["qty"]
        price = prices.get(ticker)
        if price is not None:
            total += qty * price
    return total


def update_portfolio_history(prices: Dict[str, float]) -> None:
    """Append latest portfolio value to in-memory history for simple P&L chart."""
    value = compute_portfolio_value(prices)
    st.session_state.portfolio_history.append(
        {"time": datetime.now(), "value": value}
    )
    # Keep history to a reasonable length
    if len(st.session_state.portfolio_history) > 500:
        st.session_state.portfolio_history = st.session_state.portfolio_history[-500:]


def record_trade(
    side: str,
    ticker: str,
    qty: float,
    price: float,
) -> None:
    st.session_state.trades.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "side": side,
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "value": qty * price * (1 if side == "BUY" else -1),
        }
    )


def execute_trade(side: str, ticker: str, qty: float, price: float) -> None:
    if qty <= 0:
        st.warning("Quantity must be positive.")
        return
    cost = qty * price

    positions = st.session_state.positions
    pos = positions.get(ticker, {"qty": 0.0, "avg_cost": 0.0})

    if side == "BUY":
        if st.session_state.cash < cost:
            st.error("Not enough cash to execute this trade.")
            return
        new_qty = pos["qty"] + qty
        if new_qty > 0:
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + cost) / new_qty
        pos["qty"] = new_qty
        positions[ticker] = pos
        st.session_state.cash -= cost
    else:
        if qty > pos["qty"]:
            st.error("Cannot sell more than current position.")
            return
        pos["qty"] -= qty
        st.session_state.cash += cost
        if pos["qty"] <= 0:
            positions.pop(ticker, None)
        else:
            positions[ticker] = pos

    record_trade(side, ticker, qty, price)
    st.success(f"{side} {qty} {ticker} @ {format_inr(price)}")


def render_header(prices: Dict[str, float]) -> None:
    portfolio_value = compute_portfolio_value(prices)
    cols = st.columns([2, 2, 2])
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">{format_inr(portfolio_value)}</div>
                <div class="metric-sub">Live mark‑to‑market across all open positions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Available Cash</div>
                <div class="metric-value">{format_inr(st.session_state.cash)}</div>
                <div class="metric-sub">Simulated INR buying power for new ideas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        nifty_price = prices.get("NIFTY", None)
        if nifty_price is not None:
            nifty_display = f"{nifty_price:,.2f}"
        else:
            nifty_display = "—"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">NIFTY 50 (Spot)</div>
                <div class="metric-value">{nifty_display}</div>
                <div class="metric-sub">Headline barometer for your India exposure</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_watchlist(prices: Dict[str, float]) -> str:
    st.subheader("Watchlist – NSE Large Caps (by Sector)")

    # Allow adding a new listed company to the watchlist
    with st.expander("Add listed company to watchlist"):
        new_ticker = st.text_input("Ticker symbol (e.g. HDFCLIFE)", key="add_ticker").upper().strip()
        new_sector = st.text_input("Sector (e.g. Insurance, FMCG, IT)", key="add_sector").strip()
        if st.button("Add to watchlist", key="add_btn"):
            if not new_ticker:
                st.warning("Please enter a ticker symbol.")
            elif new_ticker in st.session_state.watchlist:
                st.info("Ticker is already in the watchlist.")
            else:
                st.session_state.watchlist.append(new_ticker)
                if new_sector:
                    st.session_state.sectors[new_ticker] = new_sector
                st.success(f"Added {new_ticker} to watchlist.")

    df_rows = []
    for t in st.session_state.watchlist:
        price = prices.get(t)
        qty = st.session_state.positions.get(t, {}).get("qty", 0.0)
        mkt_value = price * qty if price is not None else 0.0
        prev = st.session_state.prev_prices.get(t)
        change_pct = None
        if prev is not None and prev > 0 and price is not None:
            change_pct = (price - prev) / prev * 100.0
        df_rows.append(
            {
                "Ticker": t,
                "Sector": st.session_state.sectors.get(t, "Unknown"),
                "Price (INR)": price,
                "Qty": qty,
                "Value (INR)": mkt_value,
                "Δ % (vs last refresh)": change_pct,
            }
        )

    selected = None
    if df_rows:
        df = pd.DataFrame(df_rows)

        sector_options = sorted(df["Sector"].unique())
        chosen_sectors = st.multiselect(
            "Filter by sector",
            options=sector_options,
            default=sector_options,
        )
        if chosen_sectors:
            df = df[df["Sector"].isin(chosen_sectors)]

        def style_change(val):
            if pd.isna(val):
                return ""
            if val > 0:
                return "color: #22c55e; font-weight: 600;"
            if val < 0:
                return "color: #ef4444; font-weight: 600;"
            return "color: #9ca3af;"

        styled = df.style.format(
            {
                "Price (INR)": "₹{:.2f}",
                "Value (INR)": "₹{:.2f}",
                "Δ % (vs last refresh)": "{:+.2f}%",
            }
        ).map(style_change, subset=["Δ % (vs last refresh)"])

        st.dataframe(styled, use_container_width=True)

        # Keep the selectbox limited to currently visible tickers
        visible_tickers = df["Ticker"].tolist()
        if visible_tickers:
            selected = st.selectbox("Select ticker", visible_tickers)
    else:
        st.info("Watchlist is empty. Add a listed company above.")

    return selected or (st.session_state.watchlist[0] if st.session_state.watchlist else "")


def render_portfolio_overview(prices: Dict[str, float]) -> None:
    """Small visual section: portfolio P&L over time and sector exposure."""
    st.markdown("#### Portfolio Overview")

    # P&L over time
    if st.session_state.portfolio_history:
        hist_df = pd.DataFrame(st.session_state.portfolio_history)
        hist_df = hist_df.set_index("time")
        st.line_chart(hist_df["value"], use_container_width=True)
    else:
        st.info("Portfolio P&L chart will appear once you start trading or refreshing data.")

    # Sector exposure
    sector_values = defaultdict(float)
    for ticker, pos in st.session_state.positions.items():
        price = prices.get(ticker)
        if price is None:
            continue
        sector = st.session_state.sectors.get(ticker, "Unknown")
        sector_values[sector] += pos["qty"] * price

    if sector_values:
        sector_df = pd.DataFrame(
            {"Sector": list(sector_values.keys()), "Value": list(sector_values.values())}
        ).set_index("Sector")
        st.bar_chart(sector_df, use_container_width=True)
    else:
        st.info("Sector exposure will display once you have open positions.")


def render_main_chart(ticker: str) -> None:
    st.subheader(f"{ticker} – Intraday (yfinance)")
    symbol = f"{ticker}.NS"
    data = yf.download(symbol, period="5d", interval="15m", auto_adjust=True, progress=False)
    if data.empty:
        st.info("No chart data available.")
        return
    price_series = data["Close"]
    st.line_chart(price_series, use_container_width=True)


def render_positions_and_trades(prices: Dict[str, float]) -> None:
    st.subheader("Positions")
    rows = []
    for ticker, pos in st.session_state.positions.items():
        price = prices.get(ticker)
        if price is None:
            continue
        qty = pos["qty"]
        avg_cost = pos["avg_cost"]
        mkt_value = qty * price
        pnl = mkt_value - qty * avg_cost
        pnl_pct = (pnl / (qty * avg_cost)) * 100 if qty * avg_cost > 0 else 0.0
        rows.append(
            {
                "Ticker": ticker,
                "Qty": qty,
                "Avg Cost (INR)": avg_cost,
                "Price (INR)": price,
                "Value (INR)": mkt_value,
                "Unrealized P&L (INR)": pnl,
                "P&L %": pnl_pct,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format(
                {
                    "Avg Cost (INR)": "₹{:.2f}",
                    "Price (INR)": "₹{:.2f}",
                    "Value (INR)": "₹{:.2f}",
                    "Unrealized P&L (INR)": "₹{:.2f}",
                    "P&L %": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No open positions yet.")

    st.subheader("Trades")
    if st.session_state.trades:
        trades_df = pd.DataFrame(st.session_state.trades)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades executed yet.")


def call_local_hf_llm(prompt: str, portfolio_summary: str) -> str:
    """
    Call a local Hugging Face text-generation model for portfolio analysis and trade intent.

    The model is asked to:
      1) Respond in natural language.
      2) Optionally emit a JSON line of the form:
         ACTION_JSON: {"trades":[{"side":"BUY","ticker":"HDFCBANK","qty":2}, {"side":"SELL","ticker":"TCS","qty":1}]}
    """
    env_model = os.getenv("LOCAL_LLM_MODEL")
    # Default to an open, transformers-friendly instruct model
    # For an 8GB GPU / CPU-friendly setup, you may want to override
    # this to a smaller model like TinyLlama or Gemma-2B in .env.
    model_id = env_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_token = os.getenv("HF_TOKEN")

    try:
        if st.session_state.hf_pipeline is None or getattr(
            st.session_state.hf_pipeline, "model_name", None
        ) != model_id:
            # Force CPU to avoid CUDA OOM issues
            device_arg = -1

            pipeline_args = {
                "model": model_id,
                "device": device_arg,
            }
            if hf_token:
                pipeline_args["token"] = hf_token

            st.session_state.hf_pipeline = pipeline("text-generation", **pipeline_args)
            # remember which model is loaded
            st.session_state.hf_pipeline.model_name = model_id  # type: ignore[attr-defined]
        gen = st.session_state.hf_pipeline(
            f"""You are FinAlly India, an AI trading copilot focused on Indian cash equities on NSE.
Use INR and Indian tickers (RELIANCE, TCS, HDFCBANK, etc.).
The user trades a virtual portfolio only—no real brokerage.
Give concise, practical analysis in 1–2 short paragraphs; do not provide legal or tax advice.

You must respond **only** to the latest user question given below. Do not invent extra user questions, do not show examples, and do not repeat the instructions.

You can also execute simulated trades on the user's behalf.
If and only if you want to execute trades for this single question, add ONE extra line at the very end of your reply:
ACTION_JSON: {{"trades":[{{"side":"BUY","ticker":"HDFCBANK","qty":2}},{{"side":"SELL","ticker":"TCS","qty":1}}]}}

Rules for ACTION_JSON:
- Use an array 'trades', with 1–4 objects.
- Each trade has exactly these keys: 'side' ("BUY" or "SELL"), 'ticker' (e.g. "HDFCBANK"), 'qty' (integer).
- The trades must correspond only to the current user question.
- Do NOT include any other keys. Do NOT include any text after the JSON on that line.
- If you are unsure what to trade, you may omit the ACTION_JSON line entirely.

Current portfolio snapshot:
{portfolio_summary}

Latest user question:
{prompt}
""",
            max_new_tokens=400,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
        text = gen[0]["generated_text"]
        return text.strip()
    except Exception as exc:
        return f"Local LLM call failed: {exc}"""


def render_chat_panel(prices: Dict[str, float]) -> None:
    st.markdown("#### AI Copilot (Preview)")

    # Simple, text-only chat history
    for msg in st.session_state.chat_messages[-8:]:
        role = "You" if msg["role"] == "user" else "FinAlly India"
        align = "flex-end" if msg["role"] == "user" else "flex-start"
        bg = "rgba(37, 99, 235, 0.18)" if msg["role"] == "user" else "rgba(15, 23, 42, 0.95)"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:{align}; margin-bottom:0.35rem;">
              <div style="max-width:88%; padding:0.45rem 0.6rem; border-radius:0.6rem;
                          background:{bg}; font-size:0.8rem; border:1px solid rgba(148,163,184,0.4);">
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; opacity:0.7; margin-bottom:0.08rem;">
                  {role}
                </div>
                <div>{msg["content"]}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_input = st.text_area(
        "Ask about your INR portfolio, sector exposure, or Indian tickers:",
        key="chat_input",
        height=80,
        placeholder="e.g. How concentrated am I in banks, and what 2–3 NIFTY names could diversify me?",
    )
    if st.button("Ask FinAlly India"):
        question = user_input.strip()
        if not question:
            st.warning("Type a question first.")
            return

        # Build a compact portfolio summary for the model
        rows = []
        for ticker, pos in st.session_state.positions.items():
            price = prices.get(ticker)
            if price is None:
                continue
            value = pos["qty"] * price
            rows.append(
                f"{ticker} ({st.session_state.sectors.get(ticker, 'Unknown')}): "
                f"qty={pos['qty']}, avg_cost={pos['avg_cost']:.2f}, "
                f"price={price:.2f}, value={value:.2f}"
            )
        summary = "\n".join(rows) if rows else "No open positions; 100% in cash."

        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.spinner("FinAlly India (local LLM) is thinking..."):
            answer = call_local_hf_llm(question, summary)

        # Try to detect and execute a structured trade intent from the answer
        executed_note = ""
        if "ACTION_JSON:" in answer:
            try:
                import json

                json_part = answer.split("ACTION_JSON:", 1)[1].strip()
                # Take only the first line after the marker
                json_line = json_part.splitlines()[0].strip()
                action = json.loads(json_line)

                trades_to_run = []

                # New schema: {"trades": [{...}, {...}]}
                if isinstance(action, dict) and isinstance(action.get("trades"), list):
                    for t in action["trades"]:
                        if (
                            isinstance(t, dict)
                            and t.get("side") in {"BUY", "SELL"}
                            and isinstance(t.get("ticker"), str)
                            and isinstance(t.get("qty"), (int, float))
                        ):
                            trades_to_run.append(
                                {
                                    "side": t["side"],
                                    "ticker": t["ticker"].upper(),
                                    "qty": float(t["qty"]),
                                }
                            )

                # Backwards compatibility: single-trade schema
                elif (
                    isinstance(action, dict)
                    and action.get("action") == "trade"
                    and action.get("side") in {"BUY", "SELL"}
                    and isinstance(action.get("ticker"), str)
                    and isinstance(action.get("qty"), (int, float))
                ):
                    trades_to_run.append(
                        {
                            "side": action["side"],
                            "ticker": action["ticker"].upper(),
                            "qty": float(action["qty"]),
                        }
                    )

                # Execute all parsed trades sequentially
                if trades_to_run:
                    notes = []
                    for t in trades_to_run:
                        ticker = t["ticker"]
                        qty = t["qty"]
                        side = t["side"]
                        # Use current price snapshot or fetch on demand
                        price = prices.get(ticker)
                        if price is None:
                            fetched = fetch_prices([ticker])
                            price = fetched.get(ticker)
                        if price is not None:
                            execute_trade(side, ticker, qty, price)
                            notes.append(
                                f"{side} {qty:g} {ticker} @ {format_inr(price)}"
                            )
                        else:
                            notes.append(
                                f"{side} {qty:g} {ticker} (no live price, trade skipped)"
                            )
                    executed_note = (
                        "\n\n(Executed trades: " + "; ".join(notes) + ")"
                    )
            except Exception:
                executed_note = "\n\n(Failed to parse trade intent from AI response.)"

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer + executed_note}
        )
        # Trigger a fresh render so the latest AI message appears at the top of the panel
        st.rerun()


def render_trade_ticket(selected_ticker: str, prices: Dict[str, float]) -> None:
    st.markdown("**Side & Ticker**")
    side = st.selectbox("Side", ["BUY", "SELL"], key="trade_side")
    ticker = st.text_input(
        "Ticker",
        value=selected_ticker or "RELIANCE",
        key="trade_ticker",
    ).upper()

    qty = st.number_input("Quantity", min_value=1.0, step=1.0, key="trade_qty")

    current_price = prices.get(ticker)
    if current_price is not None:
        st.markdown(f"Current price: **{format_inr(current_price)}**")
    else:
        st.warning("No live price available; will try to fetch on submit.")

    if st.button("Submit Order", key="trade_submit"):
        price = current_price
        if price is None:
            fetched = fetch_prices([ticker])
            price = fetched.get(ticker)
        if price is None:
            st.error("Could not fetch price for this ticker.")
            return
        execute_trade(side, ticker, qty, price)


def main() -> None:
    # Disable Streamlit's module file-watcher to avoid torch.classes path inspection issues
    #st.set_option("server.fileWatcherType", "none")

    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global theming tweaks for a more professional dark terminal look
    st.markdown(
        """
        <style>
        body {
            background-color: #020617;
        }
        .main {
            background: radial-gradient(circle at top left, #020617 0, #020617 40%, #000000 100%);
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #020617 60%, #030712 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.25);
        }
        h1, h2, h3 {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
            letter-spacing: 0.02em;
        }
        /* Top KPI tiles */
        .metric-card {
            padding: 1.0rem 1.2rem;
            border-radius: 0.9rem;
            background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.7));
            border: 1px solid rgba(51, 65, 85, 0.9);
            box-shadow:
                0 22px 55px rgba(15, 23, 42, 0.85),
                0 0 0 1px rgba(15, 23, 42, 0.9);
        }
        .metric-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 650;
            color: #f9fafb;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #9ca3af;
        }
        /* Header pills */
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 500;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid rgba(148, 163, 184, 0.55);
            color: #e5e7eb;
            gap: 0.3rem;
            background: radial-gradient(circle at top left, rgba(55, 65, 81, 0.8), rgba(15, 23, 42, 0.9));
        }
        .pill-primary {
            border-color: rgba(32, 157, 215, 0.9);
            box-shadow: 0 0 0 1px rgba(32, 157, 215, 0.35);
        }
        .pill-accent {
            border-color: rgba(236, 173, 10, 0.9);
            box-shadow: 0 0 0 1px rgba(236, 173, 10, 0.3);
        }
        .pill-quiet {
            opacity: 0.9;
        }
        /* Sector tags & badges */
        .sector-tag {
            display: inline-flex;
            align-items: center;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 500;
            border: 1px solid rgba(148, 163, 184, 0.5);
            background: rgba(15, 23, 42, 0.96);
            color: #e5e7eb;
        }
        .sector-tag span {
            opacity: 0.85;
        }
        .sector-it { border-color: rgba(32, 157, 215, 0.9); }
        .sector-bank { border-color: rgba(34, 197, 94, 0.9); }
        .sector-fmcg { border-color: rgba(236, 173, 10, 0.95); }
        .sector-energy { border-color: rgba(248, 113, 113, 0.9); }
        .sector-auto { border-color: rgba(94, 234, 212, 0.9); }
        .sector-default { border-color: rgba(148, 163, 184, 0.85); }
        /* Section titles */
        .watchlist-title {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .watchlist-title h3 {
            margin-bottom: 0.1rem;
        }
        .small-caption {
            font-size: 0.75rem;
            color: #9ca3af;
        }
        /* Dataframe containers */
        .stDataFrame {
            border-radius: 0.9rem;
            overflow: hidden;
            border: 1px solid rgba(31, 41, 55, 0.9);
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.9);
            background-color: rgba(15, 23, 42, 0.98);
        }
        /* Generic panel card for right-side tiles */
        .panel-card {
            padding: 0.75rem 0.9rem 0.9rem 0.9rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(51, 65, 85, 0.9);
            background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.9));
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.9);
        }
        .panel-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #9ca3af;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_state()

    refresh_seconds = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 15)
    st.sidebar.write(
        "FinAlly India is your AI‑ready trading cockpit for Indian cash equities.\n\n"
        "- Live NSE watchlist\n"
        "- Virtual INR trading\n"
        "- Sector‑aware positions view\n\n"
        "Prices via yfinance; trading is fully simulated."
    )

    prices = fetch_prices(st.session_state.watchlist)

    # Update price history for flash calculations and portfolio history for charts
    update_portfolio_history(prices)
    st.session_state.prev_prices = dict(prices)

    # Hero header with product positioning
    col_logo, col_title = st.columns([0.9, 5.1])
    with col_logo:
        st.markdown(
            """
            <div class="pill pill-primary">
                <span>FINALLY INDIA</span>
                <span style="opacity:0.6;">ALPHA DESK</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown(
            """
            <h1 style="margin-bottom:0.15rem;">AI Trading Workstation for NSE</h1>
            <p style="font-size:0.85rem;color:#9ca3af;margin-top:0;">
                Stream live Indian equity prices, manage a virtual ₹ portfolio, and prepare for an AI trading copilot – all in one professional terminal.
            </p>
            """,
            unsafe_allow_html=True,
        )

    render_header(prices)

    # Layout similar to FinAlly Terminal:
    # - Left: tall watchlist grid
    # - Right: stacked tiles (main chart, portfolio panels, positions + trade & chat)
    col_left, col_right = st.columns([2.2, 3.0])
    with col_left:
        selected_ticker = render_watchlist(prices)
    with col_right:
        # Top row: main chart
        with st.container():
            st.markdown(
                '<div class="panel-card"><div class="panel-title">Main Chart</div>',
                unsafe_allow_html=True,
            )
            if selected_ticker:
                render_main_chart(selected_ticker)
            else:
                st.info("Select a ticker from the watchlist to view its chart.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Middle row: portfolio overview panels (P&L + sector exposure)
        mid_col1, mid_col2 = st.columns(2)
        with mid_col1:
            st.markdown(
                '<div class="panel-card"><div class="panel-title">Portfolio Overview</div>',
                unsafe_allow_html=True,
            )
            render_portfolio_overview(prices)
            st.markdown("</div>", unsafe_allow_html=True)
        with mid_col2:
            st.markdown(
                '<div class="panel-card"><div class="panel-title">AI Copilot</div>',
                unsafe_allow_html=True,
            )
            render_chat_panel(prices)
            st.markdown("</div>", unsafe_allow_html=True)

        # Bottom row: positions + trade bar
        bottom_left, bottom_right = st.columns([2.0, 1.3])
        with bottom_left:
            st.markdown(
                '<div class="panel-card"><div class="panel-title">Positions & Trades</div>',
                unsafe_allow_html=True,
            )
            render_positions_and_trades(prices)
            st.markdown("</div>", unsafe_allow_html=True)
        with bottom_right:
            st.markdown(
                '<div class="panel-card"><div class="panel-title">Trade Bar</div>',
                unsafe_allow_html=True,
            )
            render_trade_ticket(selected_ticker, prices)
            st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"Last updated: **{datetime.now().strftime('%H:%M:%S')}**\n\n"
        "Use Streamlit's rerun or the auto-refresh slider to update prices."
    )


if __name__ == "__main__":
    main()

