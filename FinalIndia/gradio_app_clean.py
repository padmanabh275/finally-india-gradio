import os
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import torch
import yfinance as yf
from transformers import AutoModelForCausalLM, AutoTokenizer


APP_TITLE = "FinAlly India – Gradio Terminal (Clean)"
STARTING_CASH = 1_000_000.0

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


STATE = {
    "cash": STARTING_CASH,
    "positions": {},  # ticker -> {"qty": float, "avg_cost": float}
    "trades": [],
    "portfolio_history": [],
    "chat": [],
}

WATCHLIST: List[str] = list(DEFAULT_TICKERS)

HF_CHAT_MODEL = None
HF_CHAT_TOKENIZER = None
HF_MODEL_ID: str | None = None


def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"


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

    if len(data) > 0 and "Close" in data:
        last_row = data["Close"].iloc[-1]
        for t, ysym in zip(tickers, yahoo_symbols):
            if ysym in last_row.index:
                prices[t] = float(last_row[ysym])
    return prices


def compute_portfolio_value(prices: Dict[str, float]) -> float:
    total = STATE["cash"]
    for ticker, pos in STATE["positions"].items():
        price = prices.get(ticker)
        if price is not None:
            total += pos["qty"] * price
    return total


def update_portfolio_history(prices: Dict[str, float]) -> pd.DataFrame:
    value = compute_portfolio_value(prices)
    STATE["portfolio_history"].append({"time": datetime.now(), "value": value})
    if len(STATE["portfolio_history"]) > 200:
        STATE["portfolio_history"] = STATE["portfolio_history"][-200:]
    return pd.DataFrame(STATE["portfolio_history"])


def execute_trade(side: str, ticker: str, qty: float, price: float) -> str:
    if qty <= 0:
        return "Quantity must be positive."

    positions = STATE["positions"]
    pos = positions.get(ticker, {"qty": 0.0, "avg_cost": 0.0})
    cost = qty * price

    if side == "BUY":
        if STATE["cash"] < cost:
            return "Not enough cash to execute this trade."
        new_qty = pos["qty"] + qty
        if new_qty > 0:
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + cost) / new_qty
        pos["qty"] = new_qty
        positions[ticker] = pos
        STATE["cash"] -= cost
    else:
        if qty > pos["qty"]:
            return "Cannot sell more than current position."
        pos["qty"] -= qty
        STATE["cash"] += cost
        if pos["qty"] <= 0:
            positions.pop(ticker, None)
        else:
            positions[ticker] = pos

    STATE["trades"].append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "side": side,
            "ticker": ticker,
            "qty": qty,
            "price": price,
        }
    )
    return f"{side} {qty:g} {ticker} @ {format_inr(price)}"


def build_watchlist_df(prices: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for t in WATCHLIST:
        price = prices.get(t)
        pos = STATE["positions"].get(t, {"qty": 0.0})
        qty = float(pos["qty"])
        if price is not None:
            price_val = round(float(price), 2)
            value = round(price_val * qty, 2)
        else:
            price_val = None
            value = 0.0
        rows.append(
            {"Ticker": t, "Price (INR)": price_val, "Qty": qty, "Value (INR)": value}
        )
    return pd.DataFrame(rows)


def build_positions_df(prices: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for ticker, pos in STATE["positions"].items():
        price = prices.get(ticker)
        if price is None:
            continue
        qty = pos["qty"]
        avg_cost = pos["avg_cost"]
        value = qty * price
        pnl = value - qty * avg_cost
        pnl_pct = (pnl / (qty * avg_cost)) * 100 if qty * avg_cost > 0 else 0.0
        rows.append(
            {
                "Ticker": ticker,
                "Qty": qty,
                "Avg Cost (INR)": avg_cost,
                "Price (INR)": price,
                "Value (INR)": value,
                "Unrealized P&L (INR)": pnl,
                "P&L %": pnl_pct,
            }
        )
    return pd.DataFrame(rows)


def build_sector_df(prices: Dict[str, float]) -> pd.DataFrame:
    """
    Simple sector/position-size view: aggregate by ticker (one 'Equity' sector).
    Later this can be extended with real sector tags.
    """
    rows = []
    for ticker, pos in STATE["positions"].items():
        price = prices.get(ticker)
        if price is None:
            continue
        value = pos["qty"] * price
        rows.append({"Ticker": ticker, "Sector": "Equity", "Value": value})
    return pd.DataFrame(rows)


def get_chat_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global HF_CHAT_MODEL, HF_CHAT_TOKENIZER, HF_MODEL_ID

    env_model = os.getenv("LOCAL_LLM_MODEL")
    # Default to a smaller, CPU-friendly chat model
    model_id = env_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_token = os.getenv("HF_TOKEN")

    if HF_CHAT_MODEL is not None and HF_MODEL_ID == model_id:
        return HF_CHAT_MODEL, HF_CHAT_TOKENIZER

    token_kwargs = {}
    # Force CPU usage
    model_kwargs = {"torch_dtype": torch.float32}
    if hf_token:
        token_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_id, **token_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    HF_CHAT_MODEL = model
    HF_CHAT_TOKENIZER = tokenizer
    HF_MODEL_ID = model_id
    return model, tokenizer


def call_llm(message: str, snapshot: str) -> str:
    try:
        model, tokenizer = get_chat_model()
    except Exception as exc:
        return f"(LLM init failed: {exc})"

    system = (
        "You are FinAlly India, an AI trading copilot for Indian NSE cash equities. "
        "Use INR and Indian tickers only. Trading is virtual only. "
        "Reply in at most 3 short sentences.\n\n"
        "If you want to execute trades, add one line at the very end:\n"
        'ACTION_JSON: {"trades":[{"side":"BUY","ticker":"HDFCBANK","qty":2}]}\n\n'
        "Current portfolio snapshot:\n"
        f"{snapshot}\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in text:
            return text.split("<|assistant|>")[-1].strip()
        return text.strip()
    except Exception as exc:
        return f"(LLM call failed: {exc})"


def infer_trades_from_text(text: str) -> List[Dict[str, object]]:
    import re

    pattern = r"\b(buy|sell)\s+(\d+)\s+(?:shares?\s+of\s+)?([a-zA-Z0-9_.-]+)"
    matches = re.findall(pattern, text.lower())
    trades: List[Dict[str, object]] = []
    for side, qty_str, ticker in matches:
        try:
            qty = float(qty_str)
        except ValueError:
            continue
        trades.append(
            {"side": side.upper(), "ticker": ticker.upper(), "qty": qty}
        )
    return trades


def refresh_all() -> Tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = fetch_prices(WATCHLIST)
    hist_df = update_portfolio_history(prices)
    watch_df = build_watchlist_df(prices)
    pos_df = build_positions_df(prices)
    sector_df = build_sector_df(prices)
    port_val = format_inr(compute_portfolio_value(prices))
    cash_val = format_inr(STATE["cash"])
    return port_val, cash_val, watch_df, pos_df, hist_df, sector_df


def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    """
    Fetch recent price history for a single NSE ticker for the main chart.
    Uses last 5 days of 15m data via yfinance.
    """
    if not ticker:
        return pd.DataFrame({"time": [], "price": []})
    symbol = f"{ticker}.NS"
    data = yf.download(
        symbol,
        period="5d",
        interval="15m",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame({"time": [], "price": []})

    # Normalise Close column (handle potential MultiIndex from yfinance)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = close.reset_index()
    df.columns = ["time", "price"]
    return df


def fetch_ticker_sparkline(ticker: str) -> pd.DataFrame:
    """
    Lightweight sparkline: last 1 day of 5m data for the selected ticker.
    """
    if not ticker:
        return pd.DataFrame({"time": [], "price": []})
    symbol = f"{ticker}.NS"
    data = yf.download(
        symbol,
        period="1d",
        interval="5m",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame({"time": [], "price": []})

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = close.reset_index()
    df.columns = ["time", "price"]
    return df


def on_trade(side: str, ticker: str, qty: float):
    prices = fetch_prices([ticker])
    price = prices.get(ticker)
    if price is None:
        msg = f"Could not fetch price for {ticker}."
    else:
        msg = execute_trade(side, ticker, qty, price)
    port_val, cash_val, watch_df, pos_df, hist_df, sector_df = refresh_all()
    return (
        msg,
        f"**Portfolio Value**: {port_val}",
        f"**Cash**: {cash_val}",
        watch_df,
        pos_df,
        hist_df,
        sector_df,
    )


def on_chat(message: str):
    prices = fetch_prices(WATCHLIST)
    snapshot_lines = []
    for ticker, pos in STATE["positions"].items():
        price = prices.get(ticker)
        if price is None:
            continue
        value = pos["qty"] * price
        snapshot_lines.append(
            f"{ticker}: qty={pos['qty']}, avg_cost={pos['avg_cost']:.2f}, price={price:.2f}, value={value:.2f}"
        )
    snapshot = "\n".join(snapshot_lines) if snapshot_lines else "No open positions; 100% in cash."

    answer = call_llm(message, snapshot)
    notes = []
    for t in infer_trades_from_text(message):
        prices_now = fetch_prices([t["ticker"]])
        price = prices_now.get(t["ticker"])
        if price is not None:
            res = execute_trade(t["side"], t["ticker"], t["qty"], price)
            notes.append(res)

    STATE["chat"].append(("YOU", message))
    if notes:
        STATE["chat"].append(("FINALLY", answer + "\n\n" + "\n".join(notes)))
    else:
        STATE["chat"].append(("FINALLY", answer))

    chat_log = "\n\n".join(f"{r}: {c}" for r, c in STATE["chat"][-10:])
    port_val, cash_val, watch_df, pos_df, hist_df, sector_df = refresh_all()
    return (
        chat_log,
        f"**Portfolio Value**: {port_val}",
        f"**Cash**: {cash_val}",
        watch_df,
        pos_df,
        hist_df,
        sector_df,
    )


def on_add_ticker(new_ticker: str):
    t = (new_ticker or "").strip().upper()
    if not t:
        msg = "Please enter a ticker symbol."
    elif t in WATCHLIST:
        msg = f"{t} is already in the watchlist."
    else:
        WATCHLIST.append(t)
        msg = f"Added {t} to watchlist."
    prices = fetch_prices(WATCHLIST)
    watch_df = build_watchlist_df(prices)
    return msg, watch_df, gr.update(choices=WATCHLIST, value=t or WATCHLIST[0])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            """
            <div style="font-size:0.75rem;letter-spacing:0.2em;text-transform:uppercase;color:#9ca3af;">
              FINALLY INDIA  AI TRADING WORKSTATION
            </div>
            <h1 style="margin-bottom:0.1rem;">FinAlly India Terminal</h1>
            <p style="font-size:0.85rem;color:#9ca3af;margin-top:0;">
              Live NSE watchlist, virtual ₹ portfolio, and an AI copilot you can trade with.
            </p>
            """,
        )

        # KPI row
        with gr.Row():
            port_value = gr.Markdown()
            cash_value = gr.Markdown()
            index_value = gr.Markdown("**NIFTY 50**: —")

        # Main layout
        with gr.Row():
            # Left: watchlist
            with gr.Column(scale=2):
                gr.Markdown("#### Watchlist")
                watch_df = gr.Dataframe(
                    headers=["Ticker", "Price (INR)", "Qty", "Value (INR)"],
                    interactive=False,
                    row_count="dynamic",
                )
                new_ticker = gr.Textbox(label="Add NSE ticker (e.g. HDFCLIFE)")
                add_btn = gr.Button("Add to Watchlist")
                add_msg = gr.Markdown()
                # Sparkline for selected ticker
                gr.Markdown("#### Ticker Sparkline (1d / 5m)")
                sparkline_plot = gr.LinePlot(
                    x="time",
                    y="price",
                    label="Sparkline",
                    height=150,
                )
                # AI Copilot stacked under sparkline
                gr.Markdown("#### AI Copilot")
                chat_history = gr.Textbox(
                    lines=10, label="", interactive=False
                )
                chat_input = gr.Textbox(label="Ask FinAlly India")
                chat_send = gr.Button("Send")

            # Right: charts + positions + trade/chat
            with gr.Column(scale=3):
                # Main price chart for selected ticker
                with gr.Row():
                    selected_ticker = gr.Dropdown(
                        WATCHLIST, value=WATCHLIST[0], label="Main Chart Ticker"
                    )
                with gr.Row():
                    main_chart = gr.LinePlot(
                        x="time",
                        y="price",
                        label="Main Price Chart",
                    )
                # Portfolio P&L
                with gr.Row():
                    hist_plot = gr.LinePlot(x="time", y="value", label="Portfolio P&L")
                # Sector / ticker exposure
                with gr.Row():
                    sector_plot = gr.BarPlot(
                        x="Ticker",
                        y="Value",
                        label="Portfolio by Ticker",
                    )
                with gr.Row():
                    positions_df = gr.Dataframe(
                        headers=[
                            "Ticker",
                            "Qty",
                            "Avg Cost (INR)",
                            "Price (INR)",
                            "Value (INR)",
                            "Unrealized P&L (INR)",
                            "P&L %",
                        ],
                        label="Positions",
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Trade Bar")
                        side = gr.Radio(["BUY", "SELL"], value="BUY", label="Side")
                        ticker = gr.Dropdown(WATCHLIST, value=WATCHLIST[0], label="Ticker")
                        qty = gr.Slider(1, 200, value=10, step=1, label="Quantity")
                        trade_btn = gr.Button("Submit Trade")
                        trade_result = gr.Markdown()

        # Wiring
        # Initial portfolio refresh on load
        demo.load(
            fn=lambda: refresh_all(),
            inputs=[],
            outputs=[port_value, cash_value, watch_df, positions_df, hist_plot, sector_plot],
        )

        # Load main chart data when ticker changes or on startup
        demo.load(
            fn=lambda: fetch_ticker_history(WATCHLIST[0]),
            inputs=[],
            outputs=[main_chart],
        )
        selected_ticker.change(
            fn=fetch_ticker_history,
            inputs=[selected_ticker],
            outputs=[main_chart],
        )

        # Sparkline sync with selected ticker
        demo.load(
            fn=lambda: fetch_ticker_sparkline(WATCHLIST[0]),
            inputs=[],
            outputs=[sparkline_plot],
        )
        selected_ticker.change(
            fn=fetch_ticker_sparkline,
            inputs=[selected_ticker],
            outputs=[sparkline_plot],
        )

        trade_btn.click(
            fn=on_trade,
            inputs=[side, ticker, qty],
            outputs=[trade_result, port_value, cash_value, watch_df, positions_df, hist_plot, sector_plot],
        )
        chat_send.click(
            fn=on_chat,
            inputs=[chat_input],
            outputs=[chat_history, port_value, cash_value, watch_df, positions_df, hist_plot, sector_plot],
        )
        add_btn.click(
            fn=on_add_ticker,
            inputs=[new_ticker],
            outputs=[add_msg, watch_df, ticker],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        css="""
        body { background-color: #020617; color: #e5e7eb; }
        .gradio-container { max-width: 1400px !important; margin: 0 auto; }

        /* Panels: square, flat, no clipping */
        .block, .gr-box, .gr-panel {
            border-radius: 0px !important;
            border: 1px solid rgba(55, 65, 81, 0.7) !important;
            background-color: #020617 !important;
            box-shadow: none !important;
            overflow: visible !important;
        }

        /* Inputs */
        textarea, input, select {
            border-radius: 4px !important;
            border-color: rgba(75, 85, 99, 0.9) !important;
            background-color: #020617 !important;
            color: #e5e7eb !important;
        }

        /* Buttons */
        button {
            border-radius: 4px !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }
        button.primary, button:hover {
            background: linear-gradient(135deg, #22c55e, #0ea5e9) !important;
            border: none !important;
            color: #020617 !important;
        }

        /* Dataframes (watchlist / positions) - clear, flat table look */
        table {
            font-size: 0.86rem !important;
        }
        thead tr th {
            background-color: #0b1120 !important;
            color: #e5e7eb !important;
            font-weight: 600 !important;
        }
        tbody tr:nth-child(odd) {
            background-color: #020617 !important;
        }
        tbody tr:nth-child(even) {
            background-color: #030712 !important;
        }
        tbody tr:hover {
            background-color: #1d283a !important;
        }
        td, th {
            padding: 6px 10px !important;
            border-color: rgba(31, 41, 55, 0.9) !important;
            border-radius: 0px !important;
        }

        /* Accent colors for small tags (future use) */
        .accent-green { color: #22c55e; }
        .accent-red { color: #ef4444; }
        .accent-blue { color: #38bdf8; }
        """
    )

