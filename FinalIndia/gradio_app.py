import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import torch
import yfinance as yf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


APP_TITLE = "FinAlly India – Gradio Terminal"
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

WATCHLIST: List[str] = list(DEFAULT_TICKERS)

STATE = {
    "cash": STARTING_CASH,
    "positions": {},  # ticker -> {"qty": float, "avg_cost": float}
    "trades": [],  # list of dicts
    "portfolio_history": [],  # list of {"time": datetime, "value": float}
    "chat": [],  # list of (role, content)
}

HF_PIPELINE = None  # still used for non-chat fallbacks if desired
HF_MODEL_ID = None
HF_CHAT_MODEL = None
HF_CHAT_TOKENIZER = None


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
    else:
        for t, ysym in zip(tickers, yahoo_symbols):
            tkr = yf.Ticker(ysym)
            info = tkr.info or {}
            price = info.get("regularMarketPrice")
            if price is not None:
                prices[t] = float(price)

    return prices


def compute_portfolio_value(prices: Dict[str, float]) -> float:
    total = STATE["cash"]
    for ticker, pos in STATE["positions"].items():
        price = prices.get(ticker)
        if price is not None:
            total += pos["qty"] * price
    return total


def update_portfolio_history(prices: Dict[str, float]) -> None:
    value = compute_portfolio_value(prices)
    STATE["portfolio_history"].append({"time": datetime.now(), "value": value})
    if len(STATE["portfolio_history"]) > 500:
        STATE["portfolio_history"] = STATE["portfolio_history"][-500:]


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
    else:  # SELL
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
            "value": qty * price * (1 if side == "BUY" else -1),
        }
    )
    return f"{side} {qty:g} {ticker} @ {format_inr(price)}"


def get_hf_pipeline() -> Tuple[object, str]:
    global HF_PIPELINE, HF_MODEL_ID
    env_model = os.getenv("LOCAL_LLM_MODEL")
    model_id = env_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_token = os.getenv("HF_TOKEN")

    if HF_PIPELINE is not None and HF_MODEL_ID == model_id:
        return HF_PIPELINE, HF_MODEL_ID

    device_arg = -1  # CPU (safer for Spaces)
    pipeline_args = {"model": model_id, "device": device_arg}
    if hf_token:
        pipeline_args["token"] = hf_token

    HF_PIPELINE = pipeline("text-generation", **pipeline_args)
    HF_MODEL_ID = model_id
    return HF_PIPELINE, HF_MODEL_ID


def get_chat_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    Load a chat-capable model + tokenizer and remember them globally.

    Uses LOCAL_LLM_MODEL or defaults to TinyLlama Chat.
    Runs on CPU (device_map not used here) to be HF Spaces friendly.
    """
    global HF_CHAT_MODEL, HF_CHAT_TOKENIZER, HF_MODEL_ID

    env_model = os.getenv("LOCAL_LLM_MODEL")
    # Default to a smaller, CPU-friendly chat model
    model_id = env_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_token = os.getenv("HF_TOKEN")

    if HF_CHAT_MODEL is not None and HF_MODEL_ID == model_id:
        return HF_CHAT_MODEL, HF_CHAT_TOKENIZER, model_id

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
    return HF_CHAT_MODEL, HF_CHAT_TOKENIZER, model_id


def infer_trades_from_text(text: str) -> List[Dict[str, object]]:
    """
    Fallback: infer simple BUY/SELL intent directly from user text.

    Supports patterns like:
      "buy 10 HDFCBANK"
      "sell 5 tcs shares"
    """
    import re

    text_low = text.lower()
    # Support patterns like:
    #  - buy 10 reliance
    #  - buy 10 shares of reliance
    #  - sell 5 tcs shares
    pattern = r"\b(buy|sell)\s+(\d+)\s+(?:shares?\s+of\s+)?([a-zA-Z0-9_.-]+)"
    matches = re.findall(pattern, text_low)
    trades: List[Dict[str, object]] = []
    for side, qty_str, ticker in matches:
        try:
            qty = float(qty_str)
        except ValueError:
            continue
        trades.append(
            {
                "side": side.upper(),
                "ticker": ticker.upper(),
                "qty": qty,
            }
        )
    return trades


def call_llm(prompt: str, portfolio_summary: str) -> str:
    """
    Local HF LLM call using the model's chat template.

    Keeps the same ACTION_JSON schema as the Streamlit app.
    """
    try:
        model, tokenizer, model_id = get_chat_model()
    except Exception as exc:  # pragma: no cover - runtime only
        return f"Local LLM initialisation failed: {exc}"

    system_content = (
        "You are FinAlly India, an AI trading copilot for Indian NSE cash equities.\n"
        "- Use INR and Indian tickers (RELIANCE, TCS, HDFCBANK, etc.).\n"
        "- Trading is virtual only — no real brokerage.\n"
        "- Reply in at most 3 short sentences.\n"
        "- Do NOT restate these rules or the full portfolio snapshot.\n\n"
        "You MAY execute simulated trades for the current question. "
        "If and only if you decide to trade, add ONE extra line at the very end of your reply:\n"
        'ACTION_JSON: {"trades":[{"side":"BUY","ticker":"HDFCBANK","qty":2}]}\n\n'
        "Rules for ACTION_JSON:\n"
        "- Use an array 'trades' with 1–4 objects.\n"
        '- Each trade has: \"side\" (\"BUY\" or \"SELL\"), \"ticker\" (e.g. \"HDFCBANK\"), \"qty\" (integer).\n'
        "- Only include trades that match the current user request.\n"
        "- No other keys. No text after the JSON on that line.\n"
        "- If you are unsure, omit the ACTION_JSON line entirely.\n\n"
        f"Current portfolio snapshot (for your reference only):\n{portfolio_summary}\n"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Heuristic: return only the part after the last assistant tag if present
        if "<|assistant|>" in full_text:
            return full_text.split("<|assistant|>")[-1].strip()
        return full_text.strip()
    except Exception as exc:  # pragma: no cover
        return f"Local LLM call failed: {exc}"


def build_watchlist_table(prices: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for t in WATCHLIST:
        price = prices.get(t)
        pos = STATE["positions"].get(t, {"qty": 0.0, "avg_cost": 0.0})
        qty = float(pos["qty"])
        if price is not None:
            price_val = round(float(price), 2)
            value = round(price_val * qty, 2)
        else:
            price_val = None
            value = 0.0
        rows.append(
            {
                "Ticker": t,
                "Price (INR)": price_val,
                "Qty": qty,
                "Value (INR)": value,
            }
        )
    df = pd.DataFrame(rows)
    return df


def build_positions_table(prices: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for ticker, pos in STATE["positions"].items():
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
    return pd.DataFrame(rows)


def build_portfolio_charts(prices: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # P&L over time
    if not STATE["portfolio_history"]:
        update_portfolio_history(prices)
    hist_df = pd.DataFrame(STATE["portfolio_history"])

    # Sector exposure – simplified, sectors not tracked here; treat all as "Equity"
    sector_values = defaultdict(float)
    for ticker, pos in STATE["positions"].items():
        price = prices.get(ticker)
        if price is None:
            continue
        sector_values["Equity"] += pos["qty"] * price

    sector_df = (
        pd.DataFrame({"Sector": list(sector_values.keys()), "Value": list(sector_values.values())})
        if sector_values
        else pd.DataFrame({"Sector": [], "Value": []})
    )
    return hist_df, sector_df


def gradio_refresh(selected_ticker: str) -> Tuple[str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = fetch_prices(WATCHLIST)
    update_portfolio_history(prices)
    df_watch = build_watchlist_table(prices)
    df_pos = build_positions_table(prices)
    hist_df, sector_df = build_portfolio_charts(prices)
    portfolio_value = compute_portfolio_value(prices)
    port_value = f"**Portfolio Value**: {format_inr(portfolio_value)}"
    cash_text = f"**Cash**: {format_inr(STATE['cash'])}"
    index_text = "**NIFTY 50**: —"  # placeholder; can be wired later
    return port_value, cash_text, index_text, df_watch, df_pos, hist_df, sector_df


def gradio_trade(side: str, ticker: str, qty: float) -> str:
    prices = fetch_prices([ticker])
    price = prices.get(ticker)
    if price is None:
        return f"Could not fetch price for {ticker}."
    return execute_trade(side, ticker, qty, price)


def add_to_watchlist(ticker: str):
    """Add a new ticker symbol to the watchlist and refresh the table and dropdown."""
    t = (ticker or "").strip().upper()
    if not t:
        msg = "Please enter a ticker symbol."
    elif t in WATCHLIST:
        msg = f"{t} is already in the watchlist."
    else:
        WATCHLIST.append(t)
        msg = f"Added {t} to watchlist."

    prices = fetch_prices(WATCHLIST)
    df_watch = build_watchlist_table(prices)
    # Update dropdown choices to include the new symbol (keep current value if still valid)
    return msg, df_watch, gr.update(choices=WATCHLIST, value=t or WATCHLIST[0])


def trade_and_refresh(side: str, ticker: str, qty: float):
    """Execute a trade, then refresh all KPIs, tables, and charts."""
    msg = gradio_trade(side, ticker, qty)
    port_value, cash_text, index_text, df_watch, df_pos, hist_df, sector_df = gradio_refresh(ticker)
    return msg, port_value, cash_text, index_text, df_watch, df_pos, hist_df, sector_df


def gradio_chat(message: str) -> Tuple[
    str,
    str,
    str,
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    prices = fetch_prices(DEFAULT_TICKERS)
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
    STATE["chat"].append(("you", message))
    executed_note = ""
    executed_any = False

    # First try structured ACTION_JSON from the model
    if "ACTION_JSON:" in answer:
        try:
            import json

            json_part = answer.split("ACTION_JSON:", 1)[1].strip()
            json_line = json_part.splitlines()[0].strip()
            action = json.loads(json_line)
            trades_to_run = []

            if isinstance(action, dict) and isinstance(action.get("trades"), list):
                for t in action["trades"]:
                    if (
                        isinstance(t, dict)
                        and t.get("side") in {"BUY", "SELL"}
                        and isinstance(t.get("ticker"), str)
                        and isinstance(t.get("qty"), (int, float))
                    ):
                        trades_to_run.append(
                            {"side": t["side"], "ticker": t["ticker"].upper(), "qty": float(t["qty"])}
                        )

            notes = []
            for t in trades_to_run:
                prices_now = fetch_prices([t["ticker"]])
                price = prices_now.get(t["ticker"])
                if price is not None:
                    res = execute_trade(t["side"], t["ticker"], t["qty"], price)
                    notes.append(res)
            if notes:
                executed_note = "\n\n" + "\n".join(notes)
                executed_any = True
        except Exception:
            # Ignore parse failure here; we'll fall back to regex-based intent
            pass

    # If no structured trade executed, infer directly from the user message
    if not executed_any:
        inferred = infer_trades_from_text(message)
        notes = []
        for t in inferred:
            prices_now = fetch_prices([t["ticker"]])
            price = prices_now.get(t["ticker"])
            if price is not None:
                res = execute_trade(t["side"], t["ticker"], t["qty"], price)
                notes.append(res)
        if notes:
            executed_note = "\n\n" + "\n".join(notes)

    # Keep assistant responses concise in the UI: only last ~3 lines
    full_answer = answer + executed_note
    concise = "\n".join(full_answer.splitlines()[-3:]) if full_answer else full_answer

    STATE["chat"].append(("finlly india", concise))
    chat_text = "\n\n".join(f"{r.upper()}: {c}" for r, c in STATE["chat"])

    # After any trades, refresh KPIs, tables, and charts
    # Use first watchlist symbol as selected_ticker placeholder
    selected = WATCHLIST[0] if WATCHLIST else ""
    port_value, cash_text, index_text, df_watch, df_pos, hist_df, sector_df = gradio_refresh(
        selected
    )

    return (
        chat_text,
        concise,
        port_value,
        cash_text,
        index_text,
        df_watch,
        df_pos,
        hist_df,
        sector_df,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
    ) as demo:
        # Header + hero text
        gr.Markdown(
            """
            <div class="pill pill-primary">FINALLY INDIA&nbsp;&nbsp;<span style="opacity:0.7;">AI TRADING WORKSTATION</span></div>
            <h1 style="margin-bottom:0.15rem;">FinAlly India Terminal</h1>
            <p style="font-size:0.85rem;color:#9ca3af;margin-top:0;">
              Live NSE watchlist, virtual ₹ portfolio, and an AI copilot.
            </p>
            """,
        )

        # KPI row
        with gr.Row():
            port_value = gr.Markdown()
            cash_value = gr.Markdown()
            index_value = gr.Markdown()

        # Main body: left watchlist, right stack of tiles
        with gr.Row():
            # Left: Watchlist
            with gr.Column(scale=3):
                watch_df = gr.Dataframe(
                    headers=["Ticker", "Price (INR)", "Qty", "Value (INR)"],
                    row_count=(len(DEFAULT_TICKERS), "fixed"),
                    interactive=False,
                    wrap=True,
                    label="Watchlist",
                )
                new_ticker = gr.Textbox(label="Add NSE ticker (e.g. HDFCLIFE)")
                add_btn = gr.Button("Add to Watchlist")
                add_msg = gr.Markdown()
            # Right: charts, positions, trade bar, chat
            with gr.Column(scale=4):
                # Top: main chart (reusing portfolio history for now)
                gr.Markdown("**Main Chart**")
                hist_plot = gr.LinePlot(
                    x="time",
                    y="value",
                    label="Main Chart",
                )

                # Middle: P&L + sector exposure
                with gr.Row():
                    pnl_plot = gr.LinePlot(
                        x="time",
                        y="value",
                        label="Portfolio P&L",
                    )
                    sector_plot = gr.BarPlot(
                        x="Sector",
                        y="Value",
                        label="Sector Exposure",
                    )

                # Bottom: positions + trade bar + chat
                with gr.Row():
                    with gr.Column(scale=3):
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
                    with gr.Column(scale=2):
                        side = gr.Radio(["BUY", "SELL"], value="BUY", label="Side")
                        ticker = gr.Dropdown(WATCHLIST, value="HDFCBANK", label="Ticker")
                        qty = gr.Slider(1, 100, value=1, step=1, label="Quantity")
                        trade_btn = gr.Button("Submit Trade")
                        trade_result = gr.Markdown()
                        chat_history = gr.Textbox(
                            lines=10, label="AI Copilot (chat log)", interactive=False
                        )
                        chat_input = gr.Textbox(label="Ask FinAlly India")
                        chat_send = gr.Button("Send")

        trade_btn.click(
            fn=trade_and_refresh,
            inputs=[side, ticker, qty],
            outputs=[
                trade_result,
                port_value,
                cash_value,
                index_value,
                watch_df,
                positions_df,
                pnl_plot,
                sector_plot,
            ],
        )
        add_btn.click(
            fn=add_to_watchlist,
            inputs=[new_ticker],
            outputs=[add_msg, watch_df, ticker],
        )
        chat_send.click(
            fn=gradio_chat,
            inputs=[chat_input],
            outputs=[
                chat_history,
                trade_result,
                port_value,
                cash_value,
                index_value,
                watch_df,
                positions_df,
                pnl_plot,
                sector_plot,
            ],
        )

        # Initial refresh on load
        demo.load(
            fn=gradio_refresh,
            inputs=[ticker],
            outputs=[
                port_value,
                cash_value,
                index_value,
                watch_df,
                positions_df,
                pnl_plot,
                sector_plot,
            ],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        css="""
        body { background-color: #020617; color: #e5e7eb; }
        .gradio-container { max-width: 1400px !important; margin: 0 auto; }
        .tabitem, .block.padded, .block { background: transparent !important; }
        .markdown h1, .markdown h2, .markdown h3 {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
            letter-spacing: 0.03em;
        }
        .markdown p { color: #9ca3af; }
        button {
            border-radius: 999px !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }
        button.primary {
            background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
            border: none !important;
        }
        /* Flatten panels and tables for full visibility */
        .gr-box, .gr-panel, .gradio-data-frame {
            border-radius: 4px !important;
            border: 1px solid rgba(55, 65, 81, 0.9) !important;
            background-color: #020617 !important;
            box-shadow: 0 8px 24px rgba(15,23,42,0.7) !important;
            overflow: visible !important;
        }
        table {
            font-size: 0.86rem !important;
        }
        thead tr th {
            background-color: #0b1120 !important;
            color: #e5e7eb !important;
            font-weight: 600 !important;
        }
        tbody tr:nth-child(odd) { background-color: #020617 !important; }
        tbody tr:nth-child(even) { background-color: #030712 !important; }
        tbody tr:hover { background-color: #1d283a !important; }
        td, th {
            padding: 6px 10px !important;
            border-color: rgba(31, 41, 55, 0.9) !important;
        }
        textarea, input, select {
            border-radius: 0.6rem !important;
            border-color: rgba(75, 85, 99, 0.9) !important;
            background-color: #020617 !important;
            color: #e5e7eb !important;
        }
        """
    )

