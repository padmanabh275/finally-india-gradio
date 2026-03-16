# FinAlly India – Gradio Port Plan

This document breaks the `streamlit_app.py` → `gradio_app.py` port into three clear phases.
Work through them in order; each phase should be stable before you move on.

---

## Phase 1 – Layout + tables only (no trading, no chat)

**Goal:** Replicate the overall Streamlit layout and data views using Gradio Blocks, but keep everything read‑only.

### 1.1. Basic Blocks skeleton

- Use `gradio_app.py` with:
  - `build_ui()` that returns a `gr.Blocks` instance.
  - `if __name__ == "__main__": ui = build_ui(); ui.launch()` for local testing.
- Organize layout with `gr.Row()` / `gr.Column()` so it mirrors `streamlit_app.py`:
  - Header + KPI row.
  - Main body row: left watchlist, right tiles.

### 1.2. Header + KPI tiles

- Top header row:
  - `gr.Markdown` with the FINALLY INDIA pill, title, and subtitle (can use the same HTML used in `streamlit_app.py`).
- Below that, a KPI row with three tiles:
  - Portfolio Value
  - Available Cash
  - NIFTY 50 (Spot)
- Each KPI is rendered inside a `<div class="metric-card">…</div>` using the CSS classes copied from `streamlit_app.py`.

### 1.3. Left column – Watchlist

- In the main row:
  - Left `gr.Column(scale=2)` holds the **watchlist**.
- Add a `gr.Dataframe` with columns:
  - `["Ticker", "Sector", "Price (INR)", "Qty", "Value (INR)"]`
  - `interactive=False` (read‑only in Phase 1).
- Below the table, add “Add listed company” controls:
  - `gr.Textbox` for ticker symbol.
  - `gr.Textbox` for sector.
  - `gr.Button("Add to watchlist")` (wiring can be deferred to Phase 2).
- Populate the initial table from `DEFAULT_TICKERS` and default sectors, similar to `render_watchlist` in `streamlit_app.py`.

### 1.4. Right column – Positions, P&L, Sector Exposure

- Right `gr.Column(scale=3)` holds the main tiles:
  1. **Positions & Trades panel**
     - Wrap with `gr.HTML('<div class="panel-card"><div class="panel-title">Positions & Trades</div>')` and closing `</div>`.
     - Inside, add:
       - `gr.Dataframe` for positions with columns:
         - `Ticker, Qty, Avg Cost (INR), Price (INR), Value (INR), Unrealized P&L (INR), P&L %`.
       - Optionally a `gr.Dataframe` for trades (Phase 1 can just show an empty frame).

  2. **P&L and Sector panels**
     - New `gr.Row()` with two `gr.Column`s:
       - Left: `panel-card` with a `gr.LinePlot` for **Portfolio P&L**.
       - Right: `panel-card` with a `gr.BarPlot` for **Sector exposure**.

### 1.5. Initial data refresh

- Implement a `gradio_refresh(selected_ticker: str)` function that:
  - Calls `fetch_prices(DEFAULT_TICKERS)` (copied from `streamlit_app.py`).
  - Builds:
    - Portfolio value text (`format_inr(compute_portfolio_value(prices))`).
    - Watchlist DataFrame (using current `STATE` positions + prices).
    - Positions DataFrame.
    - P&L history DataFrame (based on `STATE["portfolio_history"]`).
    - Sector exposure DataFrame.
- Hook this to `demo.load(...)` so all tiles populate when the app loads.

**Phase 1 done when:**

- Running `python FinalIndia/gradio_app.py` shows:
  - Header + KPI row.
  - Left watchlist table.
  - Right positions table, P&L line plot, and sector bar plot.
- No trade execution or AI chat yet – dashboard only.

---

## Phase 2 – Wire trade logic (Trade Bar + portfolio updates)

**Goal:** Make the Gradio app behave like the Streamlit one for trading: buying/selling updates cash, positions, P&L, and charts.

### 2.1. Global state

- At the top of `gradio_app.py`, define a global `STATE`:

```python
STATE = {
    "cash": STARTING_CASH,
    "positions": {},          # ticker -> {"qty": float, "avg_cost": float}
    "trades": [],             # list of dicts
    "portfolio_history": [],  # list of {"time": datetime, "value": float}
}
```

- Reuse from `streamlit_app.py`:
  - `execute_trade(side, ticker, qty, price)` adapted to use `STATE`.
  - `compute_portfolio_value(prices)`.
  - `update_portfolio_history(prices)`.

### 2.2. Trade Bar UI

- In the bottom area of the right column, add a **Trade Bar** panel:
  - `gr.Radio(["BUY", "SELL"], value="BUY", label="Side")`
  - `gr.Dropdown(DEFAULT_TICKERS, label="Ticker")`
  - `gr.Slider(1, 100, value=1, step=1, label="Quantity")`
  - `gr.Button("Submit Trade")`
  - `gr.Markdown()` for displaying trade results.

### 2.3. Trade callback and refresh

- Implement:

```python
def gradio_trade(side: str, ticker: str, qty: float) -> str:
    prices = fetch_prices([ticker])
    price = prices.get(ticker)
    if price is None:
        return f"Could not fetch price for {ticker}."
    msg = execute_trade(side, ticker, qty, price)
    all_prices = fetch_prices(DEFAULT_TICKERS)
    update_portfolio_history(all_prices)
    return msg
```

- Then wrap it with a function that also refreshes all tiles:

```python
def trade_and_refresh(side, ticker, qty):
    msg = gradio_trade(side, ticker, qty)
    port_value, watch_df, pos_df, hist_df, sector_df = gradio_refresh(ticker)
    return msg, port_value, watch_df, pos_df, hist_df, sector_df
```

- Wire the button:

```python
trade_btn.click(
    fn=trade_and_refresh,
    inputs=[side, ticker, qty],
    outputs=[trade_result, port_value, watch_df, positions_df, hist_plot, sector_plot],
)
```

**Phase 2 done when:**

- Submitting a trade:
  - Adjusts `STATE["cash"]` and `STATE["positions"]` correctly.
  - Updates portfolio value, watchlist table, positions table, and both charts.

---

## Phase 3 – Port AI Copilot (LLM + ACTION_JSON)

**Goal:** Match the Streamlit AI behaviour: same prompt, same `ACTION_JSON` multi‑trade format, and automatic execution through the Gradio UI.

### 3.1. LLM pipeline setup

- Import `transformers.pipeline` and `torch` at the top.
- Define:

```python
HF_PIPELINE = None
HF_MODEL_ID = None
```

- Implement `get_hf_pipeline()`:
  - Reads `LOCAL_LLM_MODEL` from env or uses a default like `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
  - Uses `device=-1` (CPU) to avoid CUDA issues on Spaces.
  - Optionally includes `HF_TOKEN` for private/gated models.

### 3.2. `call_llm` with FinAlly prompt

- Copy the tightened system prompt from `streamlit_app.py` (the one that:
  - Explains India‑only equities,
  - Describes the `ACTION_JSON` format with a `trades` array,
  - Requires one optional `ACTION_JSON:` line at the end).
- Implement:

```python
def call_llm(prompt: str, portfolio_summary: str) -> str:
    gen_pipe, model_id = get_hf_pipeline()
    # build full prompt with system instructions, portfolio_summary, and latest user question
    out = gen_pipe(full_text, max_new_tokens=400, do_sample=True, temperature=0.4, top_p=0.9)
    return out[0]["generated_text"].strip()
```

### 3.3. Parse `ACTION_JSON` and execute trades

- After receiving `answer` from `call_llm`, search for `"ACTION_JSON:"`.
- Extract the JSON on the line immediately after and parse it with `json.loads`.
- Support schema:

```json
{"trades":[{"side":"BUY","ticker":"HDFCBANK","qty":2},{"side":"SELL","ticker":"TCS","qty":1}]}
```

- For each trade:
  - Fetch current price via `fetch_prices([ticker])`.
  - Call `execute_trade(side, ticker, qty, price)`.
  - Collect human‑readable descriptions like `"BUY 2 HDFCBANK @ ₹x.xx"`.
- Append a note such as:

```text
(Executed trades: BUY 2 HDFCBANK @ ₹x.xx; SELL 1 TCS @ ₹y.yy)
```

to the assistant message.

### 3.4. Gradio chat panel

- In the UI, under the Trade Bar, add:
  - `gr.Textbox(lines=10, label="AI Copilot (chat log)", interactive=False)`
  - `gr.Textbox(label="Ask FinAlly India")`
  - `gr.Button("Send")`

- Maintain `STATE["chat"] = [(role, content), ...]`.
- Implement a `gradio_chat(message: str)` function that:
  - Builds a compact portfolio snapshot from `STATE["positions"]` and latest prices.
  - Calls `call_llm(message, snapshot)`.
  - Parses and executes `ACTION_JSON` trades as above.
  - Appends both user and assistant messages to `STATE["chat"]`.
  - Builds a chat log string:

```python
chat_log = "\n\n".join(f"{r.upper()}: {c}" for r, c in STATE["chat"])
```

  - Calls `gradio_refresh(...)` to refresh portfolio tiles after any trades.
  - Returns updated `chat_log`, an answer/notification area, and refreshed KPI/tables/plots.

- Wire the button:

```python
chat_send.click(
    fn=gradio_chat,
    inputs=[chat_input],
    outputs=[chat_history, trade_result, port_value, watch_df, positions_df, hist_plot, sector_plot],
)
```

**Phase 3 done when:**

- Entering natural‑language trade instructions (e.g. “buy 2 shares of HDFCBANK”) in the chat:
  - Produces a sensible explanation.
  - Executes the corresponding trades via `ACTION_JSON`.
  - Updates positions, cash, P&L, and watchlist to match the Streamlit terminal’s behaviour.

