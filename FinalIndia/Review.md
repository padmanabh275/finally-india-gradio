# FinAlly India – Plan Review (PLAN.md)

## Overall Impression

- The India adaptation is **clear and grounded**: it keeps the core FinAlly architecture while cleanly swapping in Indian tickers, INR, and NSE/BSE context.
- Scope feels **ambitious but still manageable** for an MVP, assuming we treat real Indian data and F&O as follow‑on work rather than initial blockers.
- The document reads well as a contract between agents: it’s explicit about default watchlists, currency, and where India‑specific behavior diverges from the global project.

## Strengths

- **Strong domain focus**: explicit mention of NIFTY, SENSEX, large‑cap Indian names, INR formatting, and Indian market hours gives enough specificity for agents to avoid US‑centric assumptions.
- **Reuse of core architecture**: same SSE pattern, same database schema, same API surface – minimizes duplicated design while allowing region‑specific seeding and prompts.
- **LLM guidance is practical**: clearly states that trading is simulated, no brokerage connection, and constrains the assistant away from legal/tax advice.
- **Testing section is actionable**: India‑specific E2E flows (e.g., buying RELIANCE/TCS, portfolio diversification questions) map directly to test cases implementers can write.

## Questions / Clarifications

- **F&O scope**: The vision mentions “equity and F&O markets,” but the rest of the document only specifies equity‑style behavior (shares, no margin, no options/greeks). For now, should the plan:
  - Explicitly state **“Phase 1: cash equity only, no derivatives”**, and push F&O to a separate design document?
- **Region switching**: There is a `MARKET_REGION=IN` flag, but most of this document assumes an India‑only deployment.
  - Do we actually intend to run **multiple regions in one binary** (US + IN), or is India a dedicated variant? If it’s the latter, we could simplify and treat `MARKET_REGION` as fixed in this repo.
- **Database naming**: The doc references `finally_india.db`, but schema is unchanged and this is single‑user.
  - Is there a real need for a separate filename versus reusing `finally.db` with India seeds (and making “India” purely a data/ENV choice)?
- **Indian data provider**: The plan sensibly keeps the provider abstract, but:
  - Do we want to shortlist candidate APIs (e.g., broker APIs vs. 3rd‑party data vendors) so a future agent doesn’t have to re‑research from scratch?

## Opportunities to Simplify Further

- **Declare equity‑only explicitly**: Add a one‑sentence line early in the Vision or User Experience sections:
  - _“Phase 1 focuses on cash equity only; F&O support is out of scope for the initial build.”_
- **Lock the region for this repo**: Instead of a general multi‑region toggle, define this as:
  - _“This profile assumes India is the only active market; `MARKET_REGION=IN` is treated as fixed.”_
  This lets frontend and backend avoid conditional UI/logic for other markets.
- **Simulator‑first strategy**: Mark the real Indian data client as a **clearly labeled stretch goal** with a short note:
  - What would minimally count as “done” (e.g., read last traded price + percent change, no full order book).
- **Centralize defaults**: Consider adding a small “India Defaults” table (cash, watchlist tickers, index shown in header) so seed data and frontend copy stay in sync.

## Suggested Next Concrete Steps

1. Amend PLAN.md to:
   - State “equity‑only for Phase 1” explicitly.
   - Clarify that this repo runs in **India mode only**.
2. Add a short “India Defaults” subsection listing:
   - Starting cash, initial watchlist, default index in header, and typical simulator parameters.
3. (Optional) Add a one‑paragraph note naming 1–2 candidate Indian market data APIs as future integration targets.

