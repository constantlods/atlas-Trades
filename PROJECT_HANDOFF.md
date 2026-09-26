# PROJECT HANDOFF / CURRENT STATE SUMMARY — Trading System (as of 2026-09-25, inspected directly from source)

**Methodology note:** everything below was produced by re-reading the actual files in `/root/trading-system` in a Claude Code session (Read/grep/pytest only — no edits were made). Where a file was not re-read fresh in that session, it is marked UNKNOWN/PARTIALLY-VERIFIED rather than asserted from memory.

---

## 1. PROJECT OVERVIEW

This is a modular Python system that combines a Ross-Cameron-style momentum stock screen, a direction-agnostic "Indication-Correction-Continuation" (ICC) price-action entry framework, a local-LLM (Ollama) advisory review layer, a deterministic risk/position-sizing engine, and a paper-money execution simulator. It runs in two modes that share almost all the same code:

- **LIVE mode**: a FastAPI dashboard (`dashboard/app.py`) with a background polling worker (`dashboard/worker.py`) that scans real-time Alpaca data, evaluates setups, and lets a human click "Enter" to simulate a paper trade. No real brokerage orders are ever placed — this is 100% paper/simulated money.
- **BACKTEST/RESEARCH mode**: `backtesting/runner.py` replays the identical screen→ICC→trade-plan→risk→execution pipeline against real historical Alpaca bars, with point-in-time data reconstruction to avoid look-ahead bias, plus research instrumentation (execution tiers, multi-account-size simulation, shortability, forward-outcome measurement) that live mode does not have.

Text architecture diagram (actual, as found in code):

```
                        ┌─────────────────────────────────────────┐
                        │            Alpaca Markets API             │
                        │  TradingClient / StockHistoricalDataClient │
                        │  ScreenerClient (movers) / NewsClient      │
                        └───────────────┬─────────────────────────┘
                                        │
                    ┌───────────────────┴────────────────────┐
                    │                                          │
             LIVE PATH                                  BACKTEST PATH
                    │                                          │
   scanner/universe.py                          backtesting/universe.py
   build_candidate_universe(                     universe_a_ross_momentum /
     top_gainers, top_actives)                    universe_a_down_movers /
   [top_losers NOT passed → default 0]            universe_b_liquid_stocks /
                    │                              universe_c_etfs
                    │                                          │
   scanner/data_source.py                        backtesting/data.py
   AlpacaScannerDataSource                        HistoricalBarFetcher
   (live path, no as_of)                          (point-in-time, as_of=...)
                    │                                          │
   ross_strategy.evaluate  ◄── ONLY screen run          ross_strategy.evaluate
   (bearish_strategy is                            AND  bearish_strategy.evaluate
    never called here)                              (both run on every candidate)
                    │                                          │
   ranking/scorer.py (scores candidates)                       │
                    │                                          │
   market_structure.analyze_structure  ───────────────────────┤
   icc_strategy.detect_icc_setups (direction-agnostic)         │
                    │                                          │
   dashboard/worker.py pick_current_setup                backtesting/runner.py
   (any direction, whichever setup exists)              _earliest_confirmed
                    │                                    (causality-filtered
                    │                                     on candidate.as_of)
   trade_plan.build_trade_plan  ───────────────────────────────┤
                    │                                          │
   risk_manager.evaluate_risk  ───────────────────────────────┤
   (9 deterministic checks)                                    │
                    │                                          │
   llm_agent.review_setup (Ollama)          execution/shortability (UNKNOWN always)
   [live /api/enter only; informational,    execution/tiers (research tiers)
    can only block, never override risk]    execution/account_sim (tri-state,
                    │                         multi-equity-level)
   execution_engine.simulate_entry  ────────────────────────────┤
   portfolio_manager.PaperPortfolio                     backtesting/outcome.py
                    │                                   (forward MFE/MAE, never
   trade_journal (SQLite, WAL)                            fed back into signal)
                    │                                          │
                                                        backtesting/integrity.py
                                                        backtesting/narrative.py
```

---

## 2. CURRENT PROJECT STATUS

| Component | Status | File(s) | What it does | Known problems |
|---|---|---|---|---|
| Market data (live) | IMPLEMENTED | `market_data/alpaca_provider.py`, `market_data/base.py` | Wraps Alpaca quotes/bars/account status for live trading | Not re-read this session in full; interface confirmed via `MarketDataProvider` ABC usage elsewhere |
| Market data (historical) | IMPLEMENTED | `backtesting/data.py` (`HistoricalBarFetcher`) | Fetches 1-min regular-session (9:30–16:00 ET) bars for a specific past date via `StockBarsRequest`, IEX feed | None found; DST-safe (computes session bounds in `America/New_York`) |
| Scanner / candidate sourcing (live) | IMPLEMENTED, bullish-only | `scanner/universe.py` (`build_candidate_universe`), `dashboard/worker.py:305` | Pulls top-gainers + most-actives from Alpaca's live screener | `top_losers` parameter exists but defaults to 0 and is **never passed** by `worker.py` — live candidate pool never includes real down-movers |
| Scanner / candidate sourcing (backtest) | IMPLEMENTED | `backtesting/universe.py` (Universe A/B/C) | Universe A = journal history + live top-losers pass; Universe B = live most-actives; Universe C = caller-supplied list | Universe B/C's *candidate set* is always today's live screener state even when `as_of` is historical — only the per-symbol metrics are point-in-time correct (documented in-file) |
| Ross-style candidate selection | IMPLEMENTED | `ross_strategy/config.py`, `ross_strategy/evaluator.py` | Two hard pillars (`min_relative_volume=5.0`, `min_percent_gain=10.0`); price ($1–$20) and float (≤10M) are soft/non-blocking | None found in the logic itself |
| Bearish screen | IMPLEMENTED (module), NOT WIRED into live path | `bearish_strategy/config.py`, `bearish_strategy/evaluator.py` | Structural mirror of Ross: `min_relative_volume=5.0`, `percent_change <= -min_percent_loss(10.0)` | Only ever called from `backtesting/runner.py`. `dashboard/worker.py` never imports it (confirmed via grep — zero matches) |
| ICC logic | IMPLEMENTED | `icc_strategy/detector.py`, `icc_strategy/models.py`, `market_structure/breaks.py` | Direction-agnostic indication→correction→continuation detector, confirmed symmetric bullish/bearish | Setups are not deduplicated/cross-invalidated — multiple simultaneous setups of either direction can coexist (documented) |
| Bullish logic (live) | IMPLEMENTED | `dashboard/worker.py` | Candidates sourced bullish-only, ICC run direction-agnostically on top | — |
| Bearish logic (live) | PARTIALLY IMPLEMENTED / MISLEADING NAME | `dashboard/worker.py` `build_candidate_view` | A "bearish" simulated trade can occur **only** when a bearish ICC continuation happens to form inside a bullish-*screened* (up-moving) stock's own price action. There is no live path that sources genuine down-movers and screens them bearishly. | This is the most important nuance for anyone reading "bearish" in this system — see Section 9 |
| Bearish logic (backtest) | IMPLEMENTED | `backtesting/runner.py` | Both screens run on every candidate; direction picked by whichever screen passed (tie-broken by sign of `percent_change` if both pass) | — |
| Liquidity | IMPLEMENTED (as an execution-feasibility gate, not a signal filter) | `scanner/liquidity.py`, `risk_manager/engine.py` | Live gate: `avg_daily_volume >= min_avg_daily_volume (50,000)`, a single scalar comparison | See Section 6 |
| Spread | IMPLEMENTED (live), SYNTHETIC (backtest) | `risk_manager/engine.py`, `backtesting/runner.py` (`_BacktestMarketData`) | Live: real quote-derived `(ask-bid)/mid*100 <= 6.0%`. Backtest: fixed synthetic ~0.1% placeholder (no historical NBBO exists) | Backtest spread numbers are not real; documented in three places in code |
| Risk management | IMPLEMENTED | `risk_manager/config.py`, `risk_manager/engine.py` | 9 unconditional checks (kill switch, market hours, daily loss, open positions, stop-loss present, spread, liquidity, risk-per-trade, position-size/share bounds) | — |
| Position sizing | IMPLEMENTED, INTEGER SHARES ONLY | `risk_manager/engine.py` (`_size_position`) | `min(shares_from_risk, shares_from_position_cap, max_shares)`, both caps computed via `int(...)` (floor) | No fractional-share/notional path exists anywhere — see Section 8 |
| Execution simulation | IMPLEMENTED (paper only) | `execution_engine/simulator.py` | Fills only if risk-approved; applies slippage; conservative worst-case stop/target tie-break | Pure symmetric P&L math for either direction — the word "short" does not appear in this file |
| Shorting (real broker mechanics) | BUILT BUT COMPLETELY UNWIRED (dormant) | `execution/shortability.py` | `AlpacaShortabilityChecker` (live, via `TradingClient.get_asset`) and `backtest_shortability()` (always returns `SHORTABILITY_UNKNOWN`) both exist and are tested | Zero references anywhere in `dashboard/`, `execution_engine/`, or `risk_manager/` (confirmed via grep) — no live or paper trade, bullish or bearish, is ever gated on shortability today |
| Backtesting | IMPLEMENTED | `backtesting/runner.py`, `backtesting/config.py` | Full pipeline replay against one continuous simulated portfolio, chronologically ordered | See Section 10 for exact realism/simplification breakdown |
| Historical replay / point-in-time reconstruction | IMPLEMENTED, with real coverage gaps | `scanner/data_source.py` (`_point_in_time_metrics`) | Reconstructs `DailyMetrics` from 1-min bars truncated at `as_of`; two documented anti-lookahead fixes already applied | Returns `None` (→ `data_quality=INVALID`) whenever `as_of` predates the regular session (e.g., pre-market discovery) — in the most recent checkpoint this affected 37/129 (28.7%) of Universe A candidates |
| Database / journal | IMPLEMENTED | `trade_journal/db.py`, `trade_journal/journal.py`, `trade_journal/models.py` | SQLite, WAL mode, 4 tables: `scans`, `setups`, `trades`, `open_positions` | Schema is flat/simple relative to `backtesting/models.py`'s much richer `TradeRecord` — journal does not persist tier/participation/account-size/shortability fields |
| Reporting | IMPLEMENTED (backtest), PARTIAL (live) | `analytics/report.py`, `analytics/breakdown.py`, `backtesting/narrative.py`, `backtesting/integrity.py` | Backtest: rich integrity + performance + narrative text report. Live: equity curve + recent trades only (`/api/history`) | `analytics/*.py` internals not re-read fresh this session — presence and call sites confirmed, exact formulas UNKNOWN pending re-verification |
| Configuration | IMPLEMENTED | `config/settings.py`, `dashboard/config.py`, `backtesting/config.py`, per-module `config.py` files | All dataclasses, frozen, with `__post_init__` validation | `config/settings.py` holds only Alpaca keys + paper flag — no other env-driven settings exist |
| API integrations | IMPLEMENTED | Alpaca (`TradingClient`, `StockHistoricalDataClient`, `ScreenerClient`, `NewsClient`), Ollama (`llm_agent/client.py`) | — | Account confirmed unable to query SIP/consolidated data (IEX feed only) |
| Logging | PARTIAL | scattered `logging.getLogger(__name__)` calls, e.g. `scanner/universe.py`, `dashboard/app.py` | Standard Python logging, mostly warning/exception level on recoverable failures | No structured/centralized logging config reviewed — UNKNOWN whether logs are persisted to a file in production |
| Tests | IMPLEMENTED, broad | `tests/` (32 files) | `.venv/bin/python -m pytest -q` → **510 passed, 0 failed** | Passing tests confirm unit-level behavior of the modules above; they do not by themselves confirm strategy profitability or production readiness |
| Local LLM reasoning layer | IMPLEMENTED, advisory-only | `llm_agent/` (`client.py`, `reviewer.py`, `context.py`, `prompts.py`, `models.py`, `config.py`) | Ollama-backed; `should_execute()` is a strict whitelist — only `recommendation=="approve"` can permit a trade beyond risk-approval; `risk_approved=False` always wins; malformed/unreachable LLM responses fail safe to `"uncertain"` | `block_on_uncertain_llm` defaults to `False`, so an "uncertain" verdict does **not** block execution by default — the LLM layer is currently closer to a logged opinion than a gate |

---

## 3. EXACT SIGNAL LOGIC

**Bullish (`ross_strategy/evaluator.py` + `ross_strategy/config.py`):**
- Inputs: `ScanCandidate` (symbol, `DailyMetrics`, optional catalyst).
- Hard pillars (both required): `relative_volume >= 5.0`, `percent_change >= 10.0`.
- Soft/non-blocking: `price` between $1–$20, `float_shares <= 10,000,000` — docstring in `config.py` states these are guidelines, never gating.
- Confirmation/entry trigger is **not** in this module — a Ross pass only produces a `ScanCandidate` eligible for ICC structure analysis. The actual entry trigger is the ICC continuation (Section 4).
- Exit logic is not part of this module either — it lives in `trade_plan/planner.py` (stop/target) and `execution_engine/simulator.py` (fill/close mechanics).

**Bearish (`bearish_strategy/evaluator.py` + `bearish_strategy/config.py`):**
- Structural mirror: `relative_volume >= 5.0`, `percent_change <= -10.0` (checked as `percent_change <= -min_percent_loss`).
- Same soft price/float guidelines.
- Deliberately does **not** encode VWAP/failed-breakout/support-break features as filters — docstring states those are recorded-only elsewhere, not gating this screen.
- **Sequence, confirmation, invalidation, entry trigger, exit logic are IDENTICAL in mechanism to bullish** — both screens hand off to the same direction-agnostic ICC detector and the same `trade_plan`/`risk_manager`/`execution_engine` pipeline. There is no separate "bearish entry trigger" implementation; direction is a parameter, not a different code path.

---

## 4. ICC SETUP

Defined in `icc_strategy/models.py` / `icc_strategy/detector.py` / `market_structure/breaks.py`:

- **Indication**: a bar's `close` breaks beyond a prior swing high (bullish) or swing low (bearish) — detected by `market_structure/breaks.py:_scan_breaks`, called once per direction by `detect_breaks_of_structure`.
- **Correction**: `icc_strategy/detector.py:_find_correction` finds the first opposite-kind swing after the indication; `reaction_level` is the wick-based high/low of the window from indication to that swing.
- **Continuation**: `_find_continuation` scans forward from the correction's extreme for the first bar whose **close** moves beyond `reaction_level`.
- **Trend at indication**: `_trend_as_of` computes trend using **only** bars up to and including the indication's own bar — explicitly documented to avoid lookahead.
- **Stage property**: `ICCSetup.stage` is `"continuation"` > `"correction"` > `"indication"`, based on which optional fields are populated.
- **Trend alignment**: `ICCSetup.aligned_with_trend` — `True` only if setup direction matches `trend_at_indication`; purely informational/descriptive, not a filter anywhere in the pipeline (both `dashboard/worker.py` and `backtesting/runner.py` record it but do not gate on it).
- **Symmetry**: confirmed at the structural level — `detect_breaks_of_structure` calls the same `_scan_breaks` function on highs (bullish) and lows (bearish). Bullish and bearish ICC are fully symmetrical in code.
- **Multiplicity**: `detect_icc_setups` does **not** deduplicate or cross-invalidate setups — multiple simultaneous bullish and bearish setups can coexist over the same bars (explicitly documented as a known, deliberate simplification).
- Exact file pointers: `icc_strategy/detector.py` (all functions above), `icc_strategy/models.py` (`Indication`, `Correction`, `Continuation`, `ICCSetup`), `market_structure/breaks.py` (`_scan_breaks`, `detect_breaks_of_structure`).

---

## 5. ROSS-STYLE SCANNER (current implementation only)

- Source: Alpaca `ScreenerClient.MarketMoversRequest` (`.gainers`/`.losers`) plus `most_active_symbols`.
- Filters actually enforced (`ross_strategy/config.py`):
  - **Relative volume** ≥ 5.0× — hard.
  - **Percent gain** ≥ 10.0% — hard.
  - **Price** $1.00–$20.00 — soft, non-blocking.
  - **Float** ≤ 10,000,000 shares — soft, non-blocking.
  - **Market cap**: NOT implemented anywhere — `market_cap` field is hardcoded `None` throughout (`backtesting/runner.py`, `scanner/liquidity.py`).
  - **News/catalyst**: fetched (`scanner/data_source.py:get_latest_catalyst`) and attached, recorded as `has_catalyst`, but is **not a filter** — it's informational and fed into the LLM prompt context.
  - **Time-of-day**: no explicit time-of-day filter found in `ross_strategy` itself. The point-in-time reconstruction (`_point_in_time_metrics`) returns `None` for any `as_of` before 9:30 ET, which indirectly excludes pre-market discoveries from ever getting a signal, but this is a data-availability side effect, not a designed time-of-day filter.

---

## 6. LIQUIDITY AUDIT

- **Exact formula (live risk gate)**: `risk_manager/engine.py` — `liquidity_ok = avg_daily_volume >= config.min_avg_daily_volume`. `min_avg_daily_volume = 50,000.0` (shares, not dollars) — a single scalar comparison.
- **Units**: shares/day, not dollar volume, in the live risk gate.
- **Dollar volume**: computed and recorded (`scanner/liquidity.py: LiquidityProfile.dollar_volume`, `DailyMetrics.dollar_volume`) but is **not** part of the live risk-engine liquidity check — it feeds `execution/tiers.py`'s research tiering instead.
- **Current vs. historical vs. average volume**: `avg_daily_volume` passed into `evaluate_risk` is the metrics object's `avg_volume` (trailing average over `lookback_days`, default 20) — not the current/live single-day volume.
- **Bid/ask size, order-book depth**: `LiquidityProfile` has `bid_size`/`ask_size` fields but no code path reads them for a pass/fail decision — appears to be recorded-only.
- **Position size / account size consideration**: the core `liquidity_ok` check does **not** consider the intended position size or account equity at all — it is purely "is this security liquid," independent of how many shares you intend to buy. Participation-adjusted analysis (position size vs. liquidity) exists **only** in the separate `execution/` research package (`participation_percent`, `execution/tiers.py`), used in backtesting, not in the live risk gate.
- **Exact code location**: `risk_manager/engine.py` (`evaluate_risk`, category `"execution_feasibility"`); `scanner/liquidity.py` (`build_liquidity_profile`, `LIQUIDITY_BUCKETS` — explicitly documented "research buckets, NOT asserted-optimal thresholds"); `execution/tiers.py` (`TierThresholds`, `classify_execution_tier` — also explicitly "research parameters... not validated cutoffs").
- **What causes liquidity failure**: `avg_daily_volume` is `None` or below 50,000 shares/day.
- **Is liquidity currently (A) signal filter, (B) execution filter, (C) risk filter, or (D) combination?** → **(C) risk/execution-feasibility filter only.** It is evaluated *after* a signal is already valid (`ross_strategy`/`bearish_strategy` do not consult it at all) and is coded under `category="execution_feasibility"` in `risk_manager`. It never blocks a candidate from being scanned or from generating a signal — only from being risk-approved for execution.

---

## 7. SPREAD

- **Live formula**: `spread_percent = (ask - bid) / mid * 100`, `mid = (bid+ask)/2`; `spread_ok = 0 <= spread_percent <= max_spread_percent (6.0%)`. On `MarketDataError` (no quote available), fails safe: `spread_ok=False`.
- **Backtest**: `_BacktestMarketData.get_quote()` in `backtesting/runner.py` always returns a synthetic quote at `entry_price * 0.9995` / `entry_price * 1.0005` — a fixed ~0.1% spread, never real historical NBBO (documented in three places: `_BacktestMarketData`'s docstring, `execution/tiers.py`'s `TierThresholds` docstring, and `backtesting/models.py`'s `TradeRecord.spread` field docstring).
- **Has spread actually rejected any candidates in recent tests?** In the most recent checkpoint (`data/research_checkpoints/2026-09-25.json`), Universe A's execution-failure attribution shows **`spread: 0 (0.0%)`** of 15 recorded execution-check failures — liquidity accounted for 100% of them. Because backtested spread is synthetic and fixed near-zero, this is close to a certainty (a ~0.1% spread will essentially never breach a 6.0% cap) rather than a real-world finding about spread risk — **spread has not been meaningfully tested against this system's actual candidates** because the backtest data source has no historical NBBO.

---

## 8. ACCOUNT SIZE / POSITION SIZING

- **Configurable?** Yes. `RiskConfig.max_risk_per_trade_percent` (1.0%) and `max_position_percent` (20.0%) are dataclass fields, not hardcoded.
- **Default equity**: live dashboard `starting_equity=100,000.0` (`dashboard/config.py`); backtest `starting_equity=100,000.0` (`backtesting/config.py`) — described in-code as "BACKTEST CAPITAL... a standardized research account," explicitly **not** a liquidity requirement.
- **Is $100 / $1,000 / $100,000 supported?** Yes, via a *separate* mechanism: `execution/account_sim.py`'s `DEFAULT_RESEARCH_EQUITY_LEVELS = (100.0, 500.0, 1000.0, 5000.0, 10000.0)` plus the `100,000` standardized baseline. `simulate_account_sizes()` reuses the real `evaluate_risk()` against a clean-slate portfolio at each level. This exists **only in the backtest/research pipeline** — the live dashboard always runs at one fixed configured equity; there is no live "try this at $100" mode.
- **Position size calc**: `risk_manager/engine.py:_size_position` — `shares_from_risk = int(max_risk_dollars // risk_per_share)` where `max_risk_dollars = equity * max_risk_per_trade_percent/100`; `shares_from_position_cap = int(max_position_dollars // entry)` where `max_position_dollars = equity * max_position_percent/100`; final `shares = max(0, min(shares_from_risk, shares_from_position_cap, config.max_shares))`.
- **Risk-per-trade calc**: `equity * max_risk_per_trade_percent / 100`, divided by `risk_per_share` (entry-to-stop distance) to get a share count.
- **Max shares tradeable**: hard cap `config.max_shares = 10,000`, and `config.min_shares = 1`.
- **Does liquidity change position size?** No. `_size_position` does not reference `avg_daily_volume` at all — sizing is purely a function of equity, risk-per-trade%, position%, and the trade plan's own entry/stop. Liquidity is checked as a **separate, independent pass/fail gate** (Section 6), not as an input to the sizing formula.
- **Does position size affect execution feasibility?** Indirectly, via `execution/tiers.py`'s optional `participation_percent` dimension (position size as % of avg volume) — but this is research-only instrumentation in `backtesting/runner.py`, not part of the live risk gate's approve/reject decision.
- **Explicit answer — does the system confuse "account capital" with "market liquidity"?** **No, they are kept as two structurally separate checks** (`category="sizing"` vs `category="execution_feasibility"` in `risk_manager/engine.py`), and `execution/account_sim.py`'s docstring is explicit that `STANDARDIZED_BACKTEST_EQUITY` "is explicitly NOT a liquidity requirement." However, there is a real practical consequence worth flagging: because sizing uses **integer share counts with a floor of 1 share** and no fractional-share path, a high-priced stock (e.g., a $500+ mega-cap) at a small account size (e.g., $100 equity, 1% risk = $1 risk budget) will very often size to **0 shares** and get rejected — not because of a liquidity failure, but because the risk-based share count floors to zero before liquidity is ever checked. This is a sizing-math limitation, not a liquidity/capital confusion, but it produces a similar practical symptom ("small account can't trade this candidate") for a different underlying reason.

---

## 9. BEARISH / SHORTING LOGIC

**Does BEARISH mean avoid-long / sell-existing / short-candidate / actual-short-trade / something else?**

In this codebase, "bearish" means: *a simulated short-sale trade with the identical P&L-sign-mirrored mechanics as a long trade* — `portfolio_manager/models.py:Position.direction` is `"bullish"` or `"bearish"`, and `execution_engine/simulator.py` mirrors entry/exit/slippage math by sign for either direction. It does **not** currently mean "avoid a long" or "sell an existing long" — there is no separate "avoid" signal type; a bearish signal either becomes a simulated short position or it doesn't trade at all.

**Is shorting implemented?**
- **Borrow availability**: `execution/shortability.py` exists (`AlpacaShortabilityChecker`, real Alpaca `get_asset().shortable` check for live; `backtest_shortability()` always returns `"SHORTABILITY_UNKNOWN"` for backtests, since no historical borrow-availability data exists anywhere). **This module is fully built and unit-tested but is called from nowhere in the live trading path** — confirmed by grep across `dashboard/`, `execution_engine/`, `risk_manager/` (zero matches) and by directly reading `dashboard/worker.py:build_candidate_view` and `dashboard/app.py:api_enter` in full: neither imports or calls it.
- **Short-sale constraints (uptick rule, hard-to-borrow fees, etc.)**: NOT IMPLEMENTED anywhere.
- **Margin modeling**: NOT IMPLEMENTED. `execution_engine/simulator.py` has no margin/leverage concept at all — it's pure paper P&L.
- **Position sizing for shorts**: identical formula as longs (Section 8) — no separate short-specific sizing.
- **Entry/stop/target/exit**: identical mechanism as longs, mirrored by sign (`portfolio_manager/models.py`, `execution_engine/simulator.py`, `trade_plan/planner.py`).
- **Borrow cost**: NOT MODELED anywhere.

**If NOT implemented, say so clearly:** Real broker-level short-sale mechanics (borrow checks blocking a live trade, margin, uptick rule, borrow fees) are **NOT IMPLEMENTED / NOT WIRED IN** for either the live dashboard or the backtest execution simulator. What exists is a symmetric paper-trading simulation of the *P&L* of a short, plus a dormant, unused shortability-checking module.

**Live-path bearish reality check** (verified by reading `dashboard/worker.py` lines 117–201 and `scanner/universe.py`/`dashboard/worker.py:305` call site in full): the live worker's candidate pool is sourced *exclusively* from `top_gainer_symbols` + `most_active_symbols` (bullish/neutral universes) — `top_losers` is never requested. `build_candidate_view` then picks *whichever* ICC setup exists (`pick_current_setup`, any direction) and, if it's a confirmed continuation, builds a trade plan and evaluates risk regardless of direction. So the **only** way the live system ever produces a "bearish" simulated trade is: a stock got flagged as an up-mover by Ross's screen, and then, incidentally, its own price action later formed a bearish ICC reversal. There is no live path that sources a genuine down-mover via `bearish_strategy` and screens it bearishly — that only happens in `backtesting/runner.py`.

---

## 10. BACKTESTING

Central file: `backtesting/runner.py:run_backtest`. What's realistically simulated vs. simplified:

| Aspect | Realism |
|---|---|
| Candidate discovery | REALISTIC for Universe A's journal half (real accumulated live-scanner history) and for per-symbol metrics generally; Universe B/C's *candidate set membership* is NOT historical (documented data limitation — always today's live screener list even when `as_of` is historical) |
| Historical bars | REAL Alpaca 1-min bars, IEX feed, regular session only (`HistoricalBarFetcher`) |
| Timestamps | Real bar timestamps; session boundaries computed DST-safe in US/Eastern |
| Candidate generation | Real, per Universe A/B/C definitions (`backtesting/universe.py`) |
| Signal generation | Real — same `ross_strategy`/`bearish_strategy`/`icc_strategy` code as live, run against point-in-time-reconstructed metrics and real bars |
| Entry/exit | Simulated fills via `execution_engine.simulate_entry`/`process_bar`, same code as live |
| Slippage | Modeled (`ExecutionConfig.slippage_percent`), applied to every simulated fill |
| Spread | **SYNTHETIC** — fixed ~0.1% placeholder, not real historical NBBO (Section 7) |
| Liquidity | Real (`avg_volume` computed from real historical daily bars) |
| Position sizing | Real — same `risk_manager` code and formula as live |
| Fees | Modeled (`ExecutionConfig.fee_per_share`, `fee_per_trade`) — exact values not re-verified this session (`execution_engine/config.py` not re-read fresh) |
| Shorting | Backtest P&L simulated same as long; shortability always `SHORTABILITY_UNKNOWN` (never asserted tradeable or untradeable) |
| Stop loss / take profit | Real, computed by `trade_plan/planner.py`, same as live |
| Causality / look-ahead guard | A confirmed ICC continuation is only tradeable if `setup.continuation.timestamp >= candidate.as_of` (the candidate's discovery moment) — enforced explicitly in `run_backtest` |
| Portfolio continuity | Real — ONE continuous simulated `PaperPortfolio` across all candidates in chronological order, so `max_open_positions`/`max_daily_loss`/equity curve are meaningful, not per-candidate-isolated |

Every `TradeRecord` is retained even when a signal was never tradeable — "signal was right but couldn't execute" is treated as a first-class finding (`rejection_chain`, three-way `signal_valid`/`execution_valid`/`trade_valid` classification), not discarded.

---

## 11. POINT-IN-TIME / LOOK-AHEAD AUDIT

**Can the current system faithfully reconstruct "what did the market look like at 1:00 PM on that historical day"?**

**PARTIALLY.**

- `scanner/data_source.py:_point_in_time_metrics` genuinely reconstructs `DailyMetrics` (session open/high/low, volume-so-far, price-at-as_of, prior-day close/avg-volume) using **only** 1-minute bars up to and strictly before `as_of` (fixed with an explicit `b.timestamp + timedelta(minutes=1) <= as_of` filter to avoid the "bar timestamp marks start not end" leak), and prior daily bars explicitly filtered to strictly-before the session date (fixing a documented same-day-bar UTC/Eastern boundary leak). Both fixes are commented in-code as findings from an earlier audit.
- `icc_strategy/detector.py:_trend_as_of` uses only bars up to and including the indication's own bar — explicitly designed to avoid lookahead.
- `backtesting/runner.py` enforces causality on the *tradeable* continuation via the `as_of`-vs-`continuation.timestamp` filter.
- **BUT**: `_point_in_time_metrics` returns `None` (not a stale/wrong value, but simply unavailable) whenever `as_of` predates the regular session start (e.g., a candidate discovered pre-market) — this is the *documented, current* mechanism behind the 37/129 (28.7%) `data_quality=INVALID` records in the most recent checkpoint. It's not silent look-ahead — it fails safe to "no data" — but it does mean a meaningful fraction of real candidates cannot be point-in-time-scored at all today.
- **Known/possible remaining sources of look-ahead**:
  1. Universe B/C's candidate *set membership* is always "today's" live screener list even under a historical `as_of` (documented limitation, `backtesting/universe.py`) — using this to study "who was most active on date X" would be a real look-ahead-adjacent methodology error, though the per-symbol *metrics* used afterward are still point-in-time correct.
  2. `icc_stage`/`icc_direction`/`aligned_with_trend` recorded on a non-tradeable `TradeRecord` are explicitly **descriptive** (can reflect a stage that formed *after* discovery) — documented as intentional (a trader keeps watching forward), but any downstream consumer must not treat those fields as a point-in-time snapshot; only `signal_valid` carries the causality guarantee.
  3. Backtested spread/quote is synthetic, entry-price-derived — cannot leak future information, but also isn't testing anything real.
  4. `analytics/*.py` internals were not re-read fresh this session — a look-ahead issue there cannot be ruled out; flagged UNKNOWN.

---

## 12. RECENT EXPERIMENT RESULTS (from `data/research_checkpoints/2026-09-25.json`, generated 2026-09-25 22:58 UTC — the most recent checkpoint on disk)

**Universe A (journal + live down-movers, both directions):**
- 129 candidates evaluated. `point_in_time_reconstruction_ok: false` — **37 of 129 (28.7%) have `data_quality=INVALID`** (no point-in-time metrics available, per Section 11's mechanism), 51 VALID, 41 PARTIAL.
- 58 cleared the bullish (Ross) screen; 10 cleared the bearish screen (both screens run on every candidate).
- **16 of 129 (12.4%) valid signals** — all 16 bullish, 0 bearish this run.
- Three-way classification: 1 signal valid+executed (0.8%), 15 signal valid but execution infeasible (11.6%), 0 unresolved, 113 no valid signal (87.6%).
- Execution tiers (of 16 valid signals): 0 TIER_1, 0 TIER_2, 1 TIER_3_LOW, 15 TIER_4_UNTRADEABLE.
- Execution failure attribution: **100% of the 15 recorded failures were liquidity** (avg daily volume); 0% spread, 0% sizing, 0% account-state.
- 7 of the 16 valid signals would be traded as a short and carry `SHORTABILITY_UNKNOWN`.
- Forward signal movement (bullish, n=16): mean +2.38% at 1m, **-2.48% at 5m**, +1.57% at 15m, -2.87% at 30m, -3.71% at 60m; MFE mean +37.76%, MAE mean +20.61%. This is noisy and non-monotonic at n=16 — **not a coherent trend, small-sample noise**, not a validated edge in either direction.
- Account-size accessibility: at every equity level from $100 to $100,000, results are identical — 1 executable (6.2%), 8 not executable (50.0%), 7 undetermined/short-with-unknown-borrow (43.8%). Notable: account size from $100 to $100,000 made **zero difference** to executability in this run — the binding constraint was liquidity/shortability, not equity.
- Liquidity distribution: 58.1% of all 129 candidates are in the `<10K` avg-volume bucket; 0% in `250K+`.

**Universe B (live most-actives):** 38 candidates, all in the `500K+` liquidity bucket, 0 passed either screen, 0 signals — most-active large/liquid names simply weren't moving enough on this particular pull to pass either magnitude-based screen.

**Named tickers — checked directly against `data/trade_journal.db`'s `scans` table:**
- **ARTL** (4,561 scan rows), **CAPS** (4,858), **PMAX** (4,506), **VTGN** (4,860), **ATCH** (4,529), **FRGT** (18), **HCWB** (138), **EDVA** (984) — all genuinely present in the journal.
- **HMB** and **EDIA** — **0 rows found, do not appear in the journal.** Not invented for these two; they may be a mis-transcription of HCWB/EDVA (which are present) or may simply never have been scanned.

---

## 13. CURRENT DATA MODEL / JOURNAL

**Live journal** (`trade_journal/db.py`, SQLite WAL): 4 tables —
- `scans`: timestamp, ticker, price, float_shares, relative_volume, percent_change, catalyst, volume, ranking_score, passed, reasons.
- `setups`: timestamp, ticker, timeframe, indication_level, direction, breakout_close, correction_price, continuation_confirmed, entry, stop, target, risk_per_share, reward_per_share, market_structure_trend, llm_reasoning, final_decision.
- `trades`: timestamp, ticker, direction, entry, exit, stop, target, shares, gross_pl, fees, net_pl, mfe, mae, setup_type, result.
- `open_positions`: symbol (PK), direction, shares, entry_price, stop, target, entry_time, entry_fees, max_favorable_price, max_adverse_price — persisted so a systemd restart doesn't lose track of live exposure.

**What's missing from the live journal, relative to the much richer backtest `TradeRecord`** (`backtesting/models.py`): execution tier, participation rate, shortability status, per-account-size results, signal/execution/trade three-way validity split, provenance timestamps (`data_timestamp`/`signal_timestamp`/`execution_timestamp`), rejection chain, dollar volume, liquidity bucket, data quality flag, forward signal outcome (MFE/MAE). None of this research instrumentation is persisted for live trades today — it exists only transiently during a backtest run and in the JSON checkpoint files.

---

## 14. FILE / CODE MAP

| Directory | Files | Purpose | Stability |
|---|---|---|---|
| `ross_strategy/` | config.py, evaluator.py | Bullish screen | Stable, tested |
| `bearish_strategy/` | config.py, evaluator.py | Bearish screen | Stable, tested, but unwired from live |
| `icc_strategy/` | detector.py, models.py | ICC setup detection | Stable, tested |
| `market_structure/` | breaks.py, classifier.py, config.py, models.py, swings.py | Swing/trend/break-of-structure detection underlying ICC | classifier.py/swings.py not re-read this session — presence confirmed only |
| `risk_manager/` | config.py, engine.py, models.py | Deterministic risk/sizing gate | Stable, tested |
| `execution_engine/` | config.py, simulator.py | Paper fill/close simulation | Stable, tested |
| `execution/` | account_sim.py, shortability.py, tiers.py | Research-only: multi-equity simulation, shortability, execution tiers | Stable/tested but shortability is dormant in production |
| `portfolio_manager/` | models.py, portfolio.py | Position/ClosedTrade models, paper portfolio state | Stable, tested |
| `trade_plan/` | config.py, models.py, planner.py | Entry/stop/target construction from an ICC setup | planner.py not re-read fresh this session |
| `trade_journal/` | db.py, journal.py, models.py | SQLite persistence | Stable, tested |
| `scanner/` | data_source.py, liquidity.py, models.py, universe.py | Alpaca data access, point-in-time reconstruction, liquidity profiling, candidate universe assembly | Stable, tested |
| `dashboard/` | app.py, worker.py, config.py, state.py, backtest_runner.py, backtest_state.py | Live FastAPI service + background worker + in-app backtest trigger | Stable, tested |
| `backtesting/` | runner.py, universe.py, data.py, config.py, models.py, outcome.py, integrity.py, narrative.py, experiment.py, historical_movers.py | Full research/backtest pipeline | Actively developed this session (P0 fix); heavily tested |
| `llm_agent/` | client.py, config.py, context.py, models.py, prompts.py, reviewer.py | Ollama advisory review layer | Stable, tested |
| `ranking/` | config.py, models.py, scorer.py | Candidate scoring for dashboard display ordering | Not re-read fresh this session — exact scoring formula UNKNOWN pending verification |
| `analytics/` | breakdown.py, models.py, report.py | Backtest performance stats (win rate, profit factor, expectancy, drawdown) | Not re-read fresh this session — exact formulas UNKNOWN pending verification |
| `market_data/` | alpaca_provider.py, base.py, session.py | Live Alpaca market-data provider + session-boundary helpers | Not re-read fresh this session |
| `config/` | settings.py | Env-based `Settings` (Alpaca keys + paper flag only) | Verified this session |
| `scripts/` | daily_research_checkpoint.py | Cron-driven checkpoint generator producing the JSON files in Section 12 | Referenced, not re-read fresh this session |
| `tests/` | 32 files | Unit tests across nearly every module above | **510 passed, 0 failed** |

---

## 15. KNOWN BUGS / DESIGN PROBLEMS

*(Software/research-integrity ranking only — not a trading-strategy quality ranking.)*

- **CRITICAL** — Live dashboard cannot produce a genuine bearish/short trade from an actual down-mover: `dashboard/worker.py:305` never passes `top_losers` to `build_candidate_universe`, and never imports `bearish_strategy`. "Bearish" in live mode only happens incidentally within bullish-sourced candidates. Anyone reading "bearish signal" in dashboard output should not assume it means what `bearish_strategy` was built to mean.
- **HIGH** — Shortability checking (`execution/shortability.py`) is fully built, tested, and completely unreferenced in any live or paper trading code path (`dashboard/`, `execution_engine/`, `risk_manager/` — confirmed via grep, zero matches). No trade today, long or short, is gated on real borrow availability.
- **HIGH** — Real short-sale mechanics (margin, uptick rule, borrow cost, hard-to-borrow constraints) are entirely unmodeled. Any bearish/short backtest P&L number should be read as "directional paper P&L," not as evidence a real short of that size was actually executable.
- **MEDIUM** — Point-in-time metrics reconstruction has a real, current coverage gap: 28.7% of the latest Universe A run's candidates got `data_quality=INVALID` because their discovery timestamp predates the regular session — not a correctness bug (it fails safe to "no data"), but it materially shrinks the usable sample.
- **MEDIUM** — Backtested spread is a synthetic ~0.1% placeholder, not real historical NBBO. Any statement that "spread never rejected a candidate" in a backtest is close to tautological given this placeholder, not a finding about real spread risk.
- **MEDIUM** — Universe B/C's candidate *set* is always today's live screener state even under a historical `as_of` — only the per-symbol metrics are point-in-time correct. A run using Universe B to ask "who was most-active on date X" would be methodologically invalid.
- **MEDIUM** — Position sizing uses integer share counts with a floor of 1 share and no fractional-share/notional path; this can silently reject small-account + high-priced-stock combinations for sizing reasons that look, from the outside, like a liquidity rejection but aren't.
- **LOW** — `ranking/scorer.py` and `analytics/report.py`/`breakdown.py` internals were not re-verified fresh this session; their exact formulas are UNKNOWN pending re-read (not flagged as broken — flagged as unverified).
- **LOW** — `icc_stage`/`icc_direction`/`aligned_with_trend` on a non-tradeable `TradeRecord` are descriptive, not point-in-time snapshots (by design) — a downstream consumer that doesn't already know this could misread them as causally valid.
- **LOW** — `block_on_uncertain_llm` defaults to `False`, so the LLM advisory layer's "uncertain" verdict (which is also what a dead/slow local model degrades to) does not block execution by default — this is a configuration choice, not a bug, but it means the LLM is currently closer to informational logging than a working gate unless explicitly configured otherwise.

---

## 16. WHAT WE SHOULD NOT CHANGE YET

*(No trading recommendations — purely "needs more evidence/investigation before touching.")*

- The bearish/live-sourcing gap (Section 15, CRITICAL) needs a decision about intended scope (should the live dashboard source real down-movers at all?) before writing code — that's a product decision, not something to infer from the code.
- The point-in-time `INVALID` rate (28.7%) needs characterization across more checkpoint runs before concluding whether it's a stable ~30% ceiling or something that varies a lot by time-of-day/market conditions — one checkpoint is not enough evidence to decide this is a "big problem" vs. "expected and fine."
- `ranking/scorer.py` and `analytics/*.py` should be read fresh before any change is proposed there.
- The integer-floor position-sizing behavior (Section 8/15) — before changing it, it's worth deliberately testing how often it actually zeroes out sizing across a larger sample, rather than assuming it's a significant real-world limiter from this one session's numbers.

---

## 17. WHAT IS THE NEXT TECHNICAL STEP

**PHASE 1 — Fix research/data integrity**
- What: Characterize the `data_quality=INVALID` rate across multiple checkpoint runs (not just one) and decide whether it needs a code fix or is an expected/accepted limitation.
- Files: `scanner/data_source.py`, `scripts/daily_research_checkpoint.py`, `data/research_checkpoints/*.json`.
- Why: sections 12/13/18's numbers all depend on how much of the candidate pool can even be scored.
- Dependencies: none — can start immediately by running the checkpoint script repeatedly and comparing.
- Already partial: the reconstruction logic itself is done and tested; this phase is measurement, not new code.

**PHASE 2 — Fix signal/execution separation (bearish sourcing)**
- What: Decide and implement whether the live dashboard should source genuine down-movers (wire `bearish_strategy` + `top_losers` into `dashboard/worker.py`/`scanner/universe.py`), or explicitly document that live "bearish" only ever means "incidental reversal within a bullish-sourced candidate."
- Files: `dashboard/worker.py`, `scanner/universe.py`.
- Why: this is the single most consequential naming/scope gap found this session.
- Dependencies: none, code already exists (`bearish_strategy`) — this is a wiring decision, not new development.

**PHASE 3 — Improve position sizing / execution modeling**
- What: Decide whether fractional/notional sizing is worth adding, and whether backtested spread should attempt any real historical-quote proxy instead of the fixed synthetic placeholder.
- Files: `risk_manager/engine.py`, `backtesting/runner.py` (`_BacktestMarketData`).
- Why: both are documented, acknowledged simplifications; whether they matter enough to fix depends on Phase 1/2 findings.
- Dependencies: Phase 1's data-coverage numbers help judge whether this is worth prioritizing.

**PHASE 4 — Build bearish/short execution framework (real mechanics)**
- What: Wire `execution/shortability.py` into the live `dashboard/app.py:api_enter` and `dashboard/worker.py` paths (it already exists and is tested); decide whether margin/borrow-cost modeling is in scope at all for a paper-only system.
- Files: `dashboard/app.py`, `dashboard/worker.py`, `execution/shortability.py`.
- Why: currently a fully-built, fully-tested, completely dormant module — the lowest-effort, highest-clarity fix available (wiring existing tested code, not building new code).
- Dependencies: Phase 2's scope decision (no point wiring shortability into a live bearish path that doesn't exist yet).

**PHASE 5 — Run larger research dataset**
- What: Run the historical-mover backfill (`backtesting/historical_movers.py`) across a longer window / broader universe than the current single-checkpoint samples (n=37, n=129) to get statistically meaningful forward-outcome numbers.
- Files: `backtesting/historical_movers.py`, `scripts/daily_research_checkpoint.py`.
- Why: every performance number in Section 12 is explicitly disclosed as small-sample/non-conclusive; nothing here should be trusted as a strategy validation until sample size grows substantially.
- Dependencies: Phases 1–2 should land first so the larger run isn't measuring a known, soon-to-change gap.

---

## 18. FINAL SUMMARY

**CURRENT SYSTEM**

This is a well-tested (510 passing unit tests across 32 files), carefully-documented-in-code research and paper-trading system for a momentum/ICC price-action strategy. Its architecture cleanly separates signal detection, risk/position sizing, and execution simulation, and its backtest pipeline goes to real lengths to avoid look-ahead bias (point-in-time metric reconstruction with two documented, previously-fixed leak sources; causality-filtered continuation trading; forward-outcome measurement kept structurally separate from signal decisions). The live dashboard is a real, working FastAPI service with a background scanner/worker loop, a local-LLM advisory layer, a deterministic risk gate, and a paper execution simulator — but it is bullish-sourced only, and its "bearish" trades are an incidental byproduct of price action within bullish-screened candidates, not a genuine down-mover strategy. The backtest/research pipeline is considerably more complete than the live path: it runs both bullish and bearish screens on every candidate, models execution tiers, multi-account-size accessibility, and (unknown) shortability — none of which the live dashboard currently uses. Recent experiment numbers (n=129 candidates, 16 valid signals, 1 executed) are honest and richly instrumented but are explicitly too small to support any strategy conclusion, and roughly 29% of the candidate pool couldn't even be scored due to a real, current point-in-time data-coverage gap.

**WHAT ACTUALLY WORKS**
- Ross-style bullish screen, bearish screen (as a module), ICC detection — all direction-symmetric, tested.
- Deterministic risk engine with 9 real checks, real position sizing math.
- Paper execution simulation (fills, slippage, stop/target, worst-case tie-break).
- Point-in-time historical metric reconstruction with two real anti-lookahead fixes already applied.
- Full backtest pipeline replaying the exact live-trading code path against real historical bars, with rich research instrumentation (tiers, account-size sweep, forward outcomes, integrity reporting).
- Live FastAPI dashboard with kill switch, manual enter/close, health checks, guardrail visibility, and an in-app backtest trigger.
- 510 passing tests.

**WHAT DOES NOT WORK / IS NOT WIRED**
- No live path sources genuine down-movers for a real bearish trade.
- Shortability checking exists but is called from nowhere in production.
- No real short-sale mechanics (margin, borrow cost, uptick rule) anywhere.
- No fractional-share/notional position sizing.
- Backtested spread is synthetic, not real historical NBBO.
- Universe B/C candidate *sets* aren't historically faithful even when metrics are.

**WHAT IS UNKNOWN**
- Exact internals of `ranking/scorer.py` and `analytics/report.py`/`breakdown.py` (not re-read fresh this session).
- Whether the 28.7% point-in-time-INVALID rate is typical or an outlier (only one recent checkpoint examined).
- Exact fee values in `execution_engine/config.py` (not re-read fresh this session).
- Production logging/persistence configuration beyond scattered `logging.getLogger` calls.

**BIGGEST DATA/RESEARCH RISKS**
- Sample sizes (n=37, n=129) are far too small to draw any conclusion about the strategy.
- ~29% of candidates currently can't be point-in-time scored at all.
- Backtested spread and Universe B/C candidate-set historicity both have known, documented gaps between what's measured and real historical market conditions.

**BIGGEST SOFTWARE RISKS**
- The live/backtest divergence on bearish sourcing and shortability is the single most likely thing to mislead someone reading dashboard or checkpoint output without this document.
- Dormant, tested-but-unwired safety-relevant code (shortability) is an easy thing to forget exists and assume is protecting a trade when it isn't.

**NEXT 5 ENGINEERING TASKS**
1. Run the daily checkpoint script repeatedly over the next several sessions to see whether the 28.7% point-in-time-INVALID rate is stable, and characterize why (pre-market discovery timing, specifically).
2. Decide and document explicitly whether the live dashboard should source real down-movers, or keep "bearish" as incidental-only and rename/relabel it accordingly in the UI/journal.
3. Wire `execution/shortability.py`'s `AlpacaShortabilityChecker` into `dashboard/app.py:api_enter` (lowest-effort, highest-clarity fix — the code already exists and is tested).
4. Re-read `ranking/scorer.py` and `analytics/report.py`/`breakdown.py` in full before making any claims about candidate ordering or performance-stat formulas.
5. Run the historical-mover backfill across a materially larger date range/universe before treating any forward-outcome number as more than a hypothesis.
