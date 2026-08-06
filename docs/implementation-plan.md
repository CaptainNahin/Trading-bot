# Implementation Plan — QuantEdge Live Market Gateway

Derived from `docs/current-state-audit.md`. Ordered by dependency, not by convenience.

---

## Guiding constraints

These are not aspirational; they are enforced in code and in tests.

1. **No fabricated market data.** Every number leaving the system carries a `provider` and a
   timestamp. If a provider cannot answer, the response is a structured failure, never a plausible
   guess.
2. **Deterministic math lives in Python, never in the LLM.** The LLM receives computed features and
   may only reason about them. `services/indicators.py` is pure, tested, and reproducible.
3. **No incomplete candle is ever treated as history.** `is_closed` is a first-class field; the
   quality engine and every indicator path filter on it.
4. **Transport-independent business logic.** MCP and FastAPI are thin adapters over one service
   container. A tool and its matching HTTP route call the *same* function.
5. **Weak evidence ⇒ `NO_TRADE`. Missing/stale/invalid data ⇒ `INSUFFICIENT_DATA`.** Never a
   confident-sounding default.
6. **No probability language without calibration.** Deterministic outputs are named
   `heuristic_score`, `trend_score`, etc. `calibrated_probability` stays `null` until a real
   calibration model trained on unseen data exists.
7. **Forbidden by design:** Martingale / recovery staking, broker execution, Quotex login
   automation, TradingView scraping, private Binance endpoints, arbitrary-shell / unrestricted-FS /
   unrestricted-SQL MCP tools.

---

## Phase 1 — Foundation

**Goal:** a runnable, typed, configurable skeleton.

- `pyproject.toml` with pinned dependency ranges; uv-managed venv; ruff + mypy + pytest config.
- `src/quantedge/config.py` — `pydantic-settings` for env vars, YAML loaders for
  `config/providers.yaml`, `symbols.yaml`, `sessions.yaml`, `scanner.yaml`. Production-mode
  validation (refuses wildcard CORS, refuses missing secrets when `APP_ENV=production`).
- `src/quantedge/logging.py` — structured JSON logging with a **redaction filter** that scrubs any
  registered secret value and common key patterns from messages *and* exception tracebacks.
- `src/quantedge/errors.py` — `QuantEdgeError` hierarchy with safe, client-facing `to_dict()` that
  never leaks credentials or internal paths.
- `src/quantedge/contracts/` — strict Pydantic v2 models: `Candle`, `Quote`, `OrderBook`, `Trade`,
  `SymbolInfo`, `DataQualityReport`, `FeatureSnapshot`, `RegimeReport`, `EconomicEvent`,
  `EventRiskReport`, `ScanCandidate`, `SignalContext`, `LLMDecision`, `ProviderHealth`.
  `Decimal` for prices, timezone-aware UTC for all timestamps, `extra="forbid"` on LLM output.

**Exit criteria:** `python -c "import quantedge"` succeeds; contract unit tests pass.

---

## Phase 2 — Binance (always-on, keyless)

**Goal:** live crypto data with no credentials.

- `providers/http.py` — one resilient async client: timeouts, exponential backoff **with jitter**,
  rate-limit (429/418) handling with `Retry-After`, circuit breaker, request de-duplication.
- `providers/binance/rest.py` — `data-api.binance.vision` only. `exchangeInfo`, `ticker/price`,
  `ticker/24hr`, `ticker/bookTicker`, `klines`, `trades`, `depth`.
- `providers/binance/ws.py` — `wss://stream.binance.com:9443` combined streams (`@kline_`,
  `@bookTicker`), heartbeat/ping monitoring, stale detection, auto-reconnect with backoff.
- `services/streams.py` — collector: dedup by `(symbol, interval, open_time)`, preserves provider
  event timestamps, stores **closed** candles as history and keeps the forming candle separate.
- **Explicitly excluded:** orders, account, wallet, margin, futures, signed requests. Enforced by a
  unit test that asserts no signing code and no private path exists in the module.

**Exit criteria:** live REST calls return real data; a live WS session receives ≥1 kline message;
forced disconnect reconnects.

---

## Phase 3 — Keyed providers

- `providers/twelvedata.py` — forex/stock/index/commodity/crypto quotes + candles. Credential
  validation, free-plan limits documented (8 req/min, 800/day) and enforced by the rate limiter.
- `providers/oanda.py` — practice-environment pricing/candles/instruments. **Read-only; no order
  endpoints.** Self-disables cleanly when unconfigured (current state).
- `providers/fmp.py`, `providers/finnhub.py`, `providers/alphavantage.py` — economic calendar and
  news, each declaring exactly which endpoints its plan exposes.
- **Credential probe** (`scripts/probe_credentials.py`): for the two unlabelled keys, issue one
  minimal authenticated request per candidate vendor and record which authenticates. Adapters are
  wired only to whatever actually answers; findings go in `docs/providers.md`.
- `providers/registry.py` — priority + fallback (OANDA → Twelve Data for forex; Binance for crypto).
  Every returned object records which provider served it; **no silent mixing**.

**Exit criteria:** health check per provider returns a truthful `ok / degraded / disabled / error`.

---

## Phase 4 — Storage and deterministic analytics

- `repositories/models.py` + `migrations/` — Alembic over SQLAlchemy for all 15 required tables,
  UTC timestamps, `(provider, symbol, timeframe, open_time)` unique constraint on candles,
  append-only `audit_logs`, immutable `settled_signals`.
- `repositories/memory.py` — in-memory fallback that reports `persistence_available=false` loudly.
- `services/quality.py` — the 15 required checks → `PASS | DEGRADED | FAIL` + `quality_score`.
  **A `FAIL` blocks all candidate and LLM output.**
- `services/indicators.py` — returns, log returns, EMA 9/20/50/200, SMA 20/50/200, RSI 14,
  MACD 12/26/9, ATR 14, ADX 14, Bollinger 20/2, ROC, rolling volatility, volume change, body/wick
  ratios, MA slopes, distance from rolling extremes. Wilder smoothing where canonical. Warm-up
  periods documented and asserted.
- `services/structure.py` — confirmed swings (no forward-looking pivots), HH/HL, LH/LL, breakout and
  failed-breakout candidates.
- `services/mtf.py` — execution / confirmation / regime timeframe triples per horizon; higher-TF
  candles marked `is_closed=false` are flagged, never blended into history.
- `services/regime.py` — the 9-state deterministic classifier with evidence and contradictions.

**Exit criteria:** indicator values verified against hand-computed fixtures.

---

## Phase 5 — Scanner and LLM abstraction

- `services/scanner.py` — the 12-step deterministic pipeline; emits `heuristic_score`,
  `trend_score`, `momentum_score`, `volatility_score`, `data_quality_score`,
  `evidence_agreement_score`, plus an explicit `rejections` list with reasons.
- `services/signal_context.py` — assembles the LLM input contract (verified data only, including an
  explicit `missing_information` list).
- `providers/llm/base.py` + `agentrouter.py` + `anthropic.py` — separate adapters, **not** assumed
  wire-compatible. Selected by `LLM_PROVIDER`.
- `services/llm_review.py` — strict JSON validation of the output contract. Rejects malformed
  output, rejects any attempt to upgrade a failed quality status, forces
  `calibrated_probability=null` while no calibration model is registered.

**Exit criteria:** fabricated/malformed LLM responses are rejected by tests.

---

## Phase 6 — Transports

- `mcp_server/server.py` — all 21 tools on the **current installed** MCP Python SDK API (verified by
  inspecting the installed package, not tutorials). stdio transport.
- `api/` — the 15 FastAPI routes calling the identical service functions, OpenAPI generated,
  environment-driven CORS (never `*` in production), auth hooks stubbed for future accounts.
- `workers/` — 7 independently startable workers.

**Exit criteria:** MCP server starts and lists 21 tools; FastAPI starts and serves `/health`.

---

## Phase 7 — Quality gates

pytest (unit / contract / integration / live markers), ruff, mypy, secret scan, Dockerfile authored
(build **not** runnable — no Docker on this host, reported as such).

---

## Phase 8 — Registration and live verification

Register `quantedge-live-market` at **project scope** via `.mcp.json` using absolute Windows paths.
Then actually run: `provider_health`, `get_live_quote("BTCUSDT")`, `get_candles("BTCUSDT","1m",100)`,
start a stream, confirm messages arrive, stop cleanly. Failures are reported as failures.

---

## Phase 9 — Documentation and completion report

All required docs plus `CLAUDE.md`, then a truthful completion report distinguishing
**verified**, **not verified**, and **blocked**.

---

## Known-blocked items (carried, not hidden)

| Item | Reason | Needed from user |
| --- | --- | --- |
| PostgreSQL/Supabase persistence | `sbp_` token is a Management-API PAT, not a DB credential; no project URL supplied | `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (JWT), or a direct `DATABASE_URL` |
| OANDA forex | No token/account supplied | `OANDA_API_TOKEN`, `OANDA_ACCOUNT_ID` |
| Live LLM review | No AgentRouter or Anthropic key supplied | `AGENTROUTER_API_KEY` (+ base URL/model) or `ANTHROPIC_API_KEY` |
| `docker build` | Docker not installed on this host | Install Docker Desktop |
| Calibrated probabilities | No labelled outcome history exists yet | Time + settled signals; until then `calibrated_probability` stays `null` |
