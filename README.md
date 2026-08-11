# QuantEdge Live Market Gateway

A live-market intelligence backend that produces trade *recommendations* from
real market data — never from invented values. A deterministic scan does all
numerical work in code; an LLM reviews the result and can only ever make the
system more conservative.

**It declines more often than it recommends. That is the design, not a fault.**

## What it does

    market data  ->  quality gate  ->  indicators  ->  multi-timeframe  ->  regime
                                                                              |
                          recommendation  <-  LLM review  <-  composite score

- **Deterministic first.** Every indicator, score and threshold is computed in
  Python. The LLM never does arithmetic (see the accuracy rules below).
- **Closed candles only.** A forming bar never reaches the indicator layer, on
  any timeframe. No look-ahead, no repainting.
- **Memory.** A win is recorded. A loss is first diagnosed — *why* did it lose —
  and only then recorded, with the cause. Rules are derived from losses.
- **Refuses to guess.** `NO_TRADE` when evidence is weak, `INSUFFICIENT_DATA`
  when inputs are missing or stale.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/api/v1/health` | Provider health board |
| `POST` | `/api/v1/bot/chat` | Conversational interface |
| `POST` | `/api/v1/bot/trade-recommendation` | Structured recommendation |
| `GET` | `/api/v1/bot/time-limits` | Selectable expiries |
| `GET` | `/api/v1/bot/memories` | Recorded outcomes |
| `GET` | `/api/v1/bot/memory-stats` | Observed statistics |

All routes sit behind HTTP Basic auth (blank username).

## Running locally

    uv sync
    cp .env.example .env      # then fill in the keys you have
    uv run uvicorn quantedge.api.app:app --reload

Leave `DATABASE_URL` blank to use the bundled SQLite file. Any provider whose
key is absent disables itself cleanly — the gateway degrades, it does not crash
and it does not invent data.

Verification:

    uv run python scripts/verify_all.py     # 11 suites
    uv run ruff check src/ && uv run mypy src/

## Deployment

Serverless on Vercel (`api/index.py`), Postgres on Supabase. Pushing to `main`
triggers a production deploy through the Vercel GitHub App.

Schema:

    uv run alembic upgrade head

### Configuration

Secrets belong in **Vercel's encrypted environment store**, not in `vercel.json`
— that file is committed and therefore public. See [SECURITY.md](SECURITY.md).

| Variable | Notes |
|---|---|
| `LLM_PROVIDER` | `agentrouter` \| `anthropic` \| `gemini` \| `disabled` |
| `AGENTROUTER_API_KEY` | Reviewer credential |
| `AGENTROUTER_MODEL` | `claude-opus-5` |
| `LLM_TIMEOUT_SECONDS` | Abandon a slow review, keep the request. Must sit below the host's request ceiling |
| `DATABASE_URL` | Supabase DSN; blank falls back to SQLite |
| `BINANCE_API_KEY` | Optional. Raises rate limits only |
| `QUANTEDGE_UI_PASSWORD` | Dashboard password |

Binance access is **public market-data endpoints only**. No account, wallet,
margin, futures or order-placement surface exists in this codebase, so no API
*secret* is required — only the key, and only for rate limits.

## Accuracy rules

These are enforced in code, not by convention.

1. No claim of guaranteed accuracy is ever made.
2. Nothing is fabricated — not prices, candles, indicators, news, events,
   spreads, payouts, results, probabilities or win rates.
3. No confidence number is called a probability unless calibrated on unseen
   data. None currently is, so none is.
4. `NO_TRADE` when evidence is weak.
5. `INSUFFICIENT_DATA` when data is unavailable, stale or invalid.
6. No Martingale, loss-doubling or recovery staking. No automated execution.
7. No broker login automation.
8. No scraping.
9. No future or incomplete candles in any calculation.
10. All numerical work is deterministic code. The LLM reviews; it does not compute.

The observed win rate is reported as an observation over settled trades and is
flagged `sample_too_small` below 30 — it is never presented as the probability
of the next trade winning.

## Status

See [STATUS.md](STATUS.md) for the current state, [NEXT_STEPS.md](NEXT_STEPS.md)
for what is outstanding, and [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md)
for the build log.

## Disclaimer

This software produces analysis, not financial advice. Trading carries risk of
loss. Nothing here predicts the market.
