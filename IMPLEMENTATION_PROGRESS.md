# Implementation Progress

## Phases

| # | Phase | State |
|---|---|---|
| 1 | Audit, scaffold, config, contracts | done |
| 2 | Binance REST + WebSocket + collector | done |
| 3 | Twelve Data + calendar/news providers + fallback | done (calendar providers credential-disabled) |
| 4 | DB, quality engine, indicators, MTF, regime | done |
| 5 | Chatbot, LLM client, memory + post-mortem analysis | done |
| 6 | MCP server, FastAPI, workers | done |
| 7 | Tests | **not done** — `tests/` is empty; `scripts/verify_*.py` covers the ground |
| 8 | Lint, MCP registration, live smoke | done — ruff + mypy clean over 71 files, 11/11 suites |
| 9 | Docs | partial — README, audit, plan, STATUS, NEXT_STEPS |

## Audit pass (2026-08-10)

Ran after the primary provider was switched from Gemini (HTTP 429, quota
exhausted) to AgentRouter. Five defects found, all fixed, all of one class:
a value that was true by construction being presented as a measurement.

| Commit | Fix |
|---|---|
| `6b78ecc` | Reward:risk was pinned at 2.00 by construction |
| `6b3d460` | Agreement score binary by construction; return threshold below its own floor |
| `1729325` | LLM reviewer given the MTF snapshot and regime report the scan already computed |
| `d4cc444` | NO_TRADE leads with the contradiction that decided it, not the first data gap |

AgentRouter token-ceiling fix rides in the `6b3d460` lineage: `max_tokens` 1200 →
8000 on both the AgentRouter and Anthropic adapters, plus a health probe that
exercises the real failure mode instead of a 1-token ping that could not reach it.

## Verification

    scripts/verify_all.py     11/11 suites PASSED
    ruff check src/ scripts/  All checks passed
    mypy src/                 no issues in 71 source files

Live checks performed this pass:

- Closed-candle boundary confirmed on 1m / 15m / 1h / 4h — no forming bar
  reaches the indicator layer (Rule 9).
- Component-score variance confirmed across six symbols: trend, momentum,
  volatility, agreement and alignment all vary. `data_quality_score` reads 1.0
  on every clean symbol, which is a property of the tape, not of the code.
- The 0.5 alignment gate now rejects 3 of 6 scanned symbols. Before the fix it
  rejected nothing at any threshold in (0, 1].

## Production deployment (2026-08-11) — 3e6621f

Commit `3e6621f` "feat: migrate from Gemini to Agentrouter API" pushed to `main`
and deployed to Production via the Vercel GitHub App. Verified live:

- The reviewer is **active in production**: the chat `status` intent probes
  `default_llm_provider().health()` inside the deployed process and demands real
  text back; it reports `agentrouter ok — claude-opus-5 answering`. `vercel.json`'s
  `env` block demonstrably reaches the runtime (the same block supplies the
  reviewer key and the Twelve Data / Alpha Vantage keys).
- **Supabase is the live store**: alembic is at head `7a66cbb55ec9` on
  `aws-0-ap-southeast-1.pooler.supabase.com:6543`; all 17 tables exist;
  55 memories persist across cold starts (a `/tmp` SQLite fallback would reset).
- Binance reports `data-api.binance.vision` — the 451 host failover in
  `rest.py::_get_json` swapping off the geo-blocked `api.binance.com`, exactly as
  designed.
- Both sampled recommendations declined at the deterministic multi-timeframe gate
  (409 NO_TRADE, ~5.5s) — correct selectivity, and the reviewer escalation path
  remains unexercised in production.

Security incident restated: 8 credentials remain public in git history (see
STATUS.md → Security); rotation is the only remedy and is task #0 in NEXT_STEPS.
