# QuantEdge Live Market Gateway — Status

Last updated: 2026-08-11 (production deployment verified)

## Working

| Area | State |
|---|---|
| Market data | Binance public REST + WebSocket, live. Closed candles only. |
| Fallback providers | Twelve Data ok, Alpha Vantage ok. OANDA / FMP / Finnhub disabled (no credentials). |
| Data quality | 15 checks, gating on PASS/WARN/FAIL. |
| Indicators | Deterministic, in code. No LLM arithmetic. |
| Multi-timeframe | Execution / confirmation / regime roles, weighted 0.25 / 0.35 / 0.40. |
| Regime | 9 deterministic categories. |
| Scanner | 12-step pipeline, composite scoring, rejection reasons returned. |
| LLM review | AgentRouter (`claude-opus-5`), live and answering. |
| Signals | `generate_trade_recommendation`, memory-augmented. |
| Settlement | Outcome evaluation, observed win rate with `sample_too_small` below 30 decided. |
| Memory | Win → recorded. Loss → post-mortem cause, then recorded, then derived rules. |
| Chatbot | Intent routing, time-limit selection, conversational outcome reporting. |
| REST + MCP | FastAPI surface and MCP tool server, both verified. |
| Verification | `scripts/verify_all.py` — 11/11 suites passing. ruff + mypy clean, 71 files. |

## Production (verified 2026-08-11)

Deployed commit `3e6621f` at `https://trading-bot-six-kohl.vercel.app`, via the
Vercel GitHub App on push to `main`.

| Check | Result |
|---|---|
| HTTP Basic auth | Enforced — unauthenticated root returns 401 |
| `GET /api/v1/health` | 200 in 11.1s |
| `POST /api/v1/bot/chat` | 200 in 15.0s, answered |
| Reviewer, probed from inside the runtime | `agentrouter ok — claude-opus-5 answering` |
| Supabase schema | At head `7a66cbb55ec9`; all 17 tables present |
| Persistence | 55 memories survive cold starts, so Postgres is in use, not `/tmp` SQLite |
| `POST /bot/trade-recommendation` | 409 NO_TRADE in 5.3–5.6s (agreement gate) |

The reviewer question is settled by the chat `status` intent, which calls
`default_llm_provider().health()` in the deployed process: the probe demands real
text back, so `ok` means the model answered there and then. The two 409s above
were decided by the deterministic agreement gate *before* escalation, so they say
nothing about the reviewer either way — hence the separate probe.

The runtime reports Binance at `data-api.binance.vision` while `vercel.json` sets
`api.binance.com`. That is the 451 host failover in `rest.py::_get_json` doing its
job, not an unapplied variable: `data-api.binance.vision` is the first fallback
host, and the same env block is demonstrably live, since the reviewer credential
and both fallback market-data keys reach the runtime from it.

## Provider health (live in production, 2026-08-11)

    binance       ok        data-api.binance.vision (451 failover from api.binance.com)
    twelvedata    ok        free-tier quota applies
    alphavantage  ok        news feed, 50 items in probe
    agentrouter   ok        claude-opus-5 answering
    oanda         disabled  OANDA_API_TOKEN, OANDA_ACCOUNT_ID absent
    fmp           disabled  FMP_API_KEY absent — no economic calendar
    finnhub       disabled  FINNHUB_API_KEY absent — no economic calendar

`LLM_PROVIDER=agentrouter`. Gemini and Anthropic are no longer configured; their
keys were removed from the deployment in `3e6621f`.

## Defects found and fixed in the audit pass

Each was a value that was true by construction being presented as a measurement.

1. **AgentRouter token ceiling** (`6b3d460` lineage). `max_tokens=1200` was consumed
   entirely by the model's reasoning block, so every review returned
   `stop_reason=max_tokens` with no text and was silently discarded — while a
   1-token health probe reported `ok`. Raised to 8000; the health probe now asks
   for a real completion and fails if no text comes back. Same latent defect
   fixed in the Anthropic adapter.
2. **`alignment_score` binary by construction** (`6b3d460`). Divided by the weight
   that voted rather than the weight available; since the branch only runs when
   there are no conflicts, the ratio was 1.0 whenever it was not 0.0. A lone
   execution timeframe with both higher timeframes silent scored a perfect 1.0
   and passed the gate. Now varies; 3 of 6 symbols are correctly rejected.
3. **Dead score threshold** (`6b3d460`). `min_heuristic_score_to_return: 0.20` sat
   below the arithmetic floor of 0.275, so it rejected nothing. Raised to a
   derived 0.60, matched by the LLM escalation threshold.
4. **Reviewer starved of context** (`1729325`). `signal.py` built the LLM context
   without the multi-timeframe snapshot or regime report, so the reviewer was
   told on every call that both were "not available" — moments after the scan
   computed them. `ScanResult` now carries both. An ETHUSDT review went from
   2 supporting items to 8 supporting / 11 contradictory.
5. **Decline reason mislabelled** (`d4cc444`). The headline reason was
   `missing_information[0]`, and the two permanent gaps sort first, so every
   NO_TRADE was announced as "Liquidity session state not available" while the
   real cause was demoted. NO_TRADE now leads with the contradiction that
   decided it; INSUFFICIENT_DATA still leads with the gap.
6. **Review timeout exceeded the serverless wall clock.** The reviewer waited
   180s; Vercel kills the request at 60s. `signal.py` already degrades a failed
   review to the deterministic candidate, but that path could never run in
   production — the host killed the process first, returning an empty 504
   instead of an answer. The timeout is now `LLM_TIMEOUT_SECONDS`, set to 30 in
   production against a measured 15.9s scan, so the review expires inside our
   own process and the degrade path is reachable.

## Known limitations

- **`tests/` is empty.** The Phase 7 pytest suite was never written. The
  `scripts/verify_*.py` suites cover that ground and run green, but they are
  scripts, not parametrised tests.
- **Event risk is permanently `UNKNOWN`.** `_event_risk_for` is a documented stub
  and both calendar providers are credential-disabled. Reported as absent, never
  guessed.
- **No `SessionState` producer exists.** Nothing measures order-book depth or
  session liquidity, so the field is not populated rather than asserted.
- **No calibration model.** No confidence number is presented as a probability.
- **Observed win rate is 47.3% over 55 settled trades** (26 wins / 29 losses) as
  of 2026-08-11. It is above the 30-trade threshold, so it is reported as a real
  observation rather than `sample_too_small`. It is a record of what happened, not
  a forecast, and it is not the probability of the next trade winning.
- Selectivity changes reduce how many setups are shown. They do not make a shown
  setup more likely to be right.
- **The reviewer escalation path is unexercised in production.** Every
  recommendation sampled on 2026-08-11 declined at the deterministic
  multi-timeframe gate before reaching the LLM. The reviewer is reachable and
  answering — probed directly — but no production request has yet run the full
  escalate-and-review path end to end.
- **Request latency is close to the serverless ceiling.** A chat request measured
  15.9s in production and 71–77s from a local machine; the difference is network
  distance to Binance and Supabase, not compute. The Vercel Hobby ceiling is 60s
  and `maxDuration` is set to it. A production request that escalates to the
  reviewer has ~30s of headroom by design, and abandons the review rather than
  the request if it runs out.

## Security

- Binance access is public market-data endpoints only. No account, wallet,
  margin, futures or order-placement surface.
- `.env` is git-ignored and no secret was printed to a log or a terminal by this
  codebase.

### Open incident: eight credentials are public

`vercel.json` is git-tracked and `CaptainNahin/Trading-bot` is a **public**
repository, so every value in its `env` block is world-readable. An
unauthenticated fetch of the `raw.githubusercontent.com` URL returns HTTP 200.

Exposed, and each needing rotation: `BINANCE_API_KEY`, `BINANCE_API_SECRET`,
`DATABASE_URL` (Supabase password inline), `AGENTROUTER_API_KEY`,
`ANTHROPIC_API_KEY`, `TWELVE_DATA_API_KEY`, `ALPHA_VANTAGE_API_KEY`,
`API_AUTH_TOKEN`. A GitHub PAT also sits in plaintext in `.git/config`.

`3e6621f` removed `BINANCE_API_SECRET`, `ANTHROPIC_API_KEY` and the Gemini keys
from `HEAD`, but **git history still holds every one of them** — removing a secret
from the current file does not unpublish it. Rotation at the provider is the only
remedy; the five values still in `HEAD` remain live.

The fix is to move these into Vercel's encrypted environment store and delete the
`env` block from `vercel.json`. That was not done here because the supplied Vercel
token has no access to this project (`/v9/projects` returns `[]`, `/v2/teams`
returns 403), so the store is unreachable from this session.
