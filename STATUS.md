# QuantEdge Live Market Gateway — Status

Last updated: 2026-08-10 (audit pass)

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

## Provider health (live at last check)

    binance       ok    api.binance.com
    twelvedata    ok
    alphavantage  ok
    agentrouter   ok    claude-opus-5
    gemini        error HTTP 429 (quota exhausted)
    anthropic     error HTTP 401
    oanda/fmp/finnhub  disabled — no credentials

`LLM_PROVIDER=agentrouter`.

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
- **Observed win rate stays `sample_too_small`** until 30+ trades settle.
- Selectivity changes reduce how many setups are shown. They do not make a shown
  setup more likely to be right.

## Security

- `.env` is git-ignored; no secret has been printed or committed.
- Binance access is public market-data endpoints only. No account, wallet,
  margin, futures or order-placement surface.
- **The Binance key was transmitted in plaintext chat and should be rotated.**
