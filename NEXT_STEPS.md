# Next Steps

Ordered by what buys the most, not by effort.

## 0. Rotate the eight exposed credentials

They are public in git history right now (see STATUS.md → Security). Rotation is
the only remedy — deleting the keys from the current file does not unpublish what
history still holds. New keys must go into Vercel's encrypted environment store,
never back into `vercel.json`, whose `env` block should then be deleted. The
store is reachable from the Vercel dashboard; the token supplied to this session
has no access to the project.

## 1. Write the pytest suite

`tests/` is empty. `scripts/verify_*.py` covers the same ground and runs green,
but as scripts: no fixtures, no parametrisation, no CI hook. Port them, starting
with the ones that guard arithmetic — `verify_indicators`, `verify_regime_mtf`,
`verify_scanner` — because those are the ones a refactor can silently break.

Add a regression test per fixed defect in STATUS.md. Defects 2 and 3 in
particular were invisible for the same reason: nothing asserted that a score
could take more than one value. A test that enumerates the input space and
asserts the output varies would have caught both.

## 2. Calibration

The observed win rate passed the 30-trade threshold on 2026-08-11: 47.3% over 55
settled trades. That is now a real measurement, and it is the input calibration
needs. Fit a calibration model on unseen data so `calibrated_probability` can stop
being `None`. Until that exists, no number in the system may be called a
probability — including the 47.3%, which describes trades already closed and says
nothing about the next one.

## 3. Exercise the reviewer end to end in production

Every production recommendation sampled so far declined at the deterministic
multi-timeframe gate, so the escalate-to-reviewer path has never run under the
serverless wall clock. The reviewer answers when probed directly, and the 30s
`LLM_TIMEOUT_SECONDS` is set below the 60s host ceiling by design, but the
combined latency of scan-then-review has not been measured in production. Worth
capturing the first request that does escalate.

## Optional, in rough order of value

- **Event risk.** Wire a real economic-calendar provider so `_event_risk_for`
  stops returning `UNKNOWN`. Needs a credential.
- **Session liquidity.** Implement a `SessionState` producer, or delete the
  contract. Right now it is a field nothing can fill.
- **Order-flow depth.** The reviewer notes its absence on every call: no funding,
  open interest, or order-book depth. Would add real evidence rather than more
  of the same.
- **Backtest harness.** Deterministic replay over stored candles, to test scoring
  changes against history instead of against the current tape.
- **Data-quality score variance.** Currently 1.0 on every clean Binance symbol.
  Not wrong, but it means the 0.10 composite weight contributes a constant in
  practice. Worth confirming the degraded paths fire on real bad data.

## Do not

- Do not hardcode a win rate, or gate on one. It is an observed outcome.
- Do not add broker execution, Quotex automation, or Martingale-style staking.
- Do not add private Binance endpoints.
