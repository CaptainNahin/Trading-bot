# Next Steps

Ordered by what buys the most, not by effort.

## 1. Write the pytest suite

`tests/` is empty. `scripts/verify_*.py` covers the same ground and runs green,
but as scripts: no fixtures, no parametrisation, no CI hook. Port them, starting
with the ones that guard arithmetic — `verify_indicators`, `verify_regime_mtf`,
`verify_scanner` — because those are the ones a refactor can silently break.

Add a regression test per fixed defect in STATUS.md. Defects 2 and 3 in
particular were invisible for the same reason: nothing asserted that a score
could take more than one value. A test that enumerates the input space and
asserts the output varies would have caught both.

## 2. Settle enough trades to say anything about accuracy

The observed win rate is honest and useless at n=1. Thirty decided trades is the
point where it stops being noise. Nothing to build here, only to run.

## 3. Calibration

Once there is a settled history, fit a calibration model on unseen data so
`calibrated_probability` can stop being `None`. Until then no number in the
system may be called a probability.

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
