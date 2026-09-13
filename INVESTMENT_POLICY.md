# Investment Policy and Operating Mandate

**Owner:** Jason Graham
**Last confirmed:** 2026-09-13
**Status:** governing requirement for all research, backtests, signal design, and future execution work in this repository.

## Objective

Build a deterministic, automated investment strategy that can monitor the
owner's investments and submit trades promptly when its pre-specified signals
change. The primary objective is **minimizing drawdowns**, not maximizing
headline return or Sharpe ratio.

## Cash-flow constraint

This strategy supports an account with a planned **5.4% annual withdrawal
rate**, paid as equal monthly withdrawals:

| Item | Requirement |
|---|---:|
| Annual withdrawal | 5.4% of the account value per year |
| Monthly withdrawal | 0.45% of the applicable annual account value per month |
| Withdrawal frequency | Monthly, equal-dollar installments |

All strategy evaluation must model these withdrawals explicitly. Performance,
maximum drawdown, recovery time, liquidity, and failure scenarios must be
reported **after** modeled withdrawals—not only for a fully accumulated
portfolio.

## Strategy requirements

1. **Deterministic:** every input, calculation, threshold, position target,
   and order action must be reproducible from timestamped market data and
   versioned configuration. No discretionary trade decisions and no LLM/news
   interpretation in the execution path.
2. **Drawdown-first:** rank candidate strategies first by withdrawal-adjusted
   downside outcomes (maximum drawdown, drawdown duration, and recovery under
   the required withdrawal schedule). Return is a secondary constraint, not
   the optimization target.
3. **Execution-ready signals:** use only signals available at the decision
   timestamp; record the data cutoff, target allocation, rationale code, and
   intended order before any order is submitted.
4. **Liquidity-aware:** retain sufficient readily available cash or liquid
   holdings to satisfy scheduled withdrawals without forcing an avoidable sale
   during a drawdown. Test this explicitly.
5. **Conservative constraints:** no leverage, shorting, margin borrowing, or
   options unless Jason explicitly amends this policy in writing. Position and
   turnover limits must be configured and tested before live execution.
6. **Costs and taxes:** backtests must include realistic commissions, spread,
   slippage, and, where applicable, tax assumptions. Assumptions must be named
   alongside results.
7. **Evidence before automation:** a strategy may not trade live solely from
   an in-sample backtest. It requires leakage tests, out-of-sample evaluation,
   paper trading, broker/exchange safety controls, and explicit owner approval
   of the live execution configuration.

## Required evaluation outputs

Every strategy proposal and material parameter change must state:

- the deterministic signal specification and available-at timestamp;
- gross and net returns, volatility, maximum drawdown, drawdown duration, and
  recovery time;
- the same metrics after the 5.4% annual / equal-monthly withdrawal simulation;
- historical periods in which withdrawals would have depleted the account or
  required a forced sale;
- turnover, modeled trading costs, liquidity assumptions, and concentration;
- out-of-sample and paper-trading evidence; and
- the exact configuration and code version used.

## Change control

Do not silently relax this mandate to improve a backtest. Any change to the
withdrawal rate, cadence, drawdown priority, leverage restriction, or live
trading authority must update this document with Jason's explicit decision and
the confirmation date. Update `Last confirmed` whenever Jason reaffirms or
changes the mandate.

## Current implementation boundary

This repository currently contains research and backtesting code. It does not
yet authorize or implement broker connectivity or live order submission. Future
execution work must preserve this policy as the governing acceptance criteria.
