# Investment Policy and Operating Mandate

**Owner:** Jason Graham
**Last confirmed:** 2026-09-13 (options scope and withdrawal definition amended 2026-09-13)
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

*Definition confirmed 2026-09-13.* The rule above admitted three readings that
give materially different results. Jason confirmed the intended one:
**the annual withdrawal is set on 1 January at 5.4% of the account value on that
date, and paid in twelve equal dollar instalments over the following year.**
Income therefore varies year to year with the account and resets lower after a
bad year. This is `annual_reset` in `trading_algo/mandate.py`, which is the
default; the two rejected readings (`fixed_nominal`, `pct_current`) remain
implemented and are reported as sensitivity, because the choice moves maximum
drawdown by up to 30 points for some strategies.

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
5. **Conservative constraints:** no leverage, shorting, or margin borrowing.
   Position and turnover limits must be configured and tested before live
   execution.

   *Options — amended 2026-09-13 by Jason's explicit decision.* Jason authorised
   covered call writing, and subsequently the purchase of puts, for evaluation.
   Both were evaluated over 2000–2026 and **both are excluded from the strategy
   on the evidence**:

   - Covered calls: CBOE BXM scores Sharpe 0.32 against 0.41 for the index
     itself; the call leg costs −3.78%/yr. Overlaid on the strategy it takes
     Sharpe 0.70 → 0.65.
   - Long puts: derived exactly from real BXM prices via put-call parity, a
     1-month ATM protective put is Sharpe-neutral (0.43 vs 0.43). It halves
     volatility and halves return. Bolted onto the timing rule it *subtracts*
     (0.87 → 0.65), because holding cash removes the same downside at ~0.5%/yr
     against the put's ~3%/yr.

   No option position is used by the current strategy. Writing calls or buying
   puts in live execution requires a further amendment here. Selling puts,
   spreads, and any short option position other than a covered call remain
   prohibited. See `trading_algo/options.py` and the README.
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

Jason authorized preparation for future live trading on 2026-09-13. Account
credentials, account access, and authority to submit orders have **not** been
provided. The repository may therefore build and test non-secret execution
readiness, but it must not connect to an account, retrieve account data, or
submit an order until Jason separately provisions the necessary account access.

This policy remains the governing acceptance criteria for all future execution
work.
