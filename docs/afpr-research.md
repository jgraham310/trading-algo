# AFPR point-in-time research lane

This is event research, **not a portfolio strategy** or evidence of a profitable
AFPR service. The investment policy is unchanged: any portfolio proposal still
needs withdrawal-adjusted drawdown, liquidity and all validation gates. No
broker access, position sizing, options model, short execution or orders exist
in this lane. Stock returns do not establish option profitability; short tests
would additionally need borrow availability, recalls, financing and gap risks.

## Capture before outcomes

`trading_algo.afpr.Observation` records a source URL, hand-drawn vs engine vs
reconstruction provenance, symbol, timeframe, pattern, observed timestamp,
market-data cutoff, trigger/invalidation/target and explicit confirmation rule.
Use timezone-aware timestamps. `capture(directory, observation)` exclusively
creates numbered JSON files with actual ingestion `recorded_at`. Never backdate
observations to a chart date. Keep source screenshots/exports with their capture
time independently; a user-supplied observed timestamp is not proof of history.
The filesystem is append-only by application convention, not tamper-proof.

Capture all candidates, including rejected/watch cases and reasons, before
results are known. Rejected malformed levels may be null or nonpositive (convert
NaN to null, since JSON refuses NaN). Revisions get new observation timestamps
and increasing revision numbers; prior files are preserved. Cohorts are keyed
by **signal_id**, not revision: freeze one eligible revision before evaluation,
and never count later target edits as new independent successes. No genuine
historic AFPR observations are bundled. Synthetic tests are reconstructions.

## Event study v1

Pass `event_study(observation, bars, horizon=20, cost_bps=10, benchmark=spy)`
explicit session `open_at`, `close_at`, `open`, `close` columns. Supply the full
exchange calendar and consistent split-adjusted prices; this module cannot
identify omitted sessions, corporate actions or disappeared securities. Missing
or nonpositive prices raise instead of being forward-filled. Include delisted
and failed names in the research dataset or disclose unavailable outcomes.

The source observation must already establish confirmation. First eligible fill
is the first supplied open **strictly after** observation, including intraday
observations. The trigger is not rechecked at entry: this is a preconfirmed
thesis, not a new breakout scanner. Skip opens at/beyond invalidation or target.
Otherwise close-based target/invalidation exits fill at the next actual open,
including adverse gaps; the levels are not guaranteed execution prices. After
20 closes (configurable in advance), exit next open. Intraday wicks do not exit.
Pending entry/exit/horizon cases remain unresolved with null returns, not zero.

Costs apply separately to entry and exit notionals. The default 10 bps per side
is an illustrative commission/half-spread/slippage assumption, not a calibrated
liquidity estimate. Report sensitivity before drawing conclusions. No dividends,
taxes, borrow fees or option premiums are modeled. R uses initial entry-to-stop
risk; gaps can lose more than 1R. Excursions are explicitly **close-sampled** plus
exit gap, not intraday MFE/MAE. Benchmark net returns use exactly the same entry
and exit opens and two-sided costs, otherwise remain null.

Freeze horizon, cost assumptions, universe inclusion and a chronological holdout
cutoff before inspecting outcomes. Report total captured, rejected, skipped,
pending and resolved counts separately. Don't summarize only winners or infer
portfolio Sharpe from overlapping events. Keep exploratory and holdout results
separate; no automated optimizer or historical AFPR performance claims here.

## Existing-engine corrections

Daily engine now charges the initial entry from cash and includes initial equity
in drawdown. Terminal liquidation is still not forced. Its legacy same-close
fill convention is explicitly idealized and must not be used for AFPR news.
Weekly intake removes an extra weekly lag, but retains Friday availability
labels: without an exchange calendar, a holiday Friday conservatively delays
availability until the next observed decision day, then the daily execution lag.
This is intentional, not a hindsight Thursday publication. Legacy printed
research numbers were not recomputed and should not be treated as refreshed.

Validation (no market downloads required):

```bash
python -m pytest test_afpr.py test_backtest.py
```
