# Annualize This research source

## Scope and handling

Annualize This is an owner-authorized, subscription-funded research source.
Its email delivery may be used to form **research hypotheses only**. Do not
store newsletter bodies, charts, attachments, or other paid content in this
repository. Store only the minimal provenance and derived, non-executable
research notes needed to reproduce a test.

Nothing from this source may enter the live execution path directly. A usable
candidate must be converted into a deterministic, timestamped market-data
rule; then pass the governing policy's leakage, out-of-sample,
withdrawal-adjusted drawdown, cost, and paper-trading gates.

## Captured source records

| ID | Received | Subject | Status |
| --- | --- | --- | --- |
| AT-2026-09-13-photonics | 2026-09-13 11:28 ET | *Photonics Thematic - big picture look at one of the market scariest groups and largest TAMs - Updated* | **REJECTED 2026-09-13** |

## Intake contract for the research harness

For each future email, retain outside this repository the original message and
record here only: source ID, received timestamp, subject, declared cadence,
universe/tickers explicitly covered, candidate rule in computable terms, data
availability timestamp, and the test verdict. A record is rejected when its
claim cannot be expressed without discretionary chart interpretation.

## Verdicts

### AT-2026-09-13-photonics — REJECTED 2026-09-13

Tested by `trading_algo/intake.py` (`python -m trading_algo.intake`) under the
handling rules above: levels from closed weekly bars only, no chart reading,
point-in-time listing mask, equal-weight buy-and-hold and equal-time-in-market
controls, costs, and a pre-declared robustness grid.

1. **No timing skill.** The weekly level-transition rule lost to its own
   equal-time-in-market control at every lookback tested (10/13/20/26/52 weeks,
   0 of 5), by −0.07 to −0.17 Sharpe. Turnover is 1.6x/yr and the spread is
   −0.12 to −0.15 across 5–40bp, so cost is not the cause.
2. **The drawdown edge does not replicate.** On the policy's own criterion the
   rule beat the control by +8 to +16 points in sample — and reversed, +14.6%
   over 2000–2013 against −19.3% over 2014–2026. Nothing survived its holdout.
3. **Universe untestable, and unsuitable regardless.** The list was written
   2026-09-13; 10 of 19 names with data traded in 2000, and the 2000–02 optical
   bust appears only through its survivors. The basket's 163x headline is that
   selection, not a sector return. At −51% withdrawal-adjusted drawdown and 22%
   volatility it fails the mandate outright — the existing SPY rule delivers
   −19% at 9%.

Claim 1 of the record (weekly level transition) is rejected as having no
measurable edge. Claim 2 (photonics as an input universe) is rejected as
untestable with available data and unsuitable on the policy's own metrics. No
threshold was tuned after results were seen; the robustness grid was fixed in
advance.
