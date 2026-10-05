# trading-algo

Forecasting engine: [TimesFM 3.0](https://github.com/google-research/timesfm/releases/tag/v3.0.0).

```bash
uv venv && uv pip install -e .
```

```python
from trading_algo.timesfm_signal import load_closes, forecast, signal

closes = load_closes("SPY").values     # must end at the last CLOSED bar
f = forecast(closes, horizon=5)        # f.path, f.quantiles (h, 9), f.expected_return
pos = signal(closes, horizon=5)        # target position in [-1, 1] for the NEXT bar
```

Or `python -m trading_algo.timesfm_signal SPY`.

`signal()` only takes a side when the pessimistic quantile agrees with the median
(`conf=0.3`), then sizes by `expected_return / scale`. `scale=0.02` is a
placeholder — tune it per instrument and horizon.

## Backtest

```bash
python -m trading_algo.backtest SPY                # 5y, caches to wf_SPY.csv
python -m trading_algo.backtest SPY 2000-01-01     # full history vs MA200
```

Every strategy, the MA200 benchmark and buy & hold are scored by the same
`_stats()` on the same bars with the same cost model, so the columns are
comparable. `evaluate(..., prices=)` needs enough history to warm the MA up
before the first decision bar — it raises rather than scoring an un-warmed MA.

Two stages, because inference is the slow part and thresholds are the part you
want to iterate on:

```python
from trading_algo.backtest import walk_forward, evaluate, report
wf = walk_forward(load_closes("SPY", period="5y"))   # slow, once
wf.to_csv("wf_SPY.csv")
for conf in (0.2, 0.3, 0.4):                          # free
    print(conf, evaluate(wf, conf=conf)["sharpe"])
print(report(evaluate(wf)))
```

`walk_forward` forecasts at every bar from a rolling window ending *at* that
bar. Positions earn the next bar's return; `cost_bps` is charged on position
changes. Read the caveats `report()` prints — pretraining contamination is the
big one.

## Weinstein 4-stage model

```bash
python -m trading_algo.stage SPY 20          # ticker, years, RS benchmark (default SPY)
python -m trading_algo.stage NVDA 20 SPY
python -m trading_algo.stage --demo          # self-check, no download
```

Weekly bars, long in Stage 2 (close > 30W MA, MA rising vs 3 weeks ago, Mansfield
RS > 0), flat otherwise, held over the following week's daily bars. The stage for
week `t` is shifted before it is traded — it is only knowable at that week's close.

**RS against yourself is always 0**, so `SPY` with the script's default `SPY`
benchmark can never reach Stage 2. Columns are scored with `backtest._stats`, so
they're comparable to the TimesFM ones.

### Verdict — don't re-derive this

Tested six ways over 20 years. **No timing skill and no selection skill.**

* **Timing.** MA 10–50 weeks x slope lag 1–8, RS on/off, 10 sector SPDRs: `0/72`
  configs beat the control. The in-sample winner (+0.09) scored −0.22 out of
  sample, negative on 10/10 names — indistinguishable from the median config.
* **Selection.** Equal-weight the Stage 2 names vs equal-weight the same
  universe: negative spread in 2 of 3 universes, best t-stat +0.51. Forward
  returns by stage are non-monotone — **Stage 4 out-returns Stage 2** in both
  the sector and semiconductor universes.
* **The RS leg subtracted value in all three roles** it can play: as a Stage 2
  condition, as an entry gate, as an exit trigger. Every filter added (8W EMA,
  RS, tighter entries) made results worse and roughly doubled the trade count.
* **Beta/vol.** Edge does rise with volatility, but point-in-time vol-ranked
  terciles put the high-vol edge at +1.16%/yr, `t = +0.50`, and all of it lands
  in 2006–2015. It's the GFC, not a volatility effect.

What survives: **drawdown truncation to ~0.35–0.45x buy & hold**, roughly
constant across the volatility spectrum, costing 1–3%/yr and 60–100 trades. At
matched drawdown it roughly doubles the return of simply holding less, so the
truncation is more efficient than de-leveraging. That is the only honest
argument for the model, and max drawdown is a single-path statistic — the one a
trend filter optimizes by construction. Absent a real drawdown constraint, hold
the index.

The control in every table above is **not** buy & hold. It is the same asset
held at constant exposure equal to the rule's own time-in-market: same average
risk, zero timing. A rule that sits in cash a third of the time *should* earn
less than buy & hold, so comparing against buy & hold flatters it. `sweep.py`,
`xsec.py` and the holdout split are reusable for any long/flat signal.

```bash
python -m trading_algo.sweep      # MA x lag grid, basket, holdout
python -m trading_algo.xsec       # cross-sectional: does the stage RANK names?
python -c "from trading_algo.sweep import vol_terciles, vol_report; print(vol_report(vol_terciles()))"
```

## Weekly bull-flag breakout screen

```bash
python -m trading_algo.flag                      # bundled S&P 1500 universe, as of now
python -m trading_algo.flag NVDA AMD --min-vol 1.5
python -m trading_algo.flag --asof 2026-09-25    # what it would have said that Friday
python -m trading_algo.flag --refresh-universe   # re-pull S&P 500/400/600 from Wikipedia
```

A hit at week t, using only bars through t:

* **pole** — swing low to the setup's highest high, **≥ 20% in 1–6 weeks**.
* **flag** — the 2–6 weeks after the top, never longer than the pole. No new
  high, gives back **≤ 50%** of the pole, closes drift sideways-to-down.
* **breakout** — week t **closes above every flag high**. `above_pole_high`
  says whether it also cleared the pole top.
* **volume** — week t **≥ 1.0x the prior 10-week average** (~50-day).

`stop` is the flag low. Liquidity floor: $5 price, $5M average weekly dollar volume.

**Schedule.** `.github/workflows/bull-flag.yml` runs every Friday at
**3:00 PM New York time** (DST handled) and takes ~3 minutes for ~1,500 names.
It opens a GitHub issue labelled `bull-flag` that @mentions the repo owner,
which reaches you by email and GitHub mobile push, and closes the previous week's
issue. A run that fails opens a `FAILED` issue, so silence never means "broken".
For a phone push, set repo secret `NTFY_TOPIC` and subscribe to that topic in
the ntfy app. Run on demand from the Actions tab (`workflow_dispatch`).

Caveats:

* At 3 PM the week is **partial**. The volume test is conservative, since the
  missing last hour only adds volume. The breakout can still fail by the close.
* Yahoo data, unofficial API. If < 80% of tickers download the run aborts
  (FAILED issue) instead of reporting "no breakouts".
* GitHub **disables scheduled workflows in public repos after 60 days without
  a commit**. It emails a warning first; re-enable from the Actions tab.
* GitHub can delay scheduled runs. Cron fires at 2:45 and sleeps to 3:00 to
  absorb that.

### Does it pay? No evidence it does

Back-scanned every week, Jul 2021 – Sep 2026, current S&P 1500 constituents:
723 hits, median **1 hit/week**, none in 37% of weeks. Forward returns from the breakout
week's close, minus the same-week equal-weight universe:

| horizon | hits | excess (mean) | excess (median) | beat universe | t (week-clustered) |
|---|---|---|---|---|---|
| 1w  | 723 | +0.01% | −0.22% | 47% | −0.26 |
| 4w  | 714 | −0.72% | −1.00% | 45% | −1.71 |
| 13w | 677 | −0.75% | −2.55% | 42% | −0.80 |

Current constituents
mean survivorship bias, which flatters hits and baseline alike. Treat the
screen as a **watchlist generator**, not a signal. If you want it to be one,
test it against the same-exposure control first, the way `sweep.py` does.

## Connecting other data sources

TimesFM 3.0 takes covariates natively. Anything you connect has to become a
number per bar on the price series' grid — it forecasts channels, it does not
read news.

```python
from trading_algo.sources import align, calendar, closes, future_index

mkt  = closes("SPY", "^VIX", "TLT", period="5y").dropna()
past = align(mkt.index, vix=mkt["^VIX"], tlt=mkt["TLT"],   # lag 0: same-bar
             cpi=(cpi_series, 14))                         # lag 14: published late
fut  = calendar(future_index(mkt.index, 5))                # known in advance

wf = walk_forward(mkt["SPY"], horizon=5, past=past.ffill().bfill(), future=fut)
```

`align()` is a backward asof join with an explicit **publication lag** per
source. That lag is the whole point: a value may only appear at bar `t` if it
was knowable at `t`. Macro prints, earnings and revisions land days to weeks
after the period they describe — joining them on their reference date invents
edge that isn't there and the backtest will look great.

`past=` is anything known by bar `t`. `future=` must be indexed `horizon` bars
longer, and is *only* for things genuinely known ahead — calendar effects, index
rebalance dates, scheduled announcements. If you can't say why you know a value
in advance, it belongs in `past`.

Checks: `python test_signal.py && python test_backtest.py && python
test_sources.py && python test_flag.py` (model stubbed, no download). The alignment and leakage tests
are the ones worth keeping.

> **License:** TimesFM 3.0 *weights* are `timesfm-non-commercial-license-v1.0` —
> non-commercial, non-production only. For live trading set
> `TIMESFM_CHECKPOINT` to an Apache-2.0 2.x checkpoint or get a commercial
> license from Google. The code itself is Apache-2.0.
