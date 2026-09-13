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

## Sharpe > 2, long-only US equities, 25 years — answered

**It is not reachable, and the ceiling is provable in one number.** A strategy
that is told in advance whether each calendar *month* will be up, and is fully
long or fully flat accordingly, scores **Sharpe 2.04** over 2000–2026. Perfect
monthly foresight *only just* clears 2. Any long/flat rule on a broad US equity
index that beats 2 must therefore time the market better than an oracle with a
month of hindsight.

```bash
python -m trading_algo.timing --ceiling
```

```
PERFECT FORESIGHT, long/flat, at each decision horizon:
  knows every daily      sign in advance -> Sharpe  8.85
  knows every weekly     sign in advance -> Sharpe  3.85
  knows every monthly    sign in advance -> Sharpe  2.04
  knows every quarterly  sign in advance -> Sharpe  1.38

DAILY DIRECTIONAL SKILL required (base rate = always long = 54.2% right):
  Sharpe 0.71 needs 52.8% of days called right   <- this repo's strategy
  Sharpe 1.0  needs 54.7%
  Sharpe 2.0  needs 61.4%

INFORMATION COEFFICIENT  corr(weight, next-day excess return) = +0.040
  fundamental law, Sharpe ~ IC * sqrt(252):  0.63 (realised 0.71)
  Sharpe 2.0 would need IC = 0.126, 3.2x this signal, held for 25 years.
  Or 8 INDEPENDENT sleeves this good -- long-only, one index, they do not exist.
```

Three independent framings, same answer: the monthly oracle, the required daily
hit rate (+7pp of genuine skill over the base rate, sustained a quarter century),
and the fundamental law. Sharpe is also *estimated*, not observed — 26 years of
daily bars gives a standard error of **±0.19**, so "2" sits nearly seven
standard errors above what the best rule here actually delivers.

## The strategy that does survive

```bash
python -m trading_algo.engine --fetch    # once: 30y of daily bars into data/ (gitignored)
python -m trading_algo.timing            # full report, all robustness tables
python -m trading_algo.timing --demo     # self-check, no download
```

Long/flat SPY, weight in [0, 1], decided at each close and held one day. Two
sleeves averaged 50/50 (`trading_algo/timing.py`):

* **TREND** — close above its own MA, averaged over 100/150/200/250 days so the
  weight steps down rather than betting on one window. On ~72% of the time,
  negative skew, 8 turns/yr.
* **MEANREV** — close at a 3/5/10-day low, held one day. On ~22% of the time,
  **positive** skew (+3.1), and where most of the return per unit of exposure
  lives. It is also the expensive half: ~65 turns/yr.

Daily return correlation between the sleeves is 0.32, which is the whole reason
the blend beats either.

| 2bp/unit traded, excess of T-bills | years | ann excess | vol | **Sharpe** | max DD | exposure |
|---|---|---|---|---|---|---|
| **TIMING** | 26.6 | +5.99% | 8.8% | **0.71** ± 0.19 | −16.3% | 48% |
| buy & hold | 26.6 | +6.45% | 19.2% | 0.42 | −59.9% | 100% |
| **TIMING**, strict last 25y | 24.9 | +6.79% | 8.6% | **0.81** | −13.9% | 49% |
| buy & hold, strict last 25y | 24.9 | +8.47% | 18.9% | 0.52 | −56.1% | 100% |

Sub-periods 0.37 / 0.95 / 0.82 (2000-08 / 09-17 / 18-26); rolling 3-year Sharpe
is negative 1% of the time. **It does not beat buy & hold on return** — it gives
up ~0.5%/yr of excess return to cut volatility by half and drawdown by a third.
Every number above is net of costs and net of T-bills.

Blend weight is flat from 0.25 to 0.50 (0.71–0.72), so 50/50 is a plateau, not a
peak. The binding fragility is **cost**: Sharpe runs 0.78 / 0.74 / 0.71 / 0.60 /
0.41 at 0/1/2/5/10 bp. Above ~8bp all-in the mean-reversion sleeve stops paying.
SPY at retail size is ~1bp; anything less liquid kills it.

### Negative results — don't re-derive these

Scored identically, same bars, same cost model, excess of T-bills.

* **Covered calls make it worse.** BXM (buy-write on SPX) scores Sharpe 0.32 vs
  0.41 for SPX itself over 2000–2026; the call leg alone is −3.78%/yr at 9.6%
  vol. Overlaid on the strategy's persistent (21-day-minimum) weight it takes
  0.70 → 0.65. It buys skew (+0.24 → +0.57) and costs return. The one permitted
  form of leverage is a Sharpe *reducer* on this window.
* **The overnight anomaly is entirely a transaction cost illusion.** Night
  (close→open) excess return is +5.1%/yr against −0.7%/yr intraday — but
  night-only trades 504x/yr. At 0.5bp/side Sharpe is 0.33, at 1bp it is 0.10, at
  2bp it is −0.35. `engine.sessions()` splits the day and charges every boundary
  crossing, which is what the earlier version of this analysis failed to do.
* **Diversifying across US equity ETFs dilutes, it does not diversify.** The
  same rule equal-weighted over SPY/QQQ/IWM + 9 sector SPDRs scores 0.61 vs 0.76
  for SPY alone on the same bars (those start 2000-05-30 when IWM lists, which is
  why SPY alone reads 0.76 there and 0.71 from January); SPY/QQQ/IWM alone 0.72.
  Index-level mean reversion is stronger than sector-level, where idiosyncratic
  momentum eats it.
* **Cross-sectional mean reversion across sectors:** best 0.46, worse than the
  0.48 equal-weight buy & hold it is built from.
* **Sector momentum** (12-1, top 2/3/4, monthly): 0.49 / 0.55 / 0.49 — no better
  than a plain MA200, and a trend overlay on top made every variant worse.
* **Volatility targeting hurts.** Scaling the blend by min(1, target/realised
  vol) takes 0.71 → 0.64–0.67 for targets 8–12%. Without leverage it can only
  cut exposure, and it cuts return faster than volatility.
* **Every orthogonal macro/seasonal sleeve diluted it.** Variance risk premium
  (VIX − realised vol), 10y−3m term spread, and Nov–Apr seasonality each score
  0.38–0.46 standalone, and each *lowered* the blend when added (0.71 → 0.58–0.64).
* **Conditioning mean-reversion made it worse**, all of it: sizing by dip depth
  (0.69), gating on high VIX (0.62), gating on low VIX (0.58), requiring the
  trend filter to agree (0.48).
* **Turn-of-month** (0.31), day-of-week (best 0.12), n-sigma dip thresholds
  (0.16–0.39): real but too weak to survive blending.

### Scoring rules, fixed once

`trading_algo/engine.py`. Everything above is scored the same way, which is the
only reason the numbers are comparable:

* **Excess of T-bills**, always — `exret = r_asset − rf` from `^IRX`. Idle cash
  earns the bill rate and therefore contributes exactly zero. Without this, a
  strategy that sits in cash buys Sharpe for free off the 2000s rate curve.
* `w[t]` is decided from data up to and *including* the close of `t`, and earns
  `t+1`'s return. `run()` rejects any short or any row summing above 1.
* Costs in bp on `|dw|`, charged at the bar the trade happens.
* `timing.demo()` asserts the signal cannot read the future by recomputing it on
  a truncated series and requiring every already-decided weight to be identical.

> **Caveat that no backtest removes:** the mean-reversion sleeve's parameters
> were chosen knowing the full sample. Its stability across three disjoint
> ~9-year sub-periods (0.62 / 0.66 / 0.65 on the sleeve alone) is the argument
> that it is not fitted, and it is an argument, not a proof.

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

Checks: `python test_timing.py && python test_signal.py && python test_backtest.py
&& python test_sources.py` (model stubbed, no download). The alignment and leakage tests
are the ones worth keeping.

> **License:** TimesFM 3.0 *weights* are `timesfm-non-commercial-license-v1.0` —
> non-commercial, non-production only. For live trading set
> `TIMESFM_CHECKPOINT` to an Apache-2.0 2.x checkpoint or get a commercial
> license from Google. The code itself is Apache-2.0.
