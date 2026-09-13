# trading-algo

Forecasting engine: [TimesFM 3.0](https://github.com/google-research/timesfm/releases/tag/v3.0.0).

> **Portfolio mandate:** [`INVESTMENT_POLICY.md`](INVESTMENT_POLICY.md) governs
> all work in this repository. The strategy is drawdown-first and must be
> evaluated with the owner's 5.4% annual withdrawal rate, paid in equal monthly
> installments. This repository is research-only until an explicitly approved
> live-execution configuration passes its validation gates.

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

## The governing objective is drawdown, not Sharpe

`INVESTMENT_POLICY.md` ranks strategies by **withdrawal-adjusted downside** under
a 5.4%/yr equal-monthly withdrawal. Sharpe is blind to the thing that actually
destroys an income portfolio — sequence-of-returns risk — so it cannot answer
that question. Selling units into a 56% drawdown is permanent damage no later
rally reverses.

```bash
python -m trading_algo.mandate           # the policy report
python -m trading_algo.mandate --demo    # self-check, no download
python -m trading_algo.stress            # 153 start dates, not one
python -m trading_algo.intake            # the queued research hypothesis
```

$1,000,000 initial, 5.4%/yr taken in equal monthly dollars, 2000–2026:

| strategy | ret | vol | maxDD no wd | maxDD acct | maxDD delivered | recover mths | final x | paid x | **total x** | forced sales |
|---|---|---|---|---|---|---|---|---|---|---|
| **TIMING** | +8.2% | 8.9% | −12.8% | **−19.4%** | **−8.7%** | **23** | 1.92 | 1.77 | **3.69** | **3** |
| MA200 only | +7.2% | 11.4% | −22.8% | −42.6% | −21.4% | 59 | 1.48 | 1.29 | 2.77 | 201 |
| static 50/50 | +5.6% | 9.6% | −30.6% | −47.9% | −21.5% | 62 | 1.00 | 1.14 | 2.14 | 0 |
| buy & hold SPY | +8.5% | 19.2% | −55.2% | −69.1% | −43.2% | 85 | 2.04 | 1.32 | 3.36 | 265 |

**Buy & hold depletes the account entirely** — to zero on 2025-03-03 — under the
fixed-dollar reading of the withdrawal. Under every reading it spends 85 months
recovering and forces a sale into a drawdown 265 times. The timing rule survives
all three readings, and it also **delivers the most total value** (3.69x against
buy & hold's 3.36x), which the Sharpe framing had scored as "gives up 0.5%/yr".
Under the mandate that conclusion inverts.

Two findings worth keeping:

* **The timing rule is self-hedging for liquidity.** Three forced sales in 27
  years against buy & hold's 265, because it holds cash exactly when the market
  is down — which is when a withdrawal would otherwise sell units cheap. A
  separate cash buffer changes nothing (identical at 0/3/6/12 months): the
  strategy's own cash sleeve is either ~0 or far larger than a year of income.
* **Account-value drawdown cannot measure duration for an income portfolio.** It
  saturates near 95% time-under-water by construction, since an account paying
  out 5.4%/yr rarely makes new highs. The honest measure is drawdown of *account
  + income already paid* — total wealth delivered — which is the `maxDD
  delivered` and `recover mths` columns.

The blend moved **0.50 → 0.40** on this evidence: it minimises delivered-wealth
drawdown (−8.7% vs −9.6%), recovers in 23 months instead of 33, and hands over
3.69x instead of 3.51x, at identical Sharpe (0.72). The curve is flat from 0.35
to 0.50 on every one of those measures, so it is a plateau, not a pick. Note
that "longest underwater" alone is a knife-edge statistic — it swings 51 → 109
months on small blend changes — so steer by % time under water and the median
spell, which move smoothly.

> **Two policy items need your decision, Jason** — see the end of this section.
> (1) `INVESTMENT_POLICY.md` §5 forbids options absent a written amendment, but
> the brief authorised covered calls and later long puts. The research is done
> and both are Sharpe-negative, so nothing depends on it, but the document and
> the instruction disagree. (2) The withdrawal rule admits three readings; the
> default here is `annual_reset`.

### Sequence stress: 153 start dates, not one

One 26-year path is one sample, and January 2000 happens to be near the worst
entry available — which flatters nothing but proves nothing either. `^SP500TR`
is the S&P 500 total-return index, real and assumption-free back to 1988 (it
tracks SPY to 0.985 daily correlation; the 0.13%/yr gap is SPY's expense ratio),
so 25-year windows can start anywhere in 1988–2001.

```bash
python -m trading_algo.stress
```

| 25-year windows, 5.4%/yr withdrawn | worst final x | median final x | **worst maxDD** | recover | depleted |
|---|---|---|---|---|---|
| **TIMING** | 1.38 | 1.71 | **−25.3%** | 38 mo | **0 of 153** |
| buy & hold | 1.40 | **2.48** | −68.9% | 85 mo | 0 of 153 |

**Say the unflattering part first: buy & hold delivers more in the median** —
4.72x total against 3.55x, counting income taken. The timing rule is insurance,
and the premium is roughly 1.2x of median lifetime wealth. What it buys shows up
at the horizons where a withdrawal plan actually fails:

| worst outcome over any start | TIMING final x | buy & hold final x |
|---|---|---|
| 5-year windows (394 starts) | 0.82 | 0.51 |
| **10-year windows (334 starts)** | **0.88** | **0.37** |
| 15-year windows (273 starts) | 1.18 | 0.75 |
| 25-year windows (153 starts) | 1.38 | 1.40 |

Ten years of withdrawals from the worst entry leaves buy & hold at **37 cents on
the dollar** and the timing rule at 88. Over a full 25 years the two converge on
terminal wealth — but a plan that spends a decade down 63% is not one anyone
keeps following, and the policy's whole point is the path, not the endpoint.

Entering at the exact top:

| entry | TIMING final / maxDD | buy & hold final / maxDD |
|---|---|---|
| 2000-03 dot-com top | 1.59 / −22.8% | 1.40 / −69.0% |
| 2007-10 GFC top (18y) | 1.76 / −20.8% | 2.30 / −59.1% |

**Sustainable withdrawal rate** (ends at ≥ the starting $1m after 25 years):
TIMING supports 7.2–8.1%, buy & hold 6.7–9.6%. At the *worst* entry (March 2000)
TIMING supports **7.2% against buy & hold's 6.7%** — it wins exactly where it
matters. Either way the policy's 5.4% has headroom.

> **Read the sample size honestly.** 153 starts drawn from 38 years of one
> market share nearly all their bars: that is closer to 1.5 independent 25-year
> samples than to 153, and every window contains both 2000–02 and 2008, which is
> why worst and median drawdown are identical. The shorter horizons carry more
> genuine variation. No amount of resampling one market's history fixes this.

### Research intake: AT-2026-09-13-photonics — rejected

`research_sources/annualize_this/` defines an intake contract for an external
research source: hypotheses only, levels defined without chart reading, frozen
point-in-time universe, equal-weight *and* equal-time-in-market controls, costs,
holdouts, no tuning after results. `python -m trading_algo.intake` implements
exactly that, and nothing else.

**The universe cannot be tested, only the rule.** The 22 symbols were written on
2026-09-13 by someone who knows which photonics companies still exist. Ten of
the nineteen with data traded in 2000; the optical-networking bust that
destroyed the sector in 2000–02 — JDSU, Corvis, New Focus, Avanex and dozens
more — appears in the list through its survivors alone. Its 163x equal-weight
headline is that selection, not a sector return, and unlike `stocks.py` there is
no point-in-time photonics index to correct against. What *is* honest is the
rule against the same basket, where the survivorship gift cancels on both sides.

| rule minus equal-time-in-market control | Sharpe spread | withdrawal-adj DD spread |
|---|---|---|
| 10 / 13 / 20 / 26 / 52-week lookback | −0.08 / −0.10 / −0.13 / −0.07 / −0.17 | +10.6% / +11.8% / +8.4% / +15.9% / +14.9% |
| 2000–2013 | **+0.05** | **+14.6%** |
| 2014–2026 | **−0.34** | **−19.3%** |

Rejected on three counts. **No timing skill** — beaten by its own equal-time
control at 0 of 5 lookbacks, and at 1.6x/yr turnover cost isn't the cause (the
spread is −0.12 to −0.15 across 5–40bp). **The drawdown edge doesn't replicate**
— it looked like +8 to +16 points on the policy's own criterion, then reversed
sign out of sample. **Unsuitable regardless** — −51% withdrawal-adjusted
drawdown at 22% volatility, against −19% at 9% for the existing SPY rule.

The verdict is recorded in the intake register itself, which is where the
contract says it belongs.

## Sharpe > 2, long-only US equities, 25 years — answered

**It is not reachable, and the ceiling is provable in one number.** A strategy
that is told in advance whether each calendar *month* will be up, and is fully
long or fully flat accordingly, scores **Sharpe 2.04** over 2000–2026. Perfect
monthly foresight *only just* clears 2. Any long/flat rule on a broad US equity
index that beats 2 must therefore time the market better than an oracle with a
month of hindsight.

```bash
python -m trading_algo.timing --ceiling   # why timing cannot get there
python -m trading_algo.stocks             # why selection cannot either
python -m trading_algo.overfit            # what searching for it actually produces
python -m trading_algo.overfit --blend   # the best any combination can do
python -m trading_algo.options           # what buying puts does (nothing, for Sharpe)
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

That bounds one axis — choosing *when* to hold the market. Choosing *what* to
hold needs its own bound, because a 400-name portfolio has far more breadth than
one long/flat index bet. `python -m trading_algo.stocks` bounds it two ways:

**Perfect hindsight static selection.** Rank all 503 of today's S&P 500 members
by their realised 2001–2026 return, buy the best k in 2001, hold 25 years. This
is the answer sheet; no selection rule can beat it.

| bought with 25 years of hindsight | ann excess | vol | Sharpe |
|---|---|---|---|
| best 1 name (MNST, +34%/yr) | +33.7% | 42.8% | 0.89 |
| best 5 | +38.9% | 29.8% | 1.25 |
| **best 10** | +38.0% | 25.6% | **1.39** |
| best 50 | +25.8% | 22.6% | 1.13 |
| SPY | +8.2% | 19.0% | 0.51 |

**Knowing the winners in advance tops out at Sharpe 1.39.** Concentration adds
return and volatility together, so it moves the numerator and denominator at
once and the ratio peaks around k=10. Long-only cannot short the losers, so
market beta stays in the portfolio and the volatility floor stays around 20%;
Sharpe 2 would need ~40%/yr excess at that vol, sustained 25 years, which even
the answer sheet does not deliver.

**And the universe is rigged in our favour.** It is today's members only, so
every name that collapsed and left the index is missing. RSP is the identical
equal-weight S&P 500 portfolio built point-in-time:

| 2003–2026, identical construction | ann excess | Sharpe |
|---|---|---|
| RSP (point-in-time) | +9.40% | 0.55 |
| equal-weight *today's* members | +16.20% | 0.86 |
| **survivorship gift** | **+6.80%/yr** | **+0.30** |

Real rules on that inflated universe (5bp/unit, monthly, top 50): momentum 12-1
**0.92** (top 20: 1.03), 5-day reversal 0.85, low-vol 0.71, 1-month reversal
0.69, equal-weight 0.82. Bolting the index-timing rule on top as an exposure
scalar gives the best combination found anywhere in this repo — momentum ×
TIMING at **Sharpe 0.97**, +11.5%/yr excess, 12.0% vol, −19.2% max drawdown —
and that 0.97 still contains the +0.30 survivorship gift, so call it **~0.67**
honestly. Two is 5.1 standard errors above even the unadjusted figure.

**Breadth does not rescue it either.** The fundamental law says Sharpe scales
with the square root of the number of independent bets, so the strongest
remaining candidate is the mean-reversion rule applied *per stock across all
~500 names at once* — on any given day 70–210 of them sit at an n-day low, which
is orders of magnitude more breadth than a monthly top-50 ranking. Run with
**transaction costs switched off entirely**, on the survivorship-biased
universe, it tops out at **Sharpe 1.39**. It turns over 340x/yr, so every basis
point of real cost removes 0.15 of Sharpe: 1.24 at 1bp, **0.66 at the ~5bp
single stocks actually cost**. An impossible frictionless version of the
highest-breadth strategy available still does not reach 2.

**And no combination of any of it reaches 2 either.** `python -m
trading_algo.overfit --blend` builds all 12 sleeves built anywhere in this repo
and solves for the convex combination (non-negative weights summing to 1 — no
leverage) that maximises Sharpe *using hindsight on every mean and covariance in
the sample*. No blending scheme can beat it, including one that would work out
of sample. It reaches **Sharpe 1.01** (+11.8%/yr, 11.7% vol), and it gets there
by putting 77% of the weight on single-stock sleeves that carry the +0.30
survivorship gift. Mean pairwise sleeve correlation is **+0.68** — they are all
long equity beta, which is precisely why combining them buys so little.

So every axis is bounded, and none reaches 2:

| | perfect-foresight ceiling | best real rule here |
|---|---|---|
| **timing** (when to hold) | 2.04 (monthly oracle) | 0.71 / 0.81 over the strict 25y |
| **selection** (what to hold) | 1.39 (25y hindsight, best 10) | 0.97 biased, ~0.67 adjusted |
| **breadth** (how many bets) | 1.39 (zero cost, 500 names, biased) | 0.42 at 5bp single-stock cost |
| **combination** (blending it all) | 1.01 (ex-post optimal, biased) | 0.97 best single sleeve |

### "Just keep searching until something hits 2"

`python -m trading_algo.overfit` — 164,183 randomly drawn 3-condition long/flat
rules on SPY, built from 79 individually sensible conditions (trend, breakouts,
mean reversion, volatility regime, calendar), fitted on 2000–2013 and then
checked on 2014–2026.

```
IN-SAMPLE Sharpe across the search: median -0.01, 95th pct +0.48, MAX +0.97
  rules clearing Sharpe 2 in-sample: 0 of 164,183
```

**Not one.** Brute force cannot even *manufacture* Sharpe 2 here, given 14 years
of hindsight and no obligation to work afterwards. The target is not hiding in
the space; the space does not contain it.

What the search does show is the cost of looking. The top 25 rules average
**+0.91 in-sample and +0.30 over the next 12.7 years** — a 0.61 shrinkage that is
selection, not skill — and in-sample rank correlates only **+0.145** with what a
rule does next. For calibration, the best of N independent zero-edge rules lands
near +0.81 at N=100 and +1.41 at N=1,000,000 on luck alone. A search big enough
to surface a 2 is a search big enough to fabricate one.

The repo's own strategy, scored identically: **+0.59 in-sample (98th percentile
of the search), +0.85 out of sample.** It *gained* 0.26 out of sample where the
search winners lost 0.61 — which is the argument that it is a rule rather than a
draw. It is also still nowhere near 2.

> A bug worth recording: the first version of `overfit.py` bypassed
> `engine.run()` for speed and multiplied each position by *its own* bar's
> return instead of the next one. It reported a winner of `ret1_up AND ret1_up
> AND ret1_up` at Sharpe 8.5 with a +0.98 IS/OOS correlation. `demo()` now
> asserts that "today closed up" scores below 1.0, because an off-by-one is the
> cheapest way in the world to produce a Sharpe 2.



## The strategy that does survive

```bash
python -m trading_algo.engine --fetch    # once: 30y of daily bars into data/ (gitignored)
python -m trading_algo.stocks --fetch    # once: ~500 S&P names, for the selection bound
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

### Buying puts does not change the answer either

`python -m trading_algo.options` — and the useful half of it needs **no
option-pricing model at all**. BXM is a real published buy-write (long SPX,
short a 1-month ATM call, rolled on expiry Friday), and put-call parity then
pins the matching ATM *put* exactly from realised index and BXM returns:

```
BXM payoff over the month = min(S1, K)
protective put payoff     = max(S1, K)      and the two sum to S1 + K
```

No Black-Scholes, no implied vol, no skew guess. The derivation recovers a
1-month ATM call at **1.93% of spot** and put at **1.78%** — which is what SPX
options actually cost, and `demo()` asserts the parity identity holds to 1e-9.

| monthly rolls, excess of T-bills | ann excess | vol | Sharpe | max DD | skew |
|---|---|---|---|---|---|
| SPY | +6.19% | 17.6% | **0.43** | −52.6% | −1.24 |
| SPY + ATM put | +3.21% | 8.2% | **0.43** | −32.3% | **+1.09** |
| TIMING (cash when out) | +5.91% | 6.9% | **0.87** | −13.0% | −0.76 |
| TIMING + put on held weight | +3.94% | 6.3% | 0.65 | −11.2% | +3.71 |
| 100% long, put when signal low | +4.76% | 12.3% | 0.44 | −36.9% | −0.92 |

**Buying ATM puts is Sharpe-neutral.** Half the volatility, half the return,
skew −1.24 → +1.09, drawdown −53% → −32%. That is a real risk transformation and
worth buying if drawdown is the constraint — but it is not Sharpe, which is what
efficiently priced insurance looks like. Bolted onto the timing rule it
*subtracts*: 0.87 → 0.65, because sitting in cash already removes the downside
and costs ~0.5%/yr against the put's ~3%/yr. Using puts as the risk-off vehicle
instead of cash is much worse (0.44 vs 0.87), and buying them only when cheap
does not rescue it.

Collars (long OTM put + short OTM call — both legs now permitted) need strikes
parity cannot pin, so they need a model, and that is exactly where option
backtests go to die:

| 95/105 collar, same months | ann excess | vol | Sharpe |
|---|---|---|---|
| flat volatility surface | +9.53% | 11.2% | **0.87** |
| +2 vol points of skew | +7.32% | 11.3% | 0.69 |
| +4 vol points (the real SPX shape) | +5.01% | 11.3% | **0.49** |

A **0.45 Sharpe swing on one unobservable assumption** is not a backtest, it is
the assumption being read back. Pricing OTM options honestly needs real option
data, and even the flattering end does not approach 2.

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
* **Stock selection does not rescue it** (`trading_algo/stocks.py`): every
  long-only cross-sectional rule tested lands between 0.69 and 1.03 *on a
  survivorship-biased universe worth +0.30 of Sharpe*, and the perfect-hindsight
  static portfolio itself only reaches 1.39.

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
