"""Long/flat SPY timing: an average of trend filters and an average of
short-horizon mean-reversion triggers, blended half and half.

Two sleeves, because they earn from opposite things and are on at different
times (daily return correlation 0.32):

* TREND  -- close above its own moving average, averaged over 100/150/200/250
  days so the weight steps down gradually instead of betting on one window.
  Negative skew, on ~72% of the time, few trades.
* MEANREV -- close at a 3/5/10-day low, held one day. Positive skew, on ~22% of
  the time, and it is where most of the return per unit of exposure comes from.
  It is also the cost-sensitive half: ~65 turns a year.

Everything else tried made it worse and is recorded in the README. The blend is
0.40 because that is where withdrawal-adjusted drawdown bottoms out under
`INVESTMENT_POLICY.md`; Sharpe is flat from 0.35 to 0.50 either way.

    python -m trading_algo.timing          # full report
    python -m trading_algo.timing --demo   # self-check, no download
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS, excess, load, report, rf, score, run, sharpe_se, stats

TREND = (100, 150, 200, 250)
MEANREV = (3, 5, 10)
# 0.40, not 0.50: INVESTMENT_POLICY.md ranks on withdrawal-adjusted drawdown, and
# 0.40 minimises drawdown of delivered wealth (-8.7% vs -9.6%), recovers in 23
# months instead of 33, and hands over 3.69x instead of 3.51x -- at the same
# Sharpe. The curve is flat from 0.35 to 0.50 on every one of those measures, so
# this is a plateau, not a pick. See `python -m trading_algo.mandate`.
BLEND = 0.40


def weight(px: pd.Series, blend: float = BLEND, trend=TREND, meanrev=MEANREV) -> pd.Series:
    """Target SPY weight in [0, 1], decided at the close of each bar.

    Uses only closes up to and including that bar. Feed it the LONGEST price
    history you have, then slice the result -- warming the 250-day average up
    inside the evaluation window silently scores a year of forced flatness.
    """
    s = pd.Series(px).dropna()
    t = pd.concat([s > s.rolling(n).mean() for n in trend], axis=1).mean(axis=1)
    m = pd.concat([s <= s.rolling(n).min() for n in meanrev], axis=1).mean(axis=1)
    w = blend * t + (1 - blend) * m
    return w.where(s.rolling(max(trend)).mean().notna())  # NaN until warm, never 0


def backtest(start="2000-01-01", end=None, cost_bps: float = 2.0, blend: float = BLEND,
             ticker: str = "SPY", **kw) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Net excess returns, the weights that produced them, and excess returns."""
    w_full = weight(load()[ticker], blend, **kw)
    x = excess([ticker], start=start, end=end)
    w = pd.DataFrame({ticker: w_full.reindex(x.index)})
    if w[ticker].isna().any():
        raise ValueError(f"{max(kw.get('trend', TREND))}-day trend is not warm at {start}")
    return run(w, x, cost_bps), w, x


def _row(net):
    return {**stats(net), "se": sharpe_se(net)}


def main(start="2000-01-01", cost_bps: float = 2.0):
    net, w, x = backtest(start, cost_bps=cost_bps)
    idx = net.index
    bh = run(pd.DataFrame(1.0, index=x.index, columns=["SPY"]), x, cost_bps)
    print(f"=== SPY long/flat timing, {idx[0].date()} .. {idx[-1].date()}, "
          f"{cost_bps:g}bp per unit traded, excess of T-bills ===\n")
    print(report({"TIMING": {**score(w, x, cost_bps)},
                  "buy & hold": {**_row(bh), "exposure": 1.0, "ann_turnover": 0.0}}))

    print("\n-- blend sensitivity (0 = pure mean-reversion, 1 = pure trend); see")
    print("   `python -m trading_algo.mandate` for the drawdown-first view that picks 0.40")
    print(report({f"blend {b:.2f}": score(
        pd.DataFrame({"SPY": weight(load()["SPY"], b).reindex(x.index)}), x, cost_bps)
        for b in (0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 1.0)}))

    print("\n-- cost sensitivity (bp charged on each unit of |dw|)")
    print(report({f"{c:g} bp": _row(backtest(start, cost_bps=c)[0]) for c in (0, 1, 2, 3, 5, 10)}))

    print("\n-- sub-periods and the strict 25-year window")
    cuts = [("2000-2008", "2000-01-01", "2008-12-31"), ("2009-2017", "2009-01-01", "2017-12-31"),
            ("2018-2026", "2018-01-01", "2026-12-31"),
            (f"last 25y", str(idx[-1].normalize() - pd.DateOffset(years=25))[:10], None)]
    out = {}
    for nm, a, b in cuts:
        n2, w2, x2 = backtest(a, b, cost_bps)
        out[nm] = score(w2, x2, cost_bps)
        out[f"  buy & hold"if nm=="last 25y" else f"  b&h {nm}"] = {
            **_row(run(pd.DataFrame(1.0, index=x2.index, columns=["SPY"]), x2, cost_bps)),
            "exposure": 1.0, "ann_turnover": 0.0}
    print(report(out))

    print("\n-- calendar years (excess of T-bills, %)")
    yr = pd.DataFrame({"timing": net, "b&h": bh}).groupby(net.index.year).apply(
        lambda d: (1 + d).prod() - 1) * 100
    yr["diff"] = yr["timing"] - yr["b&h"]
    print(yr.round(1).to_string())

    roll = net.rolling(756).apply(lambda r: r.mean() / r.std() * np.sqrt(TRADING_DAYS)).dropna()
    print(f"\nrolling 3y Sharpe: min {roll.min():+.2f}  median {roll.median():+.2f}  "
          f"max {roll.max():+.2f}  share<0 {(roll < 0).mean():.0%}")
    print(f"Sharpe {stats(net)['sharpe']:.2f} +/- {sharpe_se(net):.2f} (1 s.e.); "
          f"P(true Sharpe > 2) is not distinguishable from zero at this sample size.")


def ceiling(start="2000-01-01", paths: int = 120, seed: int = 0):
    """How much timing skill Sharpe 2 would actually require. The point of this
    repo's answer to 'get me Sharpe 2', so it is a command, not a claim.

    Long/flat on SPY, weight 1 or 0. Replaces the signal with an oracle that is
    right a fixed fraction of the time and measures what that buys.
    """
    x = excess(["SPY"], start=start)["SPY"]
    r, up, n = x.to_numpy(), (x > 0).to_numpy(), len(x)
    rng = np.random.default_rng(seed)
    sh = lambda v: v.mean() / v.std(ddof=1) * np.sqrt(TRADING_DAYS)

    print(f"SPY excess of T-bills, {x.index[0].date()}..{x.index[-1].date()}: "
          f"{n} bars, {up.mean():.1%} up days, buy & hold Sharpe {sh(r):.2f}\n")

    print("PERFECT FORESIGHT, long/flat, at each decision horizon:")
    per = pd.Series(x.index)
    for label, key in (("daily", None), ("weekly", "W"), ("monthly", "M"), ("quarterly", "Q")):
        if key is None:
            pos = up.astype(float)
        else:
            g = per.dt.to_period(key).values
            pos = pd.Series(x).groupby(g).transform(lambda s: float(s.sum() > 0)).to_numpy()
        print(f"  knows every {label:<10} sign in advance -> Sharpe {sh(r * pos):5.2f}")
    print("  ^ a strategy that KNOWS each month's sign clears 2 only just. Any\n"
          "    long/flat rule beating 2 must time the market better than that.\n")

    def skill(acc):
        out = [sh(r * np.where(rng.random(n) < acc, up, ~up).astype(float)) for _ in range(paths)]
        return float(np.mean(out))

    accs = np.linspace(0.50, 0.68, 19)
    curve = [skill(a) for a in accs]
    print("DAILY DIRECTIONAL SKILL required (base rate = always long = "
          f"{up.mean():.1%} right):")
    for t in (0.71, 1.0, 1.5, 2.0, 3.0):
        print(f"  Sharpe {t:<4} needs {np.interp(t, curve, accs):.1%} of days called right"
              + ("   <- this repo's strategy" if t == 0.71 else ""))

    net, w, xx = backtest(start)
    ic = np.corrcoef(w["SPY"].shift(1).fillna(0).reindex(net.index), xx["SPY"].reindex(net.index))[0, 1]
    print(f"\nINFORMATION COEFFICIENT  corr(weight, next-day excess return) = {ic:+.3f}")
    print(f"  fundamental law, Sharpe ~ IC * sqrt(252):  {ic * np.sqrt(TRADING_DAYS):.2f} "
          f"(realised {stats(net)['sharpe']:.2f})")
    print(f"  Sharpe 2.0 would need IC = {2 / np.sqrt(TRADING_DAYS):.3f}, "
          f"{2 / np.sqrt(TRADING_DAYS) / ic:.1f}x what this signal has, held for 25 years.")
    print(f"  Or {(2 / stats(net)['sharpe']) ** 2:.0f} INDEPENDENT sleeves this good, since "
          "uncorrelated Sharpes add in quadrature.\n  Long-only, one index, no shorting: "
          "they do not exist.")


def demo():
    """Self-check: no look-ahead, correct sleeve behaviour, on synthetic prices."""
    idx = pd.bdate_range("2000-01-01", periods=800)
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 800))), index=idx)

    w = weight(px)
    assert w.iloc[:249].isna().all() and w.iloc[250:].notna().all(), "warmup leaks or over-runs"
    assert w.dropna().between(0, 1).all()

    # truncating the future cannot change any weight already decided -- the only
    # check that actually catches look-ahead in a rolling signal
    cut = 600
    assert np.allclose(weight(px.iloc[:cut]).dropna(),
                       w.iloc[:cut].dropna()), "signal reads the future"

    # a pure step-up series is always in trend and never at an n-day low
    up = pd.Series(np.arange(1, 801, dtype=float), index=idx)
    assert (weight(up).dropna() == BLEND).all(), "trend sleeve mis-scored on a monotone rise"
    assert (weight(up, blend=0.0).dropna() == 0.0).all(), "mean-rev fired on a monotone rise"
    dn = pd.Series(np.arange(800, 0, -1, dtype=float), index=idx)
    assert (weight(dn, blend=0.0).dropna() == 1.0).all(), "mean-rev missed a monotone fall"
    assert (weight(dn, blend=1.0).dropna() == 0.0).all(), "trend sleeve long in a downtrend"
    print("timing ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    elif "--ceiling" in sys.argv:
        ceiling()
    else:
        main()
