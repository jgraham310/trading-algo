"""Sequence-of-returns stress for the withdrawal plan (INVESTMENT_POLICY.md).

One 26-year path is one sample. The policy's core worry -- that withdrawals hit
during the wrong decade and never recover -- is a statement about the
DISTRIBUTION of start dates, and a single backtest cannot measure it. Starting
in January 2000 happens to be close to the worst entry available, which flatters
nothing but does not prove robustness either.

Data: `^SP500TR`, the S&P 500 total-return index, real and assumption-free back
to 1988 -- five years more than SPY, and it tracks SPY to 0.985 daily
correlation (the 0.13%/yr gap is SPY's expense ratio). With ^IRX from 1985 that
gives 38 years, so 25-year windows can start anywhere in 1988-2001 and shorter
windows give many more samples.

    python -m trading_algo.stress
    python -m trading_algo.stress --demo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS, load, rf, run, stats
from .mandate import simulate, underwater
from .timing import BLEND, weight as tw

TICKER = "^SP500TR"


def paths(ticker: str = TICKER, cost_bps: float = 2.0):
    """Daily TOTAL returns for the strategy and for buy & hold, on one series."""
    px = load()[ticker].dropna()
    r = px.pct_change()
    f = rf(r.index)
    x = pd.DataFrame({ticker: (r - f)}).dropna()
    w = pd.DataFrame({ticker: tw(px, BLEND).reindex(x.index)})
    warm = w[ticker].notna()
    x, w = x[warm], w[warm]
    fx = rf(x.index)
    strat = (run(w, x, cost_bps) + fx).dropna()
    bh = (run(pd.DataFrame(1.0, index=x.index, columns=[ticker]), x, cost_bps) + fx).dropna()
    return strat, bh, w[ticker]


def window(total: pd.Series, start, years: float, rate: float = 0.054,
           weight: pd.Series | None = None, mode="annual_reset") -> dict | None:
    """One withdrawal path starting at `start`, or None if there is not enough data."""
    s = total[total.index >= pd.Timestamp(start)]
    need = int(years * TRADING_DAYS)
    if len(s) < need:
        return None
    s = s.iloc[:need]
    sim = simulate(s, rate=rate, mode=mode,
                   weight=None if weight is None else weight.reindex(s.index))
    months, _, _ = underwater(sim["dd_delivered"])
    fin = float(sim["value"].iloc[-1] / 1_000_000)
    return {"start": pd.Timestamp(start), "final_x": fin,
            "paid_x": float(sim["withdrawal"].sum() / 1_000_000),
            "max_dd": float(sim["drawdown"].min()),
            "dd_delivered": float(sim["dd_delivered"].min()),
            "recover_months": months, "depleted": fin <= 0}


def rolling(total: pd.Series, years: float, rate: float = 0.054,
            weight: pd.Series | None = None) -> pd.DataFrame:
    """Every month-start that admits a full `years`-long window."""
    idx = total.index
    firsts = idx[pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("min") == idx]
    rows = [w for t in firsts if (w := window(total, t, years, rate, weight)) is not None]
    return pd.DataFrame(rows).set_index("start")


def safe_rate(total: pd.Series, start, years: float, weight=None,
              lo: float = 0.0, hi: float = 0.30, floor: float = 1.0) -> float:
    """Highest withdrawal rate whose account still ends at >= `floor` x its start.

    floor=1.0 asks for capital preservation in nominal terms: take income for
    `years` and hand back at least what you started with.
    """
    for _ in range(40):
        mid = (lo + hi) / 2
        r = window(total, start, years, mid, weight)
        if r is None:
            return float("nan")
        lo, hi = (mid, hi) if r["final_x"] >= floor else (lo, mid)
    return (lo + hi) / 2


def main(years: float = 25, rate: float = 0.054):
    strat, bh, w = paths()
    print(f"Sequence stress on {TICKER}, {strat.index[0].date()}..{strat.index[-1].date()} "
          f"({len(strat) / TRADING_DAYS:.0f} years).")
    print(f"{years:.0f}-year windows, {rate:.1%}/yr withdrawn, annual_reset, $1m start.\n")

    for nm, tot, wt in (("TIMING", strat, w), ("buy & hold", bh, None)):
        t = rolling(tot, years, rate, wt)
        tot_x = t["final_x"] + t["paid_x"]
        print(f"{nm}: {len(t)} overlapping start months, "
              f"{t.index[0].date()} .. {t.index[-1].date()}")
        print(f"  final x       worst {t['final_x'].min():5.2f}  median {t['final_x'].median():5.2f}"
              f"  best {t['final_x'].max():5.2f}")
        print(f"  total x       worst {tot_x.min():5.2f}  median {tot_x.median():5.2f}"
              f"  best {tot_x.max():5.2f}   (final + income paid)")
        print(f"  max drawdown  worst {t['max_dd'].min():5.1%}  median {t['max_dd'].median():5.1%}")
        print(f"  recover mths  worst {t['recover_months'].max():5.0f}  median "
              f"{t['recover_months'].median():5.0f}")
        print(f"  DEPLETED in {int(t['depleted'].sum())} of {len(t)}; ended below the starting "
              f"$1m in {int((t['final_x'] < 1).sum())}\n")

    print("!! These windows OVERLAP. 153 starts drawn from 38 years of one market share\n"
          "   almost all their bars, so they are nearer 1.5 independent 25-year samples\n"
          "   than 153. Every window here contains both 2000-02 and 2008, which is why\n"
          "   worst and median drawdown are identical. Shorter horizons vary more:\n")
    print(f"  {'horizon':<10}{'starts':>8}" + "".join(f"{h:>22}" for h in ("TIMING", "buy & hold")))
    print(f"  {'':<10}{'':>8}" + "".join(f"{'worst x / worst DD':>22}" for _ in range(2)))
    for h in (5, 10, 15, 20, 25):
        a, b = rolling(strat, h, rate, w), rolling(bh, h, rate)
        print(f"  {h:>2}-year   {len(a):>8}"
              f"{a['final_x'].min():>11.2f} /{a['max_dd'].min():>9.1%}"
              f"{b['final_x'].min():>11.2f} /{b['max_dd'].min():>9.1%}")
    print("\n  'worst x' is the least the account was ever worth at the end of a window,\n"
          "  per $1 started, after taking income throughout.\n")

    print("WORST ENTRY POINTS -- buying at the top, then withdrawing through it:")
    print(f"  {'entry':<22}{'TIMING final':>14}{'maxDD':>9}{'B&H final':>12}{'maxDD':>9}")
    for label, dt in (("1990-01 (pre-Gulf)", "1990-01-02"), ("2000-03 (dot-com top)", "2000-03-24"),
                      ("2007-10 (GFC top)", "2007-10-09"), ("2021-12 (2022 top)", "2021-12-27")):
        for h in (years, 18, 15, 10, 4):  # longest window the data allows from here
            a, b = window(strat, dt, h, rate, w), window(bh, dt, h, rate)
            if a and b:
                tag = "" if h == years else f"   ({h:.0f}y only)"
                print(f"  {label:<22}{a['final_x']:>14.2f}{a['max_dd']:>9.1%}"
                      f"{b['final_x']:>12.2f}{b['max_dd']:>9.1%}{tag}")
                break

    print(f"\nSUSTAINABLE WITHDRAWAL RATE (ends at >= the starting $1m after {years:.0f}y):")
    print(f"  {'entry':<22}{'TIMING':>10}{'buy & hold':>13}")
    for label, dt in (("1988-01 (inception)", "1988-01-05"), ("1990-01", "1990-01-02"),
                      ("2000-03 (worst)", "2000-03-24"), ("2001-09", "2001-09-13")):
        a, b = safe_rate(strat, dt, years, w), safe_rate(bh, dt, years)
        if np.isfinite(a):
            print(f"  {label:<22}{a:>10.1%}{b:>13.1%}")
    print(f"\n  The policy's {rate:.1%} is the number to compare against.")


def demo():
    """Self-check: windowing and the rate solver behave."""
    idx = pd.bdate_range("2000-01-03", periods=TRADING_DAYS * 6)
    flat = pd.Series(0.0, index=idx)

    # a zero-return account preserves capital only at a zero withdrawal rate
    assert safe_rate(flat, idx[0], 5, floor=1.0) < 0.005

    # an account compounding at g preserves capital at a rate of about g, which
    # is the property the whole exercise turns on
    for g in (0.04, 0.07, 0.10):
        grow = pd.Series((1 + g) ** (1 / TRADING_DAYS) - 1, index=idx)
        r = safe_rate(grow, idx[0], 5, floor=1.0)
        assert abs(r - g) < 0.006, (g, r)

    # too short a window returns None rather than a silently truncated answer
    assert window(flat, idx[0], 99) is None and window(flat, idx[0], 5) is not None

    # more withdrawal always leaves less behind
    assert window(flat, idx[0], 5, 0.02)["final_x"] > window(flat, idx[0], 5, 0.08)["final_x"]

    # rolling yields one row per admissible month start, and no partial windows
    t = rolling(flat, 5, 0.054)
    assert 10 <= len(t) <= 15, len(t)
    # no partial windows: the last admissible start still leaves a full 5 years
    # of BARS behind it (calendar years and trading days are not the same thing)
    assert (idx >= t.index.max()).sum() >= int(5 * TRADING_DAYS)
    assert (idx >= t.index.max()).sum() - int(5 * TRADING_DAYS) < TRADING_DAYS / 6
    print("stress ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
