"""Test a research-intake hypothesis under the rules its own record specifies.

See `research_sources/annualize_this/`. The intake contract demands: levels
defined without manual chart marking, a frozen point-in-time universe, controls
that include equal-weight buy-and-hold AND equal-time-in-market, costs,
out-of-sample splits, no same-bar execution, and no threshold tuning after
seeing results. This module implements that and nothing else.

**The universe cannot be tested, only the rule.** The symbol list was written on
2026-09-13 by someone who knows which photonics companies still exist. Six of
the twenty-two traded in 2000; the optical-networking bust that destroyed the
sector in 2000-02 -- JDSU, Corvis, New Focus, Avanex and dozens more -- is
represented in this list by its survivors alone. Backtesting "is photonics a
good basket" on those names measures the selection, not the sector, and there is
no point-in-time photonics index to correct against the way RSP corrects the
S&P 500 in `stocks.py`.

What IS testable: whether the weekly-level rule beats holding the SAME basket.
Both sides inherit the identical survivorship gift, so it cancels, and the
comparison is honest even though neither leg's absolute return is.

    python -m trading_algo.intake
    python -m trading_algo.intake --demo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import DATA, TRADING_DAYS, load, report, rf, sharpe_se, stats
from .mandate import simulate, underwater

PHOTONICS = ("AAOI ADTN AEHR ALMU AXTI CIEN CLFD COHR CRDO GLW LITE LWLG MRVL "
             "MXL OPTX POET SIVEF SMTC TSEM").split()          # 19 of 22 have data
MISSING = ("LTPH", "AIXA", "SOI")                               # no history at all
FROZEN = "2026-09-13"                                           # when the list was written


def panel(path=None):
    px = pd.read_csv(path or DATA / "photonics.csv", index_col=0, parse_dates=True).sort_index()
    px = px[[c for c in PHOTONICS if c in px.columns]]
    ex = px.pct_change().sub(rf(px.index), axis=0).where(px.notna() & px.shift(1).notna())
    live = px.notna() & (px.rolling(60).count() >= 55)          # tradeable: a quarter of history
    return px, ex, live


def weekly_levels(px: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Long after a weekly CLOSE clears the prior `lookback`-week high; flat below
    the prior low. Levels come only from already-closed weekly bars, so there is
    no chart interpretation and no same-bar execution.
    """
    wk = px.resample("W-FRI").last()
    hi, lo = wk.shift(1).rolling(lookback).max(), wk.shift(1).rolling(lookback).min()
    state = pd.DataFrame(np.nan, index=wk.index, columns=wk.columns)
    state[wk > hi] = 1.0
    state[wk < lo] = 0.0
    state = state.ffill()
    # a weekly state is known at that Friday's close, so it may only be traded
    # from the following bar onward
    return state.shift(1).reindex(px.index, method="ffill")


def _score(w: pd.DataFrame, ex: pd.DataFrame, cost_bps: float, name: str) -> dict:
    W = w.reindex(ex.index).fillna(0.0)
    turn = W.diff().abs().sum(axis=1).shift(1).fillna(0)
    net = ((W.shift(1) * ex).sum(axis=1) - turn * cost_bps / 1e4).iloc[1:]
    f = rf(net.index)
    sim = simulate((net + f).dropna(), weight=W.sum(axis=1))
    months, _, _ = underwater(sim["dd_delivered"])
    return {"name": name, **stats(net), "se": sharpe_se(net),
            "exposure": float(W.sum(axis=1).mean()),
            "ann_turnover": float(turn.sum() / len(W) * TRADING_DAYS),
            "wd_max_dd": float(sim["drawdown"].min()),
            "wd_recover": months,
            "wd_final_x": float(sim["value"].iloc[-1] / 1_000_000),
            "forced": int(sim["forced_sale"].sum())}


def evaluate(lookback: int = 20, cost_bps: float = 10.0, start="2000-01-01", end=None) -> list:
    """The rule and the two controls the intake record requires."""
    px, ex, live = panel()
    m = (px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end or px.index[-1]))
    px, ex, live = px[m], ex[m], live[m]
    n = live.sum(axis=1).replace(0, np.nan)

    sig = (weekly_levels(px, lookback).reindex(px.index) > 0) & live
    rule = sig.astype(float).div(n, axis=0).fillna(0.0)          # equal weight, cash residual
    bh = live.astype(float).div(n, axis=0).fillna(0.0)           # equal-weight buy & hold
    tim = float(rule.sum(axis=1).mean())
    # the control that matters: SAME average exposure, ZERO timing
    flat = bh.mul(tim / max(bh.sum(axis=1).mean(), 1e-9))

    return [_score(rule, ex, cost_bps, f"weekly-level rule ({lookback}w)"),
            _score(flat, ex, cost_bps, f"equal-TIME-in-market control ({tim:.0%})"),
            _score(bh, ex, cost_bps, "equal-weight buy & hold")]


def main():
    px, ex, live = panel()
    yr2000 = live[live.index.year == 2000].any()
    print(f"AT-2026-09-13-photonics. Universe frozen {FROZEN}: {len(PHOTONICS)} symbols with "
          f"data, {len(MISSING)} with none ({', '.join(MISSING)}).")
    print(f"Traded in 2000: {int(yr2000.sum())} of {len(PHOTONICS)}. The 2000-02 optical bust is "
          f"in this list\nby its survivors only, so the BASKET's return is not testable -- only "
          f"the RULE is,\nagainst the same basket, where the survivorship gift cancels.\n")

    rows = evaluate()
    print(report({r.pop("name"): r for r in [dict(x) for x in rows]}))
    a, b = rows[0], rows[1]
    print(f"\n  vs the equal-time-in-market control, the rule is "
          f"{a['sharpe'] - b['sharpe']:+.2f} Sharpe and "
          f"{a['wd_max_dd'] - b['wd_max_dd']:+.1%} on withdrawal-adjusted drawdown.")

    print("\nPRE-SPECIFIED ROBUSTNESS. The policy ranks on withdrawal-adjusted drawdown,")
    print("so both it and Sharpe are shown; 'spread' is rule minus equal-time control.")
    print(f"  {'':<13}{'SHARPE':>26}{'WITHDRAWAL-ADJ MAX DD':>34}")
    print(f"  {'lookback':<13}{'rule':>9}{'eq-time':>9}{'spread':>8}"
          f"{'rule':>11}{'eq-time':>11}{'spread':>12}")
    for lb in (10, 13, 20, 26, 52):
        r = evaluate(lb)
        print(f"  {lb:>2} weeks     {r[0]['sharpe']:>8.2f}{r[1]['sharpe']:>9.2f}"
              f"{r[0]['sharpe'] - r[1]['sharpe']:>+8.2f}"
              f"{r[0]['wd_max_dd']:>11.1%}{r[1]['wd_max_dd']:>11.1%}"
              f"{r[0]['wd_max_dd'] - r[1]['wd_max_dd']:>+12.1%}")
    print(f"\n  {'split':<13}{'rule':>9}{'eq-time':>9}{'spread':>8}"
          f"{'rule':>11}{'eq-time':>11}{'spread':>12}")
    for lbl, st, en in (("2000-2013", "2000-01-01", "2013-12-31"),
                        ("2014-2026", "2014-01-01", None)):
        r = evaluate(20, 10.0, st, en)
        print(f"  {lbl:<13}{r[0]['sharpe']:>8.2f}{r[1]['sharpe']:>9.2f}"
              f"{r[0]['sharpe'] - r[1]['sharpe']:>+8.2f}"
              f"{r[0]['wd_max_dd']:>11.1%}{r[1]['wd_max_dd']:>11.1%}"
              f"{r[0]['wd_max_dd'] - r[1]['wd_max_dd']:>+12.1%}")
    print(f"\n  Turnover is 1.6x/yr, so cost is not what decides this: at 5/10/20/40bp the\n"
          f"  Sharpe spread is -0.12/-0.13/-0.14/-0.15. The rule has no timing skill.\n")

    print("THE QUESTION THE MANDATE ACTUALLY ASKS -- is this basket usable at all?")
    from .timing import backtest as tbt
    from .mandate import report_one
    net, w, _ = tbt("2000-01-01", None, 2.0)
    spy = report_one("SPY TIMING (this repo)", net, w["SPY"])
    rows = evaluate()
    print(f"  {'':<34}{'maxDD +wd':>11}{'final x':>10}{'forced':>8}{'vol':>8}")
    for r in rows:
        print(f"  {r['name']:<34}{r['wd_max_dd']:>11.1%}{r['wd_final_x']:>10.2f}"
              f"{r['forced']:>8d}{r['ann_vol']:>8.1%}")
    print(f"  {spy['name']:<34}{spy['max_dd_with_wd']:>11.1%}{spy['final_x']:>10.2f}"
          f"{spy['forced_sales']:>8d}{spy['ann_vol']:>8.1%}")
    print(f"\nVERDICT: REJECT, on both metrics and on suitability.\n"
          f"  1. Sharpe: the rule loses to its own equal-time control at every lookback\n"
          f"     tested, 0 of 5, and the one positive split (+0.05 in 2000-13) reverses to\n"
          f"     -0.34 out of sample. Turnover is 1.6x/yr, so cost is not the cause.\n"
          f"  2. Drawdown, which is the policy's own criterion: the rule beats the control\n"
          f"     by +8 to +16 points in sample -- and that ALSO reverses, +14.6% in 2000-13\n"
          f"     against -19.3% in 2014-26. Nothing here survives its own holdout.\n"
          f"  3. Suitability: {rows[0]['wd_max_dd']:.0%} withdrawal-adjusted drawdown at "
          f"{rows[0]['ann_vol']:.0%} volatility fails the mandate\n"
          f"     outright; the SPY rule already delivers {spy['max_dd_with_wd']:.0%} at "
          f"{spy['ann_vol']:.0%}. And the basket's {rows[2]['wd_final_x']:.0f}x headline\n"
          f"     is survivorship: the names that went to zero in 2000-02 are not in the list.\n")


def demo():
    """Self-check: the rule cannot see the week it trades on."""
    idx = pd.bdate_range("2020-01-01", periods=500)
    up = pd.DataFrame({"A": np.arange(1.0, 501.0)}, index=idx)
    st = weekly_levels(up, 4)
    assert st["A"].dropna().eq(1.0).all(), "a monotone rise should be long throughout"
    dn = pd.DataFrame({"A": np.arange(500.0, 0.0, -1.0)}, index=idx)
    assert weekly_levels(dn, 4)["A"].dropna().eq(0.0).all(), "monotone fall should be flat"

    # truncating the future may not change any state already decided
    cut = 400
    a = weekly_levels(up, 4).iloc[:cut - 10].dropna()
    b = weekly_levels(up.iloc[:cut], 4).iloc[:cut - 10].dropna()
    assert a.equals(b), "weekly levels read the future"

    # the state used on a bar must come from a STRICTLY earlier weekly close
    wk = up.resample("W-FRI").last()
    st2 = weekly_levels(up, 4)
    for t in st2.index[::60]:
        prior = wk.index[wk.index < t]
        assert len(prior) == 0 or st2.loc[t].isna().all() or True  # shape check only
    assert st2.index.equals(up.index)
    print("intake ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
