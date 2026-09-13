"""Evaluate strategies against INVESTMENT_POLICY.md, not against Sharpe.

The policy ranks candidates by **withdrawal-adjusted downside**: maximum
drawdown, drawdown duration, and recovery time under a 5.4%/yr equal-monthly
withdrawal. Return is a constraint, not the objective. A Sharpe table cannot
answer that question, because Sharpe is blind to the one thing that destroys a
portfolio paying income: sequence-of-returns risk. Selling units into a 56%
drawdown is permanent damage no later rally reverses.

Three readings of "5.4% annual, equal-monthly, 0.45% of the applicable annual
account value" are implemented, because they are not the same policy and they
do not give the same answer -- see `WITHDRAWAL_MODES`. `annual_reset` is the
default as the reading most consistent with both "equal-dollar installments"
and "of the applicable annual account value".

    python -m trading_algo.mandate           # the full policy report
    python -m trading_algo.mandate --demo    # self-check, no download
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS, excess, load, rf, run, stats

WITHDRAWAL_MODES = {
    "annual_reset": "5.4% of the account value each 1 Jan, paid in 12 equal dollar amounts",
    "pct_current": "0.45% of the CURRENT account value every month (self-adjusting, never depletes)",
    "fixed_nominal": "5.4% of the STARTING value, fixed dollars forever (harshest, ignores inflation)",
}


def simulate(total_ret: pd.Series, rate: float = 0.054, mode: str = "annual_reset",
             initial: float = 1_000_000.0, weight: pd.Series | None = None,
             buffer_months: float = 0.0) -> pd.DataFrame:
    """Daily account path with monthly withdrawals taken at each month's first bar.

    `total_ret` is the strategy's TOTAL return (excess + bill rate), because a
    withdrawal is paid out of the whole account, not out of its excess.
    `weight` is the equity weight, used only to detect forced sales: a
    withdrawal is a forced sale when the cash sleeve cannot cover it.
    """
    r = pd.Series(total_ret).dropna()
    idx = r.index
    first = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("min") == idx
    jan = pd.Series(idx, index=idx).groupby(idx.year).transform("min") == idx

    v = initial
    annual = initial * rate
    rows = []
    for t, ret, isfirst, isjan in zip(idx, r.to_numpy(), first.to_numpy(), jan.to_numpy()):
        v *= 1 + ret
        if isjan and mode == "annual_reset":
            annual = max(v, 0.0) * rate
        w = 0.0
        if isfirst and v > 0:
            if mode == "pct_current":
                w = v * rate / 12
            elif mode == "fixed_nominal":
                w = initial * rate / 12
            else:
                w = annual / 12
            w = min(w, v)
            v -= w
        rows.append((t, v, w))
        if v <= 0:
            break

    out = pd.DataFrame(rows, columns=["date", "value", "withdrawal"]).set_index("date")
    out["drawdown"] = out["value"] / out["value"].cummax() - 1
    # An account paying out 5.4%/yr barely makes new highs, so "months below the
    # last peak in account value" saturates near 100% and says nothing. The
    # honest duration measure is drawdown of value PLUS income already taken --
    # total wealth delivered, which is what actually has to recover.
    delivered = out["value"] + out["withdrawal"].cumsum()
    out["delivered"] = delivered
    out["dd_delivered"] = delivered / delivered.cummax() - 1
    if weight is not None:
        cash = (1 - pd.Series(weight).reindex(out.index).fillna(0).clip(0, 1)) * out["value"]
        need = out["withdrawal"] + buffer_months * out["withdrawal"]
        out["forced_sale"] = (out["withdrawal"] > 0) & (cash < need) & (out["drawdown"] < -0.05)
    return out


def underwater(dd: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Longest unbroken stretch below a prior peak, in months, and when it ran.

    This is drawdown duration AND recovery time in one number: the span from the
    peak to the day the account next exceeds it. For an account paying income it
    matters more than the depth, because every month under water sells units
    that never come back.
    """
    under = dd < -1e-9
    if not under.any():
        return 0.0, None, None
    grp = (~under).cumsum()[under]
    sizes = grp.groupby(grp).size()
    worst = sizes.idxmax()
    span = grp[grp == worst].index
    return float(sizes.max() / TRADING_DAYS * 12), span[0], span[-1]


def report_one(name: str, net_excess: pd.Series, weight: pd.Series | None,
               mode: str = "annual_reset", **kw) -> dict:
    """Every number INVESTMENT_POLICY.md requires, for one strategy."""
    f = rf(net_excess.index)
    total = (net_excess + f).dropna()
    g = stats(total)
    sim = simulate(total, mode=mode, weight=weight, **kw)
    dd_months, uw0, uw1 = underwater(sim["dd_delivered"])
    depleted = sim["value"].iloc[-1] <= 0
    yrs = len(total) / TRADING_DAYS
    return {
        "name": name,
        "ann_return": g["ann_excess"],
        "ann_vol": g["ann_vol"],
        "max_dd_no_wd": g["max_dd"],
        "max_dd_with_wd": float(sim["drawdown"].min()),
        "max_dd_delivered": float(sim["dd_delivered"].min()),
        "dd_months": dd_months,
        "pct_underwater": float((sim["dd_delivered"] < -1e-9).mean()),
        "final_x": float(sim["value"].iloc[-1] / 1_000_000) if not depleted else 0.0,
        "withdrawn_x": float(sim["withdrawal"].sum() / 1_000_000),
        "depleted": bool(depleted),
        "forced_sales": int(sim["forced_sale"].sum()) if "forced_sale" in sim else -1,
        "years": yrs,
        "uw_from": uw0, "uw_to": uw1,
        "depleted_on": sim.index[-1] if depleted else None,
        "sim": sim,
    }


def candidates(start="2000-01-01", cost_bps=2.0):
    """The strategies this repo has produced, as (name, net excess, weight)."""
    from .timing import backtest as tbt, weight as tw

    out = []
    net, w, x = tbt(start, None, cost_bps)
    out.append(("TIMING (this repo)", net, w["SPY"]))
    bh = run(pd.DataFrame(1.0, index=x.index, columns=["SPY"]), x, cost_bps)
    out.append(("buy & hold SPY", bh, pd.Series(1.0, index=x.index)))
    spx = load()["SPY"]
    trend = (spx > spx.rolling(200).mean()).astype(float).reindex(x.index)
    out.append(("MA200 only", run(pd.DataFrame({"SPY": trend}), x, cost_bps), trend))
    half = pd.Series(0.5, index=x.index)
    out.append(("static 50% SPY / 50% cash", run(pd.DataFrame({"SPY": half}), x, cost_bps), half))
    return out


def main(mode: str = "annual_reset"):
    print("INVESTMENT_POLICY.md evaluation -- ranked by withdrawal-adjusted drawdown,\n"
          "not by Sharpe. $1,000,000 initial, 5.4%/yr withdrawn in equal monthly dollars.\n")
    for m, desc in WITHDRAWAL_MODES.items():
        print(f"  {'>' if m == mode else ' '} {m:<14} {desc}")
    print()

    rows = [report_one(n, s, w, mode) for n, s, w in candidates()]
    rows.sort(key=lambda r: r["max_dd_with_wd"], reverse=True)
    hdr = (f"{'strategy':<26}{'ret':>7}{'vol':>7}{'maxDD':>8}{'maxDD':>8}{'maxDD':>9}"
           f"{'recovr':>7}{'final':>7}{'paid':>6}{'total':>6}{'forced':>7}")
    print(hdr)
    print(f"{'':<26}{'':>7}{'':>7}{'no wd':>8}{'acct':>8}{'delivrd':>9}{'mths':>7}"
          f"{'x':>7}{'x':>6}{'x':>6}{'sales':>7}")
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['ann_return']:>+7.1%}{r['ann_vol']:>7.1%}"
              f"{r['max_dd_no_wd']:>8.1%}{r['max_dd_with_wd']:>8.1%}{r['max_dd_delivered']:>9.1%}"
              f"{r['dd_months']:>7.0f}{r['final_x']:>7.2f}{r['withdrawn_x']:>6.2f}"
              f"{r['final_x'] + r['withdrawn_x']:>6.2f}{r['forced_sales']:>7d}")
    print(f"\n  'final x' is the ending account per $1 started, after {rows[0]['years']:.0f} years of "
          f"withdrawals;\n  'paid x' is cumulative income taken; 'total x' is what the strategy "
          f"actually\n  delivered. 'maxDD delivered' is drawdown of (account + income already\n"
          f"  paid) -- the account-value drawdown of an income portfolio never recovers\n"
          f"  by construction, so it cannot measure duration. 'forced sales' counts months\n"
          f"  where the cash sleeve could not fund the withdrawal while >5% under water.\n")
    for r in rows:
        if r["depleted"]:
            print(f"  ** {r['name']} DEPLETED THE ACCOUNT on {r['depleted_on'].date()} **")
    print()

    print("SENSITIVITY to the withdrawal definition (the policy admits three readings):")
    print(f"  {'strategy':<26}" + "".join(f"{m:>16}" for m in WITHDRAWAL_MODES))
    for n, s, w in candidates():
        cells = "".join(f"{report_one(n, s, w, m)['max_dd_with_wd']:>16.1%}"
                        for m in WITHDRAWAL_MODES)
        print(f"  {n:<26}{cells}")
    print("\n  Worst-case max drawdown under withdrawals, by reading. Which one Jason\n"
          "  intends changes the numbers but not the ranking.")


def demo():
    """Self-check: the withdrawal machinery does what the policy says."""
    idx = pd.bdate_range("2000-01-03", periods=TRADING_DAYS * 3)
    flat = pd.Series(0.0, index=idx)

    # a zero-return account paying fixed nominal 5.4% loses exactly 0.45%/month
    n_months = len(pd.Series(idx).dt.to_period("M").unique())
    s = simulate(flat, mode="fixed_nominal")
    assert abs(s["withdrawal"].sum() - n_months * 4_500) < 1e-6, s["withdrawal"].sum()
    assert abs(s["value"].iloc[-1] - (1_000_000 - n_months * 4_500)) < 1e-6

    # pct_current can never deplete: it always takes a fraction of what is left
    s2 = simulate(pd.Series(-0.002, index=idx), mode="pct_current")
    assert (s2["value"] > 0).all(), "percentage-of-current withdrawal depleted the account"

    # exactly one withdrawal per calendar month, always on that month's first bar
    assert (s["withdrawal"] > 0).sum() == n_months

    # a 100% loss depletes and the simulation stops rather than going negative
    s3 = simulate(pd.Series(-0.5, index=idx), mode="annual_reset")
    assert s3["value"].iloc[-1] <= 0 and (s3["value"] >= 0).all()

    # withdrawals must make drawdown worse, never better, for a falling account
    down = pd.Series(np.r_[np.zeros(250), np.full(250, -0.004), np.zeros(256)], index=idx)
    a = simulate(down, mode="fixed_nominal")["drawdown"].min()
    b = simulate(down, rate=0.0)["drawdown"].min()
    assert a < b, "withdrawals did not deepen the drawdown"
    print("mandate ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
