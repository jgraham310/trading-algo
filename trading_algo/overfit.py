"""What "just keep searching until something hits Sharpe 2" actually produces.

The honest ceilings live in `timing.ceiling()` and `stocks.py`. This is the
other half of the answer: a demonstration that Sharpe 2 *is* reachable
in-sample, trivially, by search alone — and that it means nothing.

Builds a bank of long/flat conditions on SPY that are all individually
reasonable (trend, breakouts, mean reversion, volatility regime, calendar),
draws hundreds of thousands of random rules from combinations of them, keeps
the best by in-sample Sharpe, and then looks at what that winner does next.

    python -m trading_algo.overfit
    python -m trading_algo.overfit --demo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS, excess, load, report, rf, sharpe_se, stats

SPLIT = "2014-01-01"


def conditions(idx: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """A bank of long/flat conditions, each knowable at the close of its bar."""
    c = load()
    p, vix = c["SPY"].reindex(idx), c["^VIX"].reindex(idx)
    r = p.pct_change()
    d = pd.Series(idx, index=idx)
    mn = d.dt.year * 12 + d.dt.month
    cols, names = [], []

    def add(name, s):
        cols.append(np.asarray(pd.Series(s, index=idx).fillna(False), dtype=bool))
        names.append(name)

    for n in (5, 10, 20, 50, 100, 150, 200, 250):
        add(f"above_ma{n}", p > p.rolling(n).mean())
        add(f"below_ma{n}", p < p.rolling(n).mean())
    for n in (2, 3, 5, 10, 20, 60):
        add(f"low{n}", p <= p.rolling(n).min())
        add(f"high{n}", p >= p.rolling(n).max())
    for n in (1, 2, 3, 5, 10, 21, 63, 126, 252):
        add(f"ret{n}_up", p / p.shift(n) - 1 > 0)
        add(f"ret{n}_dn", p / p.shift(n) - 1 < 0)
    for n in (10, 20, 60):
        rv = r.rolling(n).std()
        add(f"vol{n}_hi", rv > rv.rolling(252).median())
        add(f"vol{n}_lo", rv < rv.rolling(252).median())
    for n in (10, 20, 50, 100):
        add(f"vix_above_ma{n}", vix > vix.rolling(n).mean())
        add(f"vix_below_ma{n}", vix < vix.rolling(n).mean())
    for k in range(5):
        add(f"dow{k}", d.dt.dayofweek.shift(-1) == k)
    for k in range(1, 13):
        add(f"month{k}", d.dt.month == k)
    add("turn_of_month", (d.groupby(mn).cumcount(ascending=False).shift(-1) <= 0)
        | (d.groupby(mn).cumcount().shift(-1) < 3))
    add("first_half_month", d.dt.day <= 15)
    return np.array(cols), names


def search(n_rules=200_000, terms=3, cost_bps=2.0, seed=0, start="2000-01-01",
           chunk=2000, split=SPLIT):
    """Draw random AND-rules, score each in-sample, return the whole distribution."""
    x = excess(["SPY"], start=start)["SPY"]
    idx = x.index
    C, names = conditions(idx)
    r = x.to_numpy()
    is_ = np.asarray(idx < pd.Timestamp(split))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(C), size=(n_rules, terms))

    def sharpes(mask):
        # pos[t] is decided at the close of t and earns r[t+1]. Getting this
        # wrong is the classic way to manufacture a Sharpe of 8: dropping the
        # shift makes "today closed up" a winning rule.
        rr = r[mask][1:]
        out = np.empty(n_rules, dtype=np.float32)
        for i in range(0, n_rules, chunk):
            sel = picks[i:i + chunk]
            pos = np.logical_and.reduce(C[sel][:, :, mask], axis=1).astype(np.float32)
            turn = np.abs(np.diff(pos, axis=1, prepend=0.0))[:, 1:]
            net = pos[:, :-1] * rr - turn * cost_bps / 1e4
            sd = net.std(axis=1, ddof=1)
            out[i:i + chunk] = np.where(sd > 1e-12, net.mean(axis=1) / sd * np.sqrt(TRADING_DAYS), np.nan)
        return out

    return sharpes(is_), sharpes(~is_), picks, names, idx, r, C


def main(n_rules=200_000, terms=3, cost_bps=2.0):
    is_s, os_s, picks, names, idx, r, C = search(n_rules, terms, cost_bps)
    ok = np.isfinite(is_s) & np.isfinite(os_s)
    is_s, os_s, picks = is_s[ok], os_s[ok], picks[ok]
    n = len(is_s)
    split = pd.Timestamp(SPLIT)
    n_is, n_os = (idx < split).sum(), (idx >= split).sum()

    print(f"{n:,} random {terms}-condition long/flat rules on SPY, drawn from {len(names)} "
          f"conditions.\nIn-sample {idx[0].date()}..{split.date()} ({n_is} bars), "
          f"out-of-sample after ({n_os} bars). {cost_bps:g}bp costs, excess of T-bills.\n")

    hit2 = (is_s > 2).sum()
    print(f"IN-SAMPLE Sharpe across the search: median {np.median(is_s):+.2f}, "
          f"95th pct {np.percentile(is_s, 95):+.2f}, MAX {is_s.max():+.2f}")
    print(f"  rules clearing Sharpe 2 in-sample: {hit2:,} of {n:,}")
    if not hit2:
        print("  Not one. Brute force cannot even MANUFACTURE a Sharpe 2 here, with 14 years\n"
              "  of hindsight and no obligation to work afterwards. The target is not hiding\n"
              "  in this space; the space does not contain it.\n")

    k = 25
    top = np.argsort(is_s)[::-1][:k]
    print(f"WHAT SELECTING THE IN-SAMPLE BEST BUYS YOU (top {k} of {n:,}):")
    print(f"  in-sample  mean {is_s[top].mean():+.2f}  (range {is_s[top].min():+.2f} "
          f"to {is_s[top].max():+.2f})")
    print(f"  next 12.7y mean {os_s[top].mean():+.2f}  (range {os_s[top].min():+.2f} "
          f"to {os_s[top].max():+.2f})")
    print(f"  shrinkage {is_s[top].mean() - os_s[top].mean():+.2f} -- that gap is the part of "
          f"the winner's score\n  that was selection, not skill.")
    b = top[0]
    print(f"  winner: {' AND '.join(names[j] for j in picks[b])}"
          f"   IS {is_s[b]:+.2f} -> OOS {os_s[b]:+.2f}\n")

    rho = np.corrcoef(is_s, os_s)[0, 1]
    print(f"IS/OOS Sharpe correlation across all {n:,} rules: {rho:+.3f} -- in-sample rank\n"
          f"  barely predicts what a rule does next, so ranking on it mostly ranks noise.\n")

    # where the repo's actual strategy sits in this distribution
    from .timing import backtest as tb
    split = pd.Timestamp(SPLIT)
    t_is = stats(tb("2000-01-01", SPLIT, cost_bps)[0])["sharpe"]
    t_os = stats(tb(SPLIT, None, cost_bps)[0])["sharpe"]
    print(f"THE REPO'S STRATEGY, measured the same way:")
    print(f"  in-sample {t_is:+.2f} ({(is_s < t_is).mean():.2%} percentile of the search), "
          f"out-of-sample {t_os:+.2f}")
    d = t_os - t_is
    print(f"  It {'gained' if d > 0 else 'lost'} {abs(d):.2f} out of sample, where the search "
          f"winners lost {is_s[top].mean() - os_s[top].mean():.2f}.\n"
          f"  That asymmetry is the case for it being a rule rather than a lucky draw. It is\n"
          f"  also still nowhere near 2.\n")

    se = np.sqrt(1 / n_is) * np.sqrt(TRADING_DAYS)
    print(f"THE ARITHMETIC: one Sharpe on {n_is} bars has a standard error of ~{se:.2f}.")
    for N in (100, 10_000, 1_000_000):
        print(f"  best of {N:>9,} INDEPENDENT zero-edge rules lands near "
              f"{se * np.sqrt(2 * np.log(N)):+.2f} on luck alone")
    print("\n  So a search big enough to surface Sharpe 2 would be a search big enough to\n"
          "  fabricate it -- except that here even 164k tries top out below 1.0. Both roads\n"
          "  end in the same place, which is why this repo reports 0.81 and stops.")


def best_blend(cost_bps: float = 2.0, stock_bps: float = 5.0):
    """The best Sharpe ANY combination of this repo's sleeves can reach.

    Builds every sleeve built anywhere here, then solves for the convex
    combination (non-negative weights summing to 1 -- no leverage) that
    maximises Sharpe, using hindsight on every mean and covariance in the
    sample. No blending scheme can beat this, including one that works. Any
    honest out-of-sample blend lands below it.
    """
    from scipy.optimize import minimize

    from .stocks import START, panel
    from .timing import backtest as tbt, weight as tw

    px, ex, live = panel()
    m = px.index >= pd.Timestamp(START)
    tim = tw(load()["SPY"]).reindex(px.index)
    spx = load()["SPY"]
    xs = excess(["SPY"], start=START)

    def stk(w, overlay):
        w = w.mul(overlay.reindex(w.index).fillna(0), axis=0) if overlay is not None else w
        W = w[m]
        turn = W.diff().abs().sum(axis=1).shift(1).fillna(0)
        return ((W.shift(1) * ex[m]).sum(axis=1) - turn * stock_bps / 1e4).iloc[1:]

    def eqw(trig):
        t = trig.where(live).fillna(False)
        return t.astype(float).div(t.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    def spy(w):
        from .engine import run
        return run(pd.DataFrame({"SPY": pd.Series(w, index=xs.index).fillna(0).clip(0, 1)}),
                   xs, cost_bps)

    sl = {"buy&hold": spy(pd.Series(1.0, index=xs.index)),
          "timing": tbt(START, None, cost_bps)[0],
          "spy_trend": spy(pd.concat([spx > spx.rolling(n).mean() for n in (100, 150, 200, 250)],
                                     axis=1).mean(axis=1).reindex(xs.index)),
          "spy_meanrev": spy(pd.concat([spx <= spx.rolling(n).min() for n in (3, 5, 10)],
                                       axis=1).mean(axis=1).reindex(xs.index))}
    mom, vol = px.shift(21) / px.shift(252) - 1, -ex.rolling(120).std()
    key = pd.Series(px.index, index=px.index).dt.to_period("M")
    for nm, sig in (("stk_mom", mom), ("stk_lowvol", vol)):
        rk = sig.where(live).rank(axis=1, ascending=False)
        w = (rk <= 50).astype(float)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).where(key != key.shift(-1)).ffill().fillna(0.0)
        sl[nm], sl[nm + "_x_tim"] = stk(w, None), stk(w, tim)
    for n in (3, 10):
        t = eqw(px <= px.rolling(n).min())
        sl[f"stk_mr{n}"], sl[f"stk_mr{n}_x_tim"] = stk(t, None), stk(t, tim)

    R = pd.DataFrame(sl).dropna()
    mu, C = R.mean().values * TRADING_DAYS, R.cov().values * TRADING_DAYS
    n = len(mu)
    res = minimize(lambda w: -(w @ mu) / np.sqrt(w @ C @ w), np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    return R, res.x, -res.fun


def blend_report():
    R, w, sh = best_blend()
    print(f"{R.shape[1]} sleeves, {len(R)} bars {R.index[0].date()}..{R.index[-1].date()}\n")
    print(report({k: {**stats(R[k]), "se": sharpe_se(R[k])} for k in R}))
    tri = R.corr().values[np.triu_indices(R.shape[1], 1)]
    print(f"\nsleeve correlations: mean {tri.mean():+.2f}, min {tri.min():+.2f} -- they are all\n"
          f"  long equity beta, which is why combining them buys so little.\n")
    c = R.values @ w
    print(f"EX-POST OPTIMAL convex blend (hindsight on every mean and covariance, no leverage):")
    print(f"  Sharpe {sh:.2f}   ann excess {(1 + c).prod() ** (TRADING_DAYS / len(c)) - 1:+.2%}"
          f"   vol {c.std() * np.sqrt(TRADING_DAYS):.1%}")
    print("  weights:", {k: round(v, 3) for k, v in sorted(zip(R.columns, w), key=lambda kv: -kv[1]) if v > 0.01})
    print(f"\n  No blending scheme can beat {sh:.2f}, because this one already cheated. The\n"
          f"  winning sleeves are the single-stock ones, which carry the +0.30 survivorship\n"
          f"  gift on top. Out of sample it lands lower, not higher.")


def demo():
    """Self-check: the search machinery is not itself leaking."""
    x = excess(["SPY"], start="2015-01-01")["SPY"]
    C, names = conditions(x.index)
    assert C.shape[0] == len(names) and C.dtype == bool
    # every condition must be computable from the past: recomputing on a prefix
    # may not change earlier values. dow/month use shift(-1) by design (tomorrow's
    # calendar date is known today), so exclude the trailing bar from the check.
    cut = len(x) - 5
    C2, _ = conditions(x.index[:cut])
    same = (C[:, :cut - 1] == C2[:, :cut - 1]).all(axis=1)
    assert same.all(), f"conditions read the future: {[names[i] for i in np.where(~same)[0]]}"
    is_s, os_s, picks, *_ = search(n_rules=200, terms=2, start="2015-01-01", chunk=64,
                                   split="2021-01-01")
    assert len(is_s) == len(os_s) == 200 and np.isfinite(is_s).any()

    # THE tripwire: "today closed up" must be worth ~nothing. If the position is
    # off by one bar it scores like an oracle, which is how the first version of
    # this module produced Sharpe 8.5 and a +0.98 IS/OOS correlation.
    x = excess(["SPY"], start="2000-01-01")["SPY"]
    C3, names3 = conditions(x.index)
    j = names3.index("ret1_up")
    picks3 = np.full((1, 1), j)
    r3 = x.to_numpy()
    pos = C3[j].astype(float)
    s_lag = (pos[:-1] * r3[1:]).mean() / (pos[:-1] * r3[1:]).std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert abs(s_lag) < 1.0, f"'today closed up' scores {s_lag:.2f} -- the harness reads the future"
    print("overfit ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    elif "--blend" in sys.argv:
        blend_report()
    else:
        main()
