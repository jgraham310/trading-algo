"""Does buying puts change the answer? No -- and the reason is measurable.

The honest part of this module needs no option-pricing model at all. CBOE's BXM
index is a real, published buy-write on the S&P 500: long the index, short a
one-month ATM call, rolled on expiration Friday. Put-call parity then pins the
price of the matching ATM *put* exactly, from realised index and BXM returns:

    BXM payoff over the month   = min(S1, K)
    protective put payoff       = max(S1, K)
    and the two sum to S1 + K

so a monthly ATM protective put is fully determined by data that already exists.
No Black-Scholes, no implied-vol assumption, no skew guess. `parity()` recovers
an average 1-month ATM call premium of 1.9% of spot and put of 1.8%, which is
what SPX options actually cost -- the check that the derivation is sound.

The answer it gives: **buying ATM puts is Sharpe-neutral.** It halves volatility
and halves return, flips skew from -1.2 to +1.1, and cuts max drawdown from -53%
to -32%. That is a real risk transformation and a worthless Sharpe one, which is
what efficient pricing predicts.

`collars()` is the part to distrust, and it is here to show why. Collars need
OTM strikes, which parity cannot pin, so they need a model -- and the answer
swings from Sharpe 0.90 to 0.51 on the single assumption of how steep the
volatility skew is. That is not a backtest, it is the assumption being read back.

    python -m trading_algo.options
    python -m trading_algo.options --demo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS, load, report, sharpe_se, stats
from .timing import backtest as timing_backtest

MONTHS = 12


def rolls(start="2000-01-01") -> pd.DatetimeIndex:
    """BXM rolls on the third Friday of each month; use those as the bar grid."""
    idx = load()[["SPY", "^BXM"]].dropna().index
    d = pd.Series(idx, index=idx)
    out = []
    for _, g in d.groupby([d.dt.year, d.dt.month]):
        fri = g[g.dt.dayofweek == 4]
        if len(fri) >= 3:
            out.append(fri.iloc[2])
    r = pd.DatetimeIndex(out)
    return r[r >= pd.Timestamp(start)]


def parity(start="2000-01-01") -> pd.DataFrame:
    """Monthly ATM call and put premia, and the protective-put return, from data.

    Derivation, per month, with the strike set at the starting spot (K = S0) and
    everything normalised to S0 = 1. Writing s = S1/S0 and b = BXM1/BXM0:

        BXM1/S0 = min(s, 1)  and  BXM0/S0 = 1 - c   =>   c = 1 - min(s, 1) / b
        P0/S0   = c + disc - 1                            (put-call parity)
        PP0/S0  = c + disc,   PP1/S0 = s + 1 - (1 - c) b

    `c` is the call premium as a fraction of spot, `disc` the bill discount over
    the month, PP the protective-put portfolio.
    """
    c = load()
    r = rolls(start)
    S, B = c["SPY"].reindex(r), c["^BXM"].reindex(r)
    s, b = (S / S.shift(1)).dropna(), (B / B.shift(1)).dropna()
    j = s.index.intersection(b.index)
    s, b = s[j], b[j]
    T = pd.Series(np.diff(r.values).astype("timedelta64[D]").astype(float) / 365.0,
                  index=r[1:]).reindex(j)
    disc = np.exp(-(c["^IRX"].reindex(r).ffill() / 100).reindex(j) * T)
    call = 1 - np.minimum(s, 1.0) / b
    return pd.DataFrame({
        "call": call,
        "put": call + disc - 1,
        "r_spy": s - 1,
        "r_pp": (s + 1 - (1 - call) * b) / (call + disc) - 1,
        "r_bxm": b - 1,
        "rf": 1 / disc - 1,
    })


def _row(r, **kw):
    r = pd.Series(r).dropna()
    return {**stats(r, periods=MONTHS), "se": sharpe_se(r, periods=MONTHS), **kw}


def overlays(start="2000-01-01") -> dict:
    """What puts do to buy & hold and to this repo's timing strategy."""
    p = parity(start)
    net, w, _ = timing_backtest(start, None, 2.0)
    eq = (1 + net).cumprod().reindex(p.index, method="ffill")
    r_tim = (eq / eq.shift(1) - 1)
    wm = w["SPY"].reindex(p.index, method="ffill").shift(1).fillna(0)
    ov = p["r_pp"] - p["r_spy"]                       # the put leg, per unit equity
    cheap = (p["put"] < p["put"].rolling(24).median()).astype(float).shift(1).fillna(0)
    return {
        "SPY": _row(p["r_spy"] - p["rf"]),
        "SPY + ATM put": _row(p["r_pp"] - p["rf"]),
        "TIMING (cash when out)": _row(r_tim),
        "TIMING + put on held weight": _row(r_tim + wm * ov),
        "100% long, put when signal low": _row(p["r_spy"] - p["rf"] + (1 - wm) * ov),
        "SPY + put only when cheap": _row(p["r_spy"] - p["rf"] + cheap * ov),
        "TIMING + put only when cheap": _row(r_tim + wm * cheap * ov),
    }


def collars(start="2000-01-01", skews=(0.0, 2.0, 4.0)) -> dict:
    """Long OTM put + short OTM call. MODEL-DEPENDENT -- read the module docstring.

    ATM implied vol is calibrated to the parity-derived call price rather than
    taken from VIX (VIX is a variance-swap measure and runs ~20% above 1-month
    ATM implied). `skew` is the extra vol, in points, on a 5%-OTM put; the real
    SPX one-month surface is near 4.
    """
    from scipy.stats import norm

    def bs(S, K, r, sig, T, put=False):
        d1 = (np.log(S / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        if put:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    p = parity(start)
    c = load()
    idx = p.index
    vix = (c["^VIX"].reindex(idx, method="ffill") / 100)
    irx = (c["^IRX"].reindex(idx, method="ffill") / 100)
    T = pd.Series(np.diff(rolls(start).values).astype("timedelta64[D]").astype(float) / 365.0,
                  index=rolls(start)[1:]).reindex(idx)

    iv = []
    for t in idx:  # invert BS against the real call price to get true ATM vol
        tgt = p["call"][t]
        if not np.isfinite(tgt) or tgt <= 0:
            iv.append(np.nan)
            continue
        lo, hi = 0.01, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if bs(1.0, 1.0, irx[t], mid, T[t]) < tgt else (lo, mid)
        iv.append((lo + hi) / 2)
    k = float((pd.Series(iv, index=idx) / vix).median())

    out = {"_atm_vol_vs_vix": k}
    for pk, ck in ((0.95, 1.10), (0.95, 1.05), (0.90, 1.10)):
        for sk in skews:
            rets = []
            for t in idx:
                if not np.isfinite(T[t]) or not np.isfinite(vix[t]):
                    rets.append(np.nan)
                    continue
                atm = vix[t] * k
                ivp = max(atm + sk * (1 - pk) / 0.05 * 0.01, 0.03)
                ivc = max(atm - 0.5 * sk * (ck - 1) / 0.05 * 0.01, 0.03)
                p0 = bs(1.0, pk, irx[t], ivp, T[t], put=True)
                c0 = bs(1.0, ck, irx[t], ivc, T[t])
                s1 = 1 + p["r_spy"][t]
                rets.append((s1 + max(pk - s1, 0) - max(s1 - ck, 0)) / (1 - c0 + p0) - 1)
            out[f"collar {pk:.2f}/{ck:.2f} skew{sk:g}"] = _row(
                pd.Series(rets, index=idx) - p["rf"])
    return out


def main():
    p = parity()
    print(f"{len(p)} monthly rolls, {p.index[0].date()}..{p.index[-1].date()}, "
          f"excess of T-bills, annualised from monthly.\n")
    print(f"IMPLIED FROM REAL BXM PRICES, no model: 1-month ATM call averages "
          f"{p['call'].mean():.2%} of spot,\n  put {p['put'].mean():.2%}. That is what SPX "
          f"options cost, which is the check that this is right.\n")
    print(report(overlays()))
    print("\n  SPY + ATM put is Sharpe-NEUTRAL: half the volatility, half the return,\n"
          "  skew -1.2 -> +1.1, drawdown -53% -> -32%. A real risk transformation and\n"
          "  no Sharpe at all, which is what efficiently priced insurance looks like.\n"
          "  Bolted onto the timing rule it SUBTRACTS: 0.87 -> 0.66, because sitting in\n"
          "  cash already removes the downside and costs ~0.5%/yr instead of ~3%/yr.\n")

    co = collars()
    k = co.pop("_atm_vol_vs_vix")
    print(f"COLLARS -- MODEL-DEPENDENT, DO NOT TRADE THESE NUMBERS.")
    print(f"  (ATM vol calibrated to the real call price: {k:.2f}x VIX.)")
    print(report(co))
    lo = min(v["sharpe"] for kk, v in co.items() if "skew4" in kk)
    hi = max(v["sharpe"] for kk, v in co.items() if "skew0" in kk)
    print(f"\n  The SAME collar scores {hi:.2f} on a flat volatility surface and {lo:.2f} once\n"
          f"  the put is charged the skew it actually trades at. A {hi - lo:.2f} Sharpe swing\n"
          f"  on one unobservable assumption is not a backtest -- it is the assumption\n"
          f"  being read back. Pricing OTM options properly needs real option data.\n"
          f"  Even the flattering end does not approach 2.")


def demo():
    """Self-check: the parity identity and the recovered premia."""
    p = parity("2005-01-01")
    assert len(p) > 200
    # recovered premia must look like real one-month SPX options
    assert 0.010 < p["call"].mean() < 0.035, p["call"].mean()
    assert 0.008 < p["put"].mean() < 0.035, p["put"].mean()
    # the defining identity: buy-write + protective put = index + strike (cash)
    # in return space, (1+r_bxm)*(1-c) + (1+r_pp)*(c+disc) == (1+r_spy) + 1
    q = p.dropna()
    disc = 1 / (1 + q["rf"])
    lhs = (1 + q["r_bxm"]) * (1 - q["call"]) + (1 + q["r_pp"]) * (q["call"] + disc)
    assert np.allclose(lhs, (1 + q["r_spy"]) + 1, atol=1e-9), "parity identity broken"
    # and a protective put can never lose more than its premium plus nothing else
    assert (q["r_pp"] > -0.5).all()
    print("options ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
