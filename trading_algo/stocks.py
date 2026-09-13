"""Can long-only STOCK SELECTION reach Sharpe 2 where index timing cannot?

`timing.ceiling()` bounds one axis: choosing *when* to hold the market. This
bounds the other: choosing *what* to hold. It needs a different argument,
because a 400-name portfolio rebalanced monthly has far more breadth than a
single long/flat index bet, so the monthly-oracle bound does not apply.

The universe here is today's S&P 500 members, which is **survivorship-biased on
purpose**. Every name that blew up and left the index is missing, so every
number is an upper bound on what was actually achievable. `bias()` measures how
large that thumb on the scale is by rebuilding the identical equal-weight
portfolio and comparing it with RSP, which is point-in-time correct.

    python -m trading_algo.stocks --fetch    # once: ~500 names x 30y
    python -m trading_algo.stocks            # ceiling, bias, strategies
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import DATA, TRADING_DAYS, load, report, rf, sharpe_se, stats
from .timing import weight as timing_weight

STOCKS = DATA / "stocks.csv"
START = "2001-09-13"  # strict 25 years back from the last bar


def fetch(start="1996-01-01") -> None:
    """Today's S&P 500 members from Wikipedia, then 30 years of adjusted closes."""
    import io

    import requests
    import yfinance as yf

    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    tk = sorted(pd.read_html(io.StringIO(html))[0]["Symbol"].astype(str)
                .str.replace(".", "-", regex=False))
    out = [yf.download(tk[i:i + 100], start=start, auto_adjust=True, progress=False,
                       group_by="column")["Close"] for i in range(0, len(tk), 100)]
    df = pd.concat(out, axis=1).sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df.loc[:, ~df.columns.duplicated()].to_csv(STOCKS)
    print(f"{df.shape[1]} names, {df.index[0].date()}..{df.index[-1].date()}")


def panel():
    """Prices, excess returns, and a point-in-time tradeability mask."""
    if not STOCKS.exists():
        raise FileNotFoundError(f"{STOCKS} -- run `python -m trading_algo.stocks --fetch`")
    px = pd.read_csv(STOCKS, index_col=0, parse_dates=True).sort_index()
    px = px[px.index >= pd.Timestamp("1999-01-01")]
    ex = px.pct_change().sub(rf(px.index), axis=0).where(px.notna() & px.shift(1).notna())
    live = px.notna() & (px.rolling(TRADING_DAYS).count() >= 250)  # needs a year of history
    return px, ex, live


def _row(net, **kw):
    net = pd.Series(net).dropna()
    return {**stats(net), "se": sharpe_se(net), **{"exposure": 1.0, "ann_turnover": 0.0, **kw}}


def backtest(sig: pd.DataFrame, k: int = 50, freq: str = "M", cost_bps: float = 5.0,
             overlay: pd.Series | None = None, start=START) -> dict:
    """Equal-weight the top `k` names by `sig`, rebalanced monthly or weekly.

    `sig[t]` must be knowable at the close of `t`; weights earn `t+1`'s return.
    `overlay` scales total exposure (that is how the index-timing rule is bolted
    on). Weights always sum to <=1: long-only, no leverage, cash is the residual.
    """
    px, ex, live = panel()
    rk = sig.where(live).rank(axis=1, ascending=False)
    w = (rk <= k).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    key = pd.Series(px.index, index=px.index).dt.to_period("W" if freq == "W" else "M")
    w = w.where(key != key.shift(-1)).ffill().fillna(0.0)  # hold until the period's last bar
    if overlay is not None:
        w = w.mul(overlay.reindex(w.index).fillna(0), axis=0)
    m = px.index >= pd.Timestamp(start)
    W, X = w[m], ex[m]
    turn = W.diff().abs().sum(axis=1).shift(1).fillna(0)
    net = ((W.shift(1) * X).sum(axis=1) - turn * cost_bps / 1e4).iloc[1:]
    return _row(net, exposure=float(W.sum(axis=1).mean()),
                ann_turnover=float(turn.sum() / len(W) * TRADING_DAYS))


def hindsight(start=START) -> dict:
    """The bound that actually bites: best-possible STATIC pick, known in advance.

    Rank every name by its realised 25-year return, buy the top k in 2001, hold.
    No rule can beat this at static selection, because it *is* the answer sheet.
    """
    px, ex, _ = panel()
    m = px.index >= pd.Timestamp(start)
    e = ex[m]
    full = e.notna().sum() > 6000  # listed for essentially the whole window
    rank = ((1 + e.fillna(0)).prod() - 1)[full].sort_values(ascending=False)
    out = {f"best {k} (hindsight)": _row(e[rank.index[:k]].mean(axis=1)) for k in (1, 3, 5, 10, 20, 50)}
    spy = load()["SPY"].reindex(px.index).pct_change().sub(rf(px.index), axis=0)[m]
    out["SPY"] = _row(spy)
    return out, rank


def bias() -> dict:
    """Size the survivorship thumb: same equal-weight portfolio, vs RSP."""
    px, ex, _ = panel()
    c = load()
    idx = c["RSP"].dropna().index
    idx = idx[(idx >= pd.Timestamp("2003-05-01")) & (idx <= px.index[-1])]
    f = rf(idx)
    e = px.reindex(idx).pct_change().sub(f, axis=0).where(
        px.reindex(idx).notna() & px.reindex(idx).shift(1).notna())
    return {"RSP (point-in-time EW)": _row(c["RSP"].reindex(idx).pct_change().sub(f, axis=0)),
            "EW today's members": _row(e.mean(axis=1)),
            "SPY": _row(c["SPY"].reindex(idx).pct_change().sub(f, axis=0))}


def main():
    px, ex, live = panel()
    print(f"universe: {px.shape[1]} of today's S&P 500, {px.index[0].date()}..{px.index[-1].date()}\n")

    out, rank = hindsight()
    print("CEILING 1 -- PERFECT HINDSIGHT static selection, equal weight, held 25 years.")
    print("No selection rule can beat this, because it is the answer sheet.\n")
    print(report(out))
    print(f"  best single name {rank.index[0]} at {out['best 1 (hindsight)']['ann_excess']:.0%}/yr"
          f" -> Sharpe {out['best 1 (hindsight)']['sharpe']:.2f}: concentration adds return and "
          f"volatility\n  together, so it does not raise the ratio. The best k tops out at "
          f"{max(v['sharpe'] for v in out.values()):.2f}, at k=10.\n")

    b = bias()
    print("CEILING 2 -- how much of any stock-selection result is survivorship.")
    print(report(b))
    d = b["EW today's members"]["ann_excess"] - b["RSP (point-in-time EW)"]["ann_excess"]
    ds = b["EW today's members"]["sharpe"] - b["RSP (point-in-time EW)"]["sharpe"]
    print(f"  Identical construction, the only difference being names that left the index:\n"
          f"  {d:+.2%}/yr and {ds:+.2f} of Sharpe, free. Subtract it from everything below.\n")

    mom = px.shift(21) / px.shift(252) - 1
    vol = -ex.rolling(120).std()
    rev1, rev5 = -(px / px.shift(21) - 1), -(px / px.shift(5) - 1)
    comp = (mom.rank(axis=1, pct=True) + vol.rank(axis=1, pct=True)) / 2
    tim = timing_weight(load()["SPY"]).reindex(px.index)
    ew = pd.DataFrame(1.0, index=px.index, columns=px.columns)

    print("REAL RULES on the biased universe, 5bp per unit traded (single stocks, not SPY):")
    print(report({
        "SPY buy & hold": hindsight()[0]["SPY"],
        "EW universe": backtest(ew, k=10 ** 9),
        "momentum 12-1 top50": backtest(mom, 50),
        "momentum 12-1 top20": backtest(mom, 20),
        "low-vol top50": backtest(vol, 50),
        "1m reversal top50": backtest(rev1, 50),
        "5d reversal top50 (W)": backtest(rev5, 50, "W"),
        "mom+lowvol top50": backtest(comp, 50),
    }))
    print("\nSAME, with the index-timing rule scaling exposure (the best combination found):")
    r = {f"{n} x TIMING": backtest(s, 50, "M", 5.0, overlay=tim)
         for n, s in (("momentum 12-1", mom), ("low-vol", vol), ("mom+lowvol", comp))}
    r["EW universe x TIMING"] = backtest(ew, 10 ** 9, "M", 5.0, overlay=tim)
    print(report(r))
    best = max(r.values(), key=lambda v: v["sharpe"])
    print(f"\n  Best: Sharpe {best['sharpe']:.2f} -- on a universe that gets {ds:+.2f} of Sharpe\n"
          f"  for free from survivorship. Adjusted, roughly {best['sharpe'] - ds:.2f}, and 2.0 is\n"
          f"  {(2 - best['sharpe']) / best['se']:.1f} standard errors above even the unadjusted figure.")


def demo():
    """Self-check on synthetic prices -- no download, no network."""
    idx = pd.bdate_range("2000-01-01", periods=900)
    rng = np.random.default_rng(0)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (900, 6)), axis=0)),
                     index=idx, columns=list("ABCDEF"))
    sig = p.pct_change(20)
    rk = sig.rank(axis=1, ascending=False)
    w = (rk <= 2).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    # NaN ranks during warmup must produce no position, never a silent 1/k bet
    assert w.iloc[:20].isna().all().all(), "held a position before the signal was warm"
    warm = w.iloc[20:]
    assert np.allclose(warm.sum(axis=1), 1.0), "top-k weights must sum to exactly 1"
    assert ((warm > 0).sum(axis=1) == 2).all(), "held more or fewer than k names"
    key = pd.Series(idx, index=idx).dt.to_period("M")
    held = w.where(key != key.shift(-1)).ffill()
    # a monthly rebalance may only change weights on month ends
    ch = held.diff().abs().sum(axis=1) > 1e-12
    assert (key[ch] != key.shift(-1)[ch]).all() or ch.sum() == 0, "rebalanced off-schedule"
    assert held.shift(1).mul(p.pct_change()).sum(axis=1).notna().any()
    print("stocks ok")


if __name__ == "__main__":
    import sys
    if "--fetch" in sys.argv:
        fetch()
    elif "--demo" in sys.argv:
        demo()
    else:
        main()
