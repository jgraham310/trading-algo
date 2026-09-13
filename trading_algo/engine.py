"""Daily long/flat backtester scored in EXCESS of T-bills.

Conventions, fixed once so every strategy in this repo is comparable:

* `w[t]` is the target weight decided from data up to and INCLUDING the close of
  bar `t`, and traded at that close. It earns bar `t+1`'s return.
* Returns are **excess of cash**: `exret = asset_return - rf`. Capital not in the
  market sits in T-bills and therefore contributes exactly zero. That is the
  whole point -- it stops a strategy from buying Sharpe by simply being flat.
* Costs are charged in basis points on |dw|, at the bar where the trade happens.
* No leverage: `w` is clipped into [0, 1] per asset and the row must sum to <=1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DATA = Path(__file__).resolve().parent.parent / "data"


PANEL = ["SPY", "QQQ", "IWM", "RSP", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU",
         "XLV", "XLY", "TLT", "IEF", "SHY", "GLD", "HYG", "LQD",
         "^VIX", "^VIX3M", "^IRX", "^TNX", "^FVX", "^GSPC", "^BXM", "^PUT",
         # ^SP500TR is the S&P 500 TOTAL RETURN index back to 1988 -- five years
         # more than SPY and the only assumption-free way to stress start dates
         # before 1993. It tracks SPY to 0.985 daily correlation, the 0.13%/yr
         # gap being SPY's expense ratio.
         "^SP500TR"]


def fetch(start="1985-01-01") -> None:
    """Refresh data/{close,open}.csv. ~13MB, gitignored, re-runnable."""
    import yfinance as yf

    DATA.mkdir(exist_ok=True)
    df = yf.download(PANEL, start=start, auto_adjust=True, progress=False, group_by="column")
    for field in ("Close", "Open"):
        d = df[field]
        d.index = d.index.tz_localize(None) if d.index.tz else d.index
        d.to_csv(DATA / f"{field.lower()}.csv")
    c = df["Close"]
    missing = [t for t in PANEL if t not in c or c[t].notna().sum() == 0]
    print(f"{len(c)} bars, {c.index[0].date()}..{c.index[-1].date()}"
          + (f"  MISSING: {missing}" if missing else ""))


def load(field: str = "close") -> pd.DataFrame:
    """Cached daily panel. `python -m trading_algo.engine --fetch` refreshes it."""
    path = DATA / f"{field}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run `python -m trading_algo.engine --fetch`")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def rf(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free from ^IRX (13w bill, quoted as an annual percent).

    Uses the PREVIOUS bar's quote: the rate you earn over bar t is the one you
    could observe before it started.
    """
    irx = load()["^IRX"].reindex(index).ffill().shift(1)
    return (1 + irx.clip(lower=0) / 100) ** (1 / TRADING_DAYS) - 1


def excess(tickers, start=None, end=None) -> pd.DataFrame:
    """Excess daily returns for `tickers`, on the bars where all of them trade."""
    px = load()[list(tickers)].dropna()
    if start is not None:
        px = px[px.index >= start]
    if end is not None:
        px = px[px.index <= end]
    r = px.pct_change()
    return r.sub(rf(r.index), axis=0).iloc[1:]


def run(w: pd.DataFrame | pd.Series, exret: pd.DataFrame | pd.Series,
        cost_bps: float = 2.0) -> pd.Series:
    """Net excess return series. `w[t]` is held over `exret[t+1]`."""
    w = pd.DataFrame(w).reindex(pd.DataFrame(exret).index)
    x = pd.DataFrame(exret)[w.columns] if w.columns.isin(pd.DataFrame(exret).columns).all() \
        else pd.DataFrame(exret).set_axis(w.columns, axis=1)
    if (w.fillna(0) < -1e-9).any().any():
        raise ValueError("short positions -- long/flat only")
    if (w.fillna(0).sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("gross exposure >1 -- no leverage")
    gross = (w.shift(1) * x).sum(axis=1)
    cost = w.diff().abs().sum(axis=1).shift(1) * cost_bps / 1e4
    out = (gross - cost)
    # the first bar has no prior weight to trade from and no position to earn on
    return out.iloc[1:]


def stats(net: pd.Series, w: pd.DataFrame | pd.Series | None = None,
          periods: int = TRADING_DAYS) -> dict:
    """Everything reported about a strategy, computed one way only."""
    r = pd.Series(net).dropna().to_numpy()
    n = len(r)
    sd = r.std(ddof=1)
    eq = np.cumprod(1 + r)
    m = {
        "years": n / periods,
        "ann_excess": float(eq[-1] ** (periods / n) - 1),
        "ann_vol": float(sd * np.sqrt(periods)),
        "sharpe": float("nan") if sd < 1e-12 else float(r.mean() / sd * np.sqrt(periods)),
        "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
        "skew": float(pd.Series(r).skew()),
        "worst_day": float(r.min()),
    }
    if w is not None:
        w = pd.DataFrame(w)
        m["exposure"] = float(w.sum(axis=1).mean())
        m["ann_turnover"] = float(w.diff().abs().sum(axis=1).sum() / len(w) * periods)
    return m


def sharpe_se(net: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Standard error of the annualized Sharpe (Lo 2002, iid approximation).

    ~sqrt((1 + S^2/2) / n) in per-period units. A 25-year daily Sharpe of 1.0
    carries an SE near 0.2 -- which is why 'beat 2' is a statement about ~4 SE,
    not about a decimal place.
    """
    r = pd.Series(net).dropna()
    s = r.mean() / r.std(ddof=1)
    return float(np.sqrt((1 + s * s / 2) / len(r)) * np.sqrt(periods))


def report(rows: dict[str, dict]) -> str:
    """One table, same columns, every strategy."""
    df = pd.DataFrame(rows).T
    cols = [c for c in ("years", "ann_excess", "ann_vol", "sharpe", "se", "max_dd",
                        "exposure", "ann_turnover", "skew") if c in df.columns]
    fmt = {"ann_excess": "{:+.2%}", "ann_vol": "{:.1%}", "sharpe": "{:.2f}", "se": "{:.2f}",
           "max_dd": "{:.1%}", "exposure": "{:.0%}", "ann_turnover": "{:.1f}x",
           "years": "{:.1f}", "skew": "{:+.2f}"}
    return df[cols].to_string(formatters={k: v.format for k, v in fmt.items() if k in cols})


def score(w, exret, cost_bps: float = 2.0) -> dict:
    net = run(w, exret, cost_bps)
    return {**stats(net, pd.DataFrame(w).reindex(net.index)), "se": sharpe_se(net)}


# ---------------------------------------------------------------- sessions ---
# A calendar day is two tradeable sessions: NIGHT (prior close -> open, ~17.5h)
# and DAY (open -> close, 6.5h). Splitting them doubles the decisions available
# to a long/flat strategy and -- more importantly -- lets it hold the session
# that actually pays. Costs are charged at every session boundary, so a
# night-only rule pays two crossings per night and cannot hide its turnover.

NIGHT_SHARE = 17.5 / 24


def sessions(ticker: str = "SPY", start=None, end=None) -> pd.DataFrame:
    """Excess returns per session: columns `night`, `day`, one row per bar.

    `night[t]` = open[t]/close[t-1] - 1, `day[t]` = close[t]/open[t] - 1, each
    less its share of that day's bill accrual. Compounding night and day
    reproduces the close-to-close return, which `demo()` asserts.
    """
    op, cl = load("open")[ticker], load("close")[ticker]
    df = pd.DataFrame({"o": op, "c": cl}).dropna()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    f = rf(df.index)
    out = pd.DataFrame({
        "night": df["o"] / df["c"].shift(1) - 1 - f * NIGHT_SHARE,
        "day": df["c"] / df["o"] - 1 - f * (1 - NIGHT_SHARE),
    }, index=df.index)
    return out.iloc[1:]


def run_sessions(w_night, w_day, ses: pd.DataFrame, cost_bps: float = 1.0) -> pd.Series:
    """Net excess per calendar day. Weights are decided at the session's START.

    `w_night[t]` is set at the close of t-1 (it is the position carried into the
    open of t); `w_day[t]` is set at the open of t. A cost of `cost_bps` is paid
    on every change of weight at each of the two boundaries, so holding a
    position straight through the open costs nothing and flipping costs twice.
    """
    n = pd.Series(w_night, index=ses.index).fillna(0.0).clip(0, 1)
    d = pd.Series(w_day, index=ses.index).fillna(0.0).clip(0, 1)
    gross = n * ses["night"] + d * ses["day"]
    at_open = (d - n).abs()                       # rebalance at the opening print
    at_close = (n.shift(-1) - d).abs().fillna(d)  # rebalance at the closing print
    return (gross - (at_open + at_close) * cost_bps / 1e4).iloc[1:]


def session_stats(w_night, w_day, ses, cost_bps: float = 1.0) -> dict:
    net = run_sessions(w_night, w_day, ses, cost_bps)
    n = pd.Series(w_night, index=ses.index).fillna(0).clip(0, 1)
    d = pd.Series(w_day, index=ses.index).fillna(0).clip(0, 1)
    turn = ((d - n).abs() + (n.shift(-1) - d).abs().fillna(d)).sum() / len(ses) * TRADING_DAYS
    return {**stats(net), "se": sharpe_se(net),
            "exposure": float((n * NIGHT_SHARE + d * (1 - NIGHT_SHARE)).mean()),
            "ann_turnover": float(turn)}


def demo():
    """Self-check: the conventions above actually hold."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    x = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1, 0, 0, 1, 1]}, index=idx)
    net = run(w, x, cost_bps=0)
    # w[0]=1 earns x[1]; w[2]=0 earns nothing of x[3]
    assert np.allclose(net.to_numpy(), [-0.02, 0.03, 0.0, 0.0, 0.02]), net.to_numpy()
    # a costed round trip loses exactly 2 * bps
    assert abs(run(w, x, cost_bps=10).sum() - (net.sum() - 2 * 10 / 1e4)) < 1e-12

    # LEAKAGE: a weight built from tomorrow's return must be the ONLY thing that
    # scores perfectly. If a 1-bar shift does not destroy the edge, the harness
    # is reading right.
    oracle = pd.DataFrame({"A": (x["A"] > 0).astype(float).shift(-1).fillna(0)})
    assert np.isclose(run(oracle, x, 0).sum(), x["A"][x["A"] > 0].iloc[1:].sum())
    assert run(oracle.shift(1), x, 0).sum() < run(oracle, x, 0).sum()

    for bad in (pd.DataFrame({"A": [-1.0] * 6}, index=idx),
                pd.DataFrame({"A": [2.0] * 6}, index=idx)):
        try:
            run(bad, x)
        except ValueError:
            continue
        raise AssertionError("accepted a short or a levered weight")
    # sessions compound back to the close-to-close return
    ses = sessions("SPY", start="2015-01-01")
    cl = load("close")["SPY"].reindex(ses.index)
    f = rf(ses.index)
    recon = (1 + ses["night"] + f * NIGHT_SHARE) * (1 + ses["day"] + f * (1 - NIGHT_SHARE)) - 1
    assert np.allclose(recon, cl / cl.shift(1) - 1, atol=1e-10, equal_nan=True), "session split lost return"
    # holding through the open is free; flipping every session is not
    one = pd.Series(1.0, index=ses.index)
    # the only cost a hold-through pays is the single liquidation on the last bar
    assert np.isclose(run_sessions(one, one, ses, 0).sum() - run_sessions(one, one, ses, 10).sum(), 10 / 1e4)
    assert run_sessions(one, one * 0, ses, 10).sum() < run_sessions(one, one * 0, ses, 0).sum()
    print("engine ok")


if __name__ == "__main__":
    import sys
    fetch() if "--fetch" in sys.argv else demo()
