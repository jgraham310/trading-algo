"""Align arbitrary data sources onto the target's bar grid, without leakage.

The only rule that matters: a value may appear at bar t ONLY if it was knowable
at bar t. Every source therefore carries a publication lag. Same-bar market data
(another ticker's close, VIX) has lag 0. Anything released after the period it
describes -- macro prints, earnings, revisions -- has a positive lag, and
getting it wrong manufactures edge that does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align(index: pd.DatetimeIndex, ffill_limit: int | None = None, **sources) -> pd.DataFrame:
    """Put every source on `index`, shifted by its publication lag.

    Each source is `name=series` (lag 0) or `name=(series, lag_days)`.
    Each bar takes the last value published at or before it -- a backward asof
    join, which cannot read right. `ffill_limit` is counted in BARS OF `index`
    (not in source rows): a value survives the bar it appears on plus that many
    more, then goes NaN rather than growing indefinitely stale.

    >>> idx = pd.bdate_range("2024-01-01", periods=5)
    >>> cpi = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))
    >>> align(idx, cpi=(cpi, 2))["cpi"].tolist()   # not visible until +2 days
    [nan, nan, 1.0, 1.0, 1.0]
    >>> align(idx, cpi=(cpi, 2), ffill_limit=1)["cpi"].tolist()   # then expires
    [nan, nan, 1.0, 1.0, nan]
    """
    index = _naive(index)
    tgt = pd.DataFrame({"_t": index})
    out = {}
    for name, spec in sources.items():
        s, lag = spec if isinstance(spec, tuple) else (spec, 0)
        s = pd.Series(s).dropna().sort_index()
        src = pd.DataFrame({"_t": _naive(s.index) + pd.Timedelta(days=lag),
                            "v": s.to_numpy()}).sort_values("_t")
        src["pub"] = src["_t"]
        m = pd.merge_asof(tgt, src, on="_t", direction="backward")
        if ffill_limit is not None:
            # age in target bars since this value first became visible
            m.loc[m.groupby("pub").cumcount() > ffill_limit, "v"] = np.nan
        out[name] = m["v"].to_numpy()
    return pd.DataFrame(out, index=index)


def _naive(index) -> pd.DatetimeIndex:
    """tz-naive, nanosecond resolution -- merge_asof requires both sides match."""
    idx = pd.DatetimeIndex(index)
    return (idx.tz_localize(None) if idx.tz else idx).as_unit("ns").normalize()


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features genuinely known in advance -- the only safe *future* covariates.

    Anything else you can name is a forecast of the future, not the future.
    """
    index = _naive(index)
    dow, dom = index.dayofweek.to_numpy(float), index.day.to_numpy(float)
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 5),
            "dow_cos": np.cos(2 * np.pi * dow / 5),
            "mon_sin": np.sin(2 * np.pi * dom / 31),
            "mon_cos": np.cos(2 * np.pi * dom / 31),
        },
        index=index,
    )


def future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Extend a bar index `horizon` business days past its end."""
    index = _naive(index)
    return index.append(pd.bdate_range(index[-1], periods=horizon + 1)[1:])


def closes(*tickers: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Same-bar market data -- lag 0, since the close is known at the close."""
    import yfinance as yf

    df = yf.download(list(tickers), period=period, interval=interval,
                     auto_adjust=True, progress=False)["Close"]
    return df.tz_localize(None) if df.index.tz else df
