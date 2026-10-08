"""Per-symbol CSV cache. Fetch only bars after the last cached date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bars import clean


def merge(old: pd.DataFrame | None, new: pd.DataFrame | None) -> pd.DataFrame | None:
    """Union by date; on overlap the NEW row wins (vendors revise bars)."""
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    return clean(pd.concat([old, new]))


def get(provider, symbol: str, cache_dir) -> pd.DataFrame | None:
    f = Path(cache_dir) / f"{symbol.lstrip('^')}.csv"
    old = clean(pd.read_csv(f, index_col=0, parse_dates=True)) if f.exists() else None
    start = None if old is None else str((old.index[-1] + pd.Timedelta(days=1)).date())
    # ponytail: re-request the last cached day too if you need revision-safety; add if vendor revises.
    df = merge(old, provider.daily(symbol, start))
    if df is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f)
    return df
