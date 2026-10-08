"""Bar access. A provider returns daily OHLCV or None when it has no such series.

None is the contract for "unknown": rules that need that series must report
"unknown" and drop their weight; nothing here fabricates data.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


class Provider(Protocol):
    def daily(self, symbol: str, start: str | None = None) -> pd.DataFrame | None: ...


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, DatetimeIndex, sorted, no duplicate dates."""
    df = df.rename(columns=str.lower)[COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()
