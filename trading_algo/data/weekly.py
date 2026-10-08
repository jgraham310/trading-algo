"""Weekly bars from daily, Friday-labelled, with an explicit partial flag."""

from __future__ import annotations

import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def weekly(daily: pd.DataFrame, asof=None) -> pd.DataFrame:
    """Weeks end on Friday. `partial` is True while the week's Friday is after `asof`
    (default: last daily bar), so a holiday-Friday week in the past is complete but
    a current week is not. Decision code must drop partial rows."""
    asof = pd.Timestamp(asof) if asof is not None else daily.index[-1]
    w = daily.resample("W-FRI").agg(AGG).dropna(subset=["close"])
    w["partial"] = w.index > asof
    return w
