"""Offline provider: `<dir>/<SYMBOL>.csv` with date,open,high,low,close,volume."""

from pathlib import Path

import pandas as pd

from ..bars import clean


class CsvProvider:
    def __init__(self, directory):
        self.dir = Path(directory)

    def daily(self, symbol, start=None):
        f = self.dir / f"{symbol.lstrip('^')}.csv"
        if not f.exists():
            return None
        df = clean(pd.read_csv(f, index_col=0, parse_dates=True))
        return df if start is None else df.loc[start:]
