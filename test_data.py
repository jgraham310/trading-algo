import tempfile

import pandas as pd

from trading_algo.data import cache
from trading_algo.data.bars import clean
from trading_algo.data.providers.csv import CsvProvider
from trading_algo.data.providers.ibkr import IbkrProvider
from trading_algo.data.weekly import weekly


def bars(dates, closes):
    c = pd.Series(closes, index=pd.to_datetime(dates), dtype=float)
    return pd.DataFrame({"open": c - 1, "high": c + 2, "low": c - 2, "close": c, "volume": 100.0})


# --- weekly resample: hand-computed. 2026-09-14 is a Monday.
d = bars(["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
          "2026-09-21", "2026-09-22", "2026-09-23"], [10, 11, 12, 13, 14, 20, 21, 22])
w = weekly(d)
assert list(w.index.strftime("%Y-%m-%d")) == ["2026-09-18", "2026-09-25"]   # Friday labels
r = w.iloc[0]
assert (r.open, r.high, r.low, r.close, r.volume) == (9, 16, 8, 14, 500)
assert list(w.partial) == [False, True]                    # last week ends Wed: partial
assert weekly(d, asof="2026-09-25").partial.tolist() == [False, False]

# holiday Friday (no Friday bar) in a past week is complete, not partial
h = bars(["2026-04-01", "2026-04-02", "2026-04-06"], [1, 2, 3])
assert weekly(h).partial.tolist() == [False, True]   # Good Friday week complete, Apr 6 week not
assert weekly(h, asof="2026-04-10").partial.tolist() == [False, False]

# --- cache merge: union, new wins on overlap, sorted, no dups
old = bars(["2026-09-14", "2026-09-15"], [10, 11])
new = bars(["2026-09-15", "2026-09-16"], [99, 12])
m = cache.merge(old, new)
assert m.close.tolist() == [10, 99, 12] and m.index.is_monotonic_increasing
assert cache.merge(None, new) is new and cache.merge(old, None) is old

# --- cache.get only asks the provider for bars after the last cached date
class Spy:
    def __init__(self, df): self.df, self.starts = df, []
    def daily(self, symbol, start=None):
        self.starts.append(start)
        return self.df if start is None else self.df.loc[start:]

full = bars(["2026-09-14", "2026-09-15", "2026-09-16"], [10, 11, 12])
with tempfile.TemporaryDirectory() as t:
    p = Spy(full.iloc[:2])
    cache.get(p, "AAA", t)
    p.df = full
    out = cache.get(p, "AAA", t)
    assert p.starts == [None, "2026-09-16"] and out.close.tolist() == [10, 11, 12]

# --- csv provider: present and missing ("unknown") series
with tempfile.TemporaryDirectory() as t:
    full.to_csv(f"{t}/SPY.csv")
    c = CsvProvider(t)
    assert c.daily("SPY").close.tolist() == [10, 11, 12]
    assert c.daily("NYAD") is None

# --- ibkr provider refuses live ports (no connection is made)
for port in (4001, 7496, 7497 + 1):
    try:
        IbkrProvider("127.0.0.1", port, 1)
        raise AssertionError("accepted a non-paper port")
    except ValueError:
        pass
IbkrProvider("127.0.0.1", 4002, 1)
print("data ok")
