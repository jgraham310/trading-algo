"""IBKR historical-bars provider. PAPER GATEWAY ONLY, market data only.

Never calls account, position or order APIs. Refuses any port that is not a
paper port. Not exercised by tests; see test_data.py for the offline path.
"""

from __future__ import annotations

import time

import pandas as pd

from ..bars import clean

PAPER_PORTS = {4002, 7497}


class IbkrProvider:
    def __init__(self, host: str, port: int, client_id: int, min_interval_s: float = 11,
                 max_retries: int = 3):
        if port not in PAPER_PORTS:
            raise ValueError(f"refusing port {port}: paper ports only {sorted(PAPER_PORTS)}")
        self.host, self.port, self.client_id = host, port, client_id
        self.gap, self.retries, self._last, self._ib = min_interval_s, max_retries, 0.0, None

    @classmethod
    def from_config(cls, cfg: dict):
        c = cfg["ibkr"]
        return cls(c["host"], c["port"], c["client_id"], c["min_interval_s"], c["max_retries"])

    def _connect(self):
        if self._ib is None:
            from ib_async import IB  # lazy: not a project dependency yet
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id, readonly=True)
        return self._ib

    def _throttle(self):
        wait = self._last + self.gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def daily(self, symbol, start=None):
        from ib_async import Index, Stock
        ib = self._connect()
        contract = Index(symbol.lstrip("^"), "CBOE") if symbol.startswith("^") else Stock(symbol, "SMART", "USD")
        for attempt in range(self.retries):
            self._throttle()
            try:
                bars = ib.reqHistoricalData(contract, "", "10 Y" if start is None else "2 Y",
                                            "1 day", "TRADES",
                                            useRTH=True, formatDate=1)
                break
            except Exception:
                time.sleep(self.gap * 2 ** attempt)   # backoff; never loops forever
        else:
            return None
        if not bars:
            return None   # unavailable -> "unknown", not faked
        df = clean(pd.DataFrame([vars(b) for b in bars]).set_index("date"))
        return df if start is None else df.loc[start:]
