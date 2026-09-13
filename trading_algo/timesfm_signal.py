"""TimesFM 3.0 as the forecasting engine behind a trading signal.

Model: https://github.com/google-research/timesfm/releases/tag/v3.0.0

NOTE: the TimesFM 3.0 *weights* ship under timesfm-non-commercial-license-v1.0
(non-commercial, non-production only). Set TIMESFM_CHECKPOINT to an Apache-2.0
2.x checkpoint if you intend to trade real money with this.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CHECKPOINT = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-3.0-pytorch")

# TimesFM returns 9 quantiles, 0.1 .. 0.9
QUANTILES = np.round(np.arange(0.1, 0.95, 0.1), 1)

_forecaster = None


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model():
    """Load once. The checkpoint download is ~GBs, so never do this per call."""
    global _forecaster
    if _forecaster is None:
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        _forecaster = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=CHECKPOINT,
                per_core_batch_size=32,
                device=_device(),
            )
        )
    return _forecaster


@dataclass
class Forecast:
    last: float  # last observed close, i.e. the decision-bar price
    path: np.ndarray  # (horizon,) median forecast, price space
    quantiles: np.ndarray  # (horizon, 9) price space, QUANTILES

    @property
    def expected_return(self) -> float:
        return float(self.path[-1] / self.last - 1.0)

    def quantile_return(self, q: float) -> float:
        i = int(np.argmin(np.abs(QUANTILES - q)))
        return float(self.quantiles[-1, i] / self.last - 1.0)


def forecast(prices, horizon: int = 5, context: int = 2048) -> Forecast:
    """Forecast `horizon` bars ahead from a 1D price series.

    `prices` must end at the bar you are deciding on -- element [-1] is the last
    CLOSED bar. The forecast covers bars [-1]+1 .. [-1]+horizon.
    """
    x = np.asarray(prices, dtype=np.float32).ravel()
    x = x[np.isfinite(x)][-context:]
    if len(x) < 32:
        raise ValueError(f"need >=32 finite bars of context, got {len(x)}")

    out = next(iter(_model().predict_batch([x], horizon=horizon, return_quantiles=True)))
    return Forecast(last=float(x[-1]), path=np.asarray(out.forecast), quantiles=np.asarray(out.quantiles))


def position(er: float, lo: float, hi: float, scale: float = 0.02) -> float:
    """Map an expected return + its confidence band to a position in [-1, 1].

    `lo`/`hi` are the pessimistic/optimistic quantile returns. A side is only
    taken when the band's near side still clears flat -- otherwise the model is
    not saying anything a coin flip wouldn't. `scale` is the expected return
    that maps to full size; tune it per instrument.
    """
    if er > 0 and lo <= 0:
        return 0.0
    if er < 0 and hi >= 0:
        return 0.0
    return float(np.clip(er / scale, -1.0, 1.0))


def signal(prices, horizon: int = 5, conf: float = 0.3, scale: float = 0.02, **kw) -> float:
    """Target position in [-1, 1] for the bar AFTER `prices[-1]`."""
    f = kw.pop("_forecast", None)
    if f is None:
        f = forecast(prices, horizon=horizon, **kw)
    return position(f.expected_return, f.quantile_return(conf), f.quantile_return(1 - conf), scale)


def load_closes(ticker: str, period: str = "2y", interval: str = "1d", start=None, end=None):
    """Adjusted closes. auto_adjust is explicit: over decades, reinvested
    dividends are the difference between a right answer and a wrong one."""
    import yfinance as yf

    kw = {"start": start, "end": end} if start else {"period": period}
    df = yf.Ticker(ticker).history(interval=interval, auto_adjust=True, **kw)
    s = df["Close"].dropna()
    s.index = s.index.tz_localize(None) if s.index.tz else s.index
    return s.rename(ticker)


if __name__ == "__main__":
    import sys

    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    closes = load_closes(tk)
    f = forecast(closes.values)
    print(f"{tk}  last={f.last:.2f}  5d={f.path[-1]:.2f}  "
          f"er={f.expected_return:+.2%}  q30={f.quantile_return(0.3):+.2%}  "
          f"q70={f.quantile_return(0.7):+.2%}")
    print(f"position={signal(closes.values, _forecast=f):+.2f}")
