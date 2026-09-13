"""Walk-forward backtest for the TimesFM signal.

Two stages on purpose: `walk_forward` is the expensive one (one model call per
run), `evaluate` is free. So you infer once and sweep thresholds for nothing.

READ THE CAVEATS in `report()` before you believe any number this prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sources import _naive
from .timesfm_signal import QUANTILES, _model, position

TRADING_DAYS = 252


def walk_forward(prices, horizon: int = 5, context: int = 512, batch: int = 64,
                 past: pd.DataFrame | None = None,
                 future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast at every bar using only bars up to and including that bar.

    `past`   -- covariates known at each bar (other closes, VIX, lagged macro).
                Must be indexed like `prices`.
    `future` -- covariates known *in advance* over the forecast window too, so
                indexed `horizon` bars LONGER than `prices` (see
                `sources.calendar` / `sources.future_index`). If you cannot say
                why you know these values ahead of time, they belong in `past`.
    """
    s = pd.Series(prices).dropna().astype("float32")
    px = s.to_numpy()
    n = len(px)
    if n < context + horizon + 2:
        raise ValueError(f"need >={context + horizon + 2} bars, got {n}")

    pc = _channels(past, s.index, n) if past is not None else None
    fc = _channels(future, s.index, n + horizon) if future is not None else None

    # t is the decision bar. Context is px[t-context+1 : t+1] -- inclusive of t,
    # exclusive of everything after. Stop early enough to have a realized
    # px[t+horizon] to score against.
    ts = np.arange(context - 1, n - horizon)
    contexts = [px[t - context + 1 : t + 1] for t in ts]
    assert all(c[-1] == px[t] for c, t in zip(contexts, ts))  # no off-by-one

    rows = []
    for i in range(0, len(contexts), batch):
        sl = slice(i, i + batch)
        kw = {}
        if pc is not None:
            kw["past_only_covariates"] = [pc[:, t - context + 1 : t + 1] for t in ts[sl]]
        if fc is not None:
            # context window PLUS the horizon -- legitimate only because these
            # are known in advance by construction.
            kw["past_future_covariates"] = [fc[:, t - context + 1 : t + 1 + horizon] for t in ts[sl]]
        for out in _model().predict_batch(contexts[sl], horizon=horizon, return_quantiles=True, **kw):
            q = np.asarray(out.quantiles)
            q = q[-1] if q.ndim == 2 else q[0, -1]
            f = np.asarray(out.forecast).ravel()
            rows.append((float(f[-1]), *np.asarray(q).ravel().tolist()))

    df = pd.DataFrame(rows, columns=["point", *[f"q{int(q * 100)}" for q in QUANTILES]])
    df.index = _naive(s.index[ts])  # tz-naive so it round-trips through CSV
    df["last"] = px[ts]
    df["actual"] = px[ts + horizon]  # realized price at the forecast target
    df["next_ret"] = px[ts + 1] / px[ts] - 1.0  # what a position taken at t earns
    return df


def _channels(cov: pd.DataFrame, index: pd.Index, want: int) -> np.ndarray:
    """(n_channels, want) float32, verified to start on the price index."""
    cov = pd.DataFrame(cov)
    if len(cov) != want:
        raise ValueError(f"covariates have {len(cov)} rows, need {want}")
    if not cov.index[: len(index)].equals(pd.Index(index)):
        raise ValueError("covariate index must match the price index bar for bar")
    a = cov.to_numpy(dtype="float32").T
    if not np.isfinite(a).all():
        raise ValueError("covariates contain NaN/inf -- ffill or drop the warmup "
                         "rows before passing them in")
    return a


def _stats(pos: np.ndarray, next_ret: np.ndarray, cost_bps: float, prefix: str = "",
           periods: int = TRADING_DAYS) -> dict:
    """Score a position series. `pos[i]` is held over `next_ret[i]`."""
    turnover = np.abs(np.diff(pos, prepend=0.0))
    gross = pos * next_ret
    net = gross - turnover * cost_bps / 1e4
    n = len(net)
    return {
        f"{prefix}time_in_mkt": float((pos != 0).mean()),
        f"{prefix}trades": int((turnover > 1e-9).sum()),
        f"{prefix}ann_turnover": float(turnover.sum() / n * periods),
        f"{prefix}cost_drag": float(_ann(gross, n, periods) - _ann(net, n, periods)),
        f"{prefix}ann_return": _ann(net, n, periods),
        f"{prefix}sharpe": _sharpe(net, periods),
        f"{prefix}max_dd": _max_dd(net),
        f"{prefix}hit_rate": float((net[pos != 0] > 0).mean()) if (pos != 0).any() else float("nan"),
    }


def ma_positions(prices, window: int = 200) -> pd.Series:
    """Classic trend filter: long while the close is above its trailing MA.

    The MA at bar t uses closes up to and INCLUDING t, and the position it
    implies is held over bar t+1 -- same convention as `signal`, so the two are
    directly comparable.
    """
    s = pd.Series(prices).dropna()
    ma = s.rolling(window).mean()
    # NaN, not 0.0, during warmup: `close > NaN` is False, which would silently
    # score an un-warmed MA as "flat" instead of refusing to score it.
    return (s > ma).astype(float).where(ma.notna())


def evaluate(df: pd.DataFrame, conf: float = 0.3, scale: float = 0.02, cost_bps: float = 1.0,
             prices=None, ma_window: int = 200) -> dict:
    """Score a walk_forward frame. `cost_bps` is charged on position *changes*.

    Pass `prices` (the full close series, long enough to warm up the MA) to also
    score the moving-average benchmark over the identical bars.
    """
    lo, hi = f"q{int(conf * 100)}", f"q{int((1 - conf) * 100)}"
    er = df["point"] / df["last"] - 1.0
    pos = np.array([
        position(e, l, h, scale)
        for e, l, h in zip(er, df[lo] / df["last"] - 1.0, df[hi] / df["last"] - 1.0)
    ])
    nxt = df["next_ret"].to_numpy()

    # Forecast skill vs. the only baseline that matters for prices: last value.
    err_model = np.abs(df["point"] - df["actual"])
    err_naive = np.abs(df["last"] - df["actual"])
    moved = df["actual"] != df["last"]
    dir_hit = (np.sign(df["point"] - df["last"]) == np.sign(df["actual"] - df["last"]))[moved]

    m = {
        "bars": len(df),
        "mae_ratio": float(err_model.mean() / err_naive.mean()),  # <1.0 = beats random walk
        "dir_acc": float(dir_hit.mean()),
        # On a drifting series "always up" is already a decent guess. Beating
        # this base rate -- not beating 50% -- is what counts as skill.
        "up_rate": float((df["actual"] > df["last"])[moved].mean()),
        **_stats(pos, nxt, cost_bps),
        **_stats(np.ones(len(df)), nxt, cost_bps, prefix="bh_"),
    }

    if prices is not None:
        ma = ma_positions(prices, ma_window)
        ma.index = _naive(ma.index)
        ma = ma.reindex(_naive(df.index))
        if ma.isna().any():
            raise ValueError(f"MA{ma_window} needs {ma_window} bars of warmup before "
                             f"{df.index[0].date()} -- pass a longer price history")
        m.update(_stats(ma.to_numpy(), nxt, cost_bps, prefix="ma_"))
        m["ma_window"] = ma_window
    return m


def _ann(r, n, periods: int = TRADING_DAYS) -> float:
    return float(np.prod(1 + r) ** (periods / n) - 1)


def _sharpe(r, periods: int = TRADING_DAYS) -> float:
    sd = np.std(r, ddof=1)
    if sd < 1e-12:  # degenerate / all-flat: Sharpe is undefined, not astronomical
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods))


def _max_dd(r) -> float:
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def report(m: dict) -> str:
    has_ma = "ma_sharpe" in m
    col = (lambda k: f"{m['ma_' + k]:>9}") if has_ma else (lambda k: "")

    def row(label, key, fmt):
        cells = [format(m[key], fmt), format(m["ma_" + key], fmt) if has_ma else None,
                 format(m["bh_" + key], fmt)]
        return f"  {label:<13}" + "".join(f"{c:>12}" for c in cells if c is not None)

    head = f"  {'':<13}{'TIMESFM':>12}" + (f"{'MA' + str(m['ma_window']):>12}" if has_ma else "") + f"{'BUY&HOLD':>12}"
    return f"""\
bars={m['bars']}  time_in_mkt={m['time_in_mkt']:.0%}  trades={m['trades']}  turnover={m['ann_turnover']:.1f}x/yr

FORECAST SKILL   (does the model beat a random walk at all?)
  MAE vs naive   {m['mae_ratio']:.3f}      <1.00 = better than "price stays put"
  direction acc  {m['dir_acc']:.1%}      vs {m['up_rate']:.1%} for always-guessing-up

{head}
{row('ann return', 'ann_return', '+.1%')}
{row('sharpe', 'sharpe', '.2f')}
{row('max drawdown', 'max_dd', '+.1%')}
{row('time in mkt', 'time_in_mkt', '.0%')}
{row('trades', 'trades', 'd')}
{row('hit rate', 'hit_rate', '.1%')}

CAVEATS -- this is a screen, not evidence:
  * TimesFM 3.0 was pretrained on data through ~2026. Backtesting it on a
    liquid, widely-published series is CONTAMINATED: the model may have seen
    these exact bars. Treat in-sample-of-pretraining results as an upper bound.
  * One instrument, one period, one parameter set. Sweep, and check other
    tickers, before reading anything into a single number.
  * No slippage model, no borrow cost, no fills. Executes at the close, always.
"""


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from .timesfm_signal import load_closes

    tk = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = sys.argv[2] if len(sys.argv) > 2 else None  # e.g. 2000-01-01
    warm = "1998-01-01" if start else None  # context+MA warmup before `start`

    cache = Path(f"wf_{tk}{'_' + start[:4] if start else ''}.csv")
    px = load_closes(tk, period="10y", start=warm)
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        # utc=True: a cache written across a DST change has mixed offsets
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
        print(f"[cached {cache}]")
    else:
        df = walk_forward(px if start else load_closes(tk, period="5y"))
        df.to_csv(cache)
    if start:
        df = df[df.index >= start]
    print(f"\n=== {tk}  {df.index[0].date()} .. {df.index[-1].date()} ===")
    print(report(evaluate(df, prices=px)))
