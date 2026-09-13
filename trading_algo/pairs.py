"""Co-integration pairs signal: rolling hedge ratio, z-score, position state.

All three numbers on a given row use only data up to and including that bar's
close. `Position` is `Signal` shifted one bar, so it is what you actually hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_pairs_signals(
    series_a: pd.Series,
    series_b: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float | None = 4.0,
    window: int = 30,
    z_window: int | str | None = None,
    max_half_life: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Dynamic hedge ratios, spreads, z-scores and positions for two price series.

    Returns (dataframe, rolling hedge ratio). Warmup is ~2 * window bars: one
    window for the hedge ratio, another for the z-score of the resulting spread.

    `z_window` sets the z-score lookback: None reuses `window`, an int fixes it,
    and "half_life" sizes it per bar from the spread's own measured speed of
    mean reversion (see `_half_life_window`), which needs 4 * window of extra
    warmup to estimate.
    """
    df = pd.DataFrame({"Asset_A": series_a, "Asset_B": series_b})

    # Rolling OLS slope of A on B == cov/var. The intercept is redundant: the
    # rolling z-score below subtracts the spread's own mean anyway.
    cov = df["Asset_A"].rolling(window).cov(df["Asset_B"])
    var = df["Asset_B"].rolling(window).var()
    df["Hedge_Ratio"] = cov / var.where(var > 0)

    df["Spread"] = df["Asset_A"] - df["Hedge_Ratio"] * df["Asset_B"]

    if z_window == "half_life" or max_half_life is not None:
        df["Half_Life"] = half_life(df["Spread"], 4 * window)

    if z_window == "half_life":
        df["Z_Window"] = _half_life_window(df["Spread"], 4 * window, window, 4 * window)
        roll = df["Spread"].rolling(_VarWindow(df["Z_Window"]), min_periods=window)
    else:
        df["Z_Window"] = int(z_window or window)
        roll = df["Spread"].rolling(int(z_window or window))
    mean, std = roll.mean(), roll.std()
    df["Z_Score"] = (df["Spread"] - mean) / std.where(std > 0)

    # `~(hl <= max)` rather than `hl > max`: a NaN half-life is no measurable
    # reversion, which must screen the pair out, not wave it through.
    blocked = None if max_half_life is None else ~(df["Half_Life"] <= max_half_life)
    df["Signal"] = _positions(df["Z_Score"].to_numpy(), entry_z, exit_z, stop_z,
                              None if blocked is None else blocked.to_numpy())
    #  1: long spread (buy A, short hedge_ratio * B);  -1: short spread;  0: flat
    df["Position"] = df["Signal"].shift(1).fillna(0)

    # The hedge is struck once, on the entry bar, and held for the life of the
    # trade. Re-striking it every bar as Hedge_Ratio drifts means paying the
    # spread on both legs daily, which swamps the edge.
    entry = (df["Signal"] != 0) & (df["Signal"] != df["Signal"].shift(1))
    df["Trade_Hedge"] = df["Hedge_Ratio"].where(entry).ffill().where(df["Signal"] != 0)

    return df, df["Hedge_Ratio"]


class _VarWindow(pd.api.indexers.BaseIndexer):
    """Per-bar rolling window length. pandas supports this natively; the
    alternative is slicing the spread in a Python loop."""

    def __init__(self, sizes):
        self.sizes = np.asarray(sizes, dtype=np.int64)
        super().__init__()

    def get_window_bounds(self, num_values, min_periods=None, center=None,
                          closed=None, step=None):
        end = np.arange(1, num_values + 1, dtype=np.int64)
        return np.maximum(end - self.sizes[:num_values], 0), end


def half_life(spread: pd.Series, window: int = 240) -> pd.Series:
    """Ornstein-Uhlenbeck half-life of the spread, in bars, measured per bar.

    Fit ds_t = lambda * s_{t-1} + c on a rolling window; lambda < 0 means the
    spread pulls back toward its mean, and it covers half the remaining
    distance in -ln(2)/lambda bars. NaN where lambda >= 0 -- that window shows
    no reversion at all, which is a screening answer, not a missing value.

    Uses only data up to and including each bar. Unclipped: this is the
    measurement, and callers decide what is too slow to trade.
    """
    lag = spread.shift(1)
    lam = spread.diff().rolling(window).cov(lag) / lag.rolling(window).var()
    return -np.log(2) / lam.where(lam < 0)


def _half_life_window(spread: pd.Series, window: int, lo: int, hi: int) -> pd.Series:
    """Z-score lookback sized by the measured half-life, clipped to [lo, hi] so
    one noisy lambda cannot pick an absurd lookback. A pair that reverts in
    three days should not be scored against a sixty-day mean, and vice versa.

    ponytail: an unmeasurable half-life falls back to the slowest allowed
    lookback rather than standing the pair down -- sizing a window is not the
    place to make the trade/no-trade call. That is what max_half_life is for.
    """
    return half_life(spread, window).clip(lo, hi).fillna(hi).round().astype(int)


def _positions(
    z: np.ndarray, entry_z: float, exit_z: float, stop_z: float | None,
    blocked: np.ndarray | None = None,
) -> np.ndarray:
    """Entry/exit state machine over the z-score.

    Exit is side-aware: a long closes once z recovers to -exit_z, a short once
    z falls to +exit_z. So the default exit_z=0.0 means "exit at the mean" and
    actually fires, which `abs(z) <= 0` does not.

    `blocked` bars are not tradeable at all (see max_half_life): flatten and
    stay out, without arming the stop lockout -- the pair did nothing wrong,
    it just stopped qualifying.

    stop_z flattens a position whose spread keeps diverging -- the co-integration
    has broken, and mean reversion is no longer the bet. After a stop the pair is
    locked out until z comes back inside the entry band, otherwise the next bar
    re-enters the same losing trade.

    ponytail: plain loop, O(n). A vectorised band test misses the bar where z
    gaps clean across the exit band; vectorise only if this is ever hot.
    """
    stop = np.inf if stop_z is None else stop_z
    out = np.zeros(len(z))
    pos = 0
    locked = False
    for i, zi in enumerate(z):
        if blocked is not None and blocked[i]:
            pos = 0
            out[i] = pos
            continue
        if np.isnan(zi):
            pos, locked = 0, False
        elif pos != 0 and abs(zi) >= stop:
            pos, locked = 0, True  # relationship broke; stand down
        elif pos == 1 and zi >= -exit_z:
            pos = 0
        elif pos == -1 and zi <= exit_z:
            pos = 0
        if locked and not np.isnan(zi) and abs(zi) < entry_z:
            locked = False
        if pos == 0 and not locked and not np.isnan(zi):
            if zi <= -entry_z:
                pos = 1  # spread cheap: buy A, short B
            elif zi >= entry_z:
                pos = -1  # spread rich: short A, buy B
        out[i] = pos
    return out


def spread_returns(df: pd.DataFrame, cost_bps: float = 5.0) -> pd.Series:
    """Net per-bar return on gross notional, after round-trip costs.

    The trade is 1 share of A against -h shares of B, so PnL is dA - h * dB and
    the capital at risk is |A| + |h * B|. cost_bps is charged on the notional
    actually traded, which -- since the hedge is frozen at entry -- means only
    on the entry and exit bars.
    """
    a, b = df["Asset_A"], df["Asset_B"]
    pos, hedge = df["Position"], df["Trade_Hedge"].shift(1).fillna(0)

    pnl = pos * (a.diff() - hedge * b.diff())
    notional = (a.abs() + (hedge * b).abs()).shift(1)

    # Shares turned over on each leg when the position or the struck hedge moves.
    traded = (pos - pos.shift(1)).abs() * a + (pos * hedge).diff().abs() * b
    cost = traded.abs() * cost_bps / 1e4

    return ((pnl - cost) / notional.where(notional > 0)).fillna(0)


def check_cointegration(series_a: pd.Series, series_b: pd.Series) -> float:
    """Engle-Granger p-value over the WHOLE sample. Diagnostic only, never a
    signal input: selecting pairs on this and then backtesting them is exactly
    the look-ahead this module otherwise avoids."""
    from statsmodels.tsa.stattools import coint

    return coint(series_a, series_b)[1]


if __name__ == "__main__":
    np.random.seed(42)
    days = 250

    returns_b = np.random.normal(0.0005, 0.01, days)
    price_b = pd.Series(100 * np.exp(np.cumsum(returns_b)))
    # Stationary residual: A - 1.5 * B is white noise, so the pair really is
    # co-integrated. `np.cumsum(noise)` here would make it a random walk, i.e.
    # I(1), and there would be no stable hedge ratio to find.
    price_a = 1.5 * price_b + 10 + np.random.normal(0, 0.5, days)

    df, hedge = calculate_pairs_signals(price_a, price_b, window=30)
    z, sig, pos = df["Z_Score"], df["Signal"], df["Position"]

    live = z.notna()
    entries = (sig != 0) & (sig != sig.shift(1))
    assert (sig[entries] == -np.sign(z[entries])).all(), "entry must fade the spread"
    assert (z[entries].abs() >= 2.0).all(), "entered inside the band"
    assert (z[entries].abs() < 4.0).all(), "entered beyond the stop"
    assert (sig[z.isna()] == 0).all(), "no position during warmup"
    assert (sig == 0).sum() > 0, "exit_z=0.0 never flattens -- exit rule is dead"
    assert sig.shift(1).fillna(0).equals(pos), "Position must lag Signal by a bar"
    assert (sig[live & (z.abs() >= 4.0)] == 0).all(), "stop_z must flatten"
    # Stop then lockout: long at -2.5, stopped at -4.5, held out through -3.0 and
    # -2.5, released at -1.0, free to re-enter at the next -2.5.
    trace = np.array([np.nan, 0, -2.5, -3.0, -4.5, -3.0, -2.5, -1.0, -2.5])
    assert list(_positions(trace, 2.0, 0.0, 4.0)) == [0, 0, 1, 1, 0, 0, 0, 0, 1]
    assert list(_positions(trace, 2.0, 0.0, None)) == [0, 0, 1, 1, 1, 1, 1, 1, 1]
    # No look-ahead: truncating the series must not change earlier rows.
    head, _ = calculate_pairs_signals(price_a[:200], price_b[:200], window=30)
    assert np.allclose(head["Signal"], sig[:200]), "signal depends on future data"
    # Hedge frozen for the life of each trade.
    held = df["Trade_Hedge"][sig != 0]
    assert held.notna().all() and df["Trade_Hedge"][sig == 0].isna().all()
    # Co-integrated demo data => the rolling hedge ratio finds the true beta.
    assert abs(hedge.dropna().median() - 1.5) < 0.1, "rolling beta should recover 1.5"
    # Costs can only hurt.
    assert (spread_returns(df, 5.0) <= spread_returns(df, 0.0) + 1e-12).all()

    # Half-life sizing: build OU spreads with known half-lives and see if the
    # measured window tracks them. phi = 0.5 ** (1 / hl) per bar.
    for true_hl in (3, 20):
        phi, x, path = 0.5 ** (1 / true_hl), 0.0, []
        for _ in range(3000):
            x = phi * x + np.random.normal(0, 1)
            path.append(x)
        est = _half_life_window(pd.Series(path), 240, 1, 1000).dropna()
        assert abs(est.median() - true_hl) < 0.4 * true_hl, (true_hl, est.median())
    # The screen: slow or unmeasurable half-life => no position, ever.
    screened = calculate_pairs_signals(price_a, price_b, window=30, max_half_life=10)[0]
    hl = screened["Half_Life"]
    assert (screened["Signal"][~(hl <= 10)] == 0).all(), "screen must flatten slow bars"
    assert (screened["Signal"] != 0).sum() <= (sig != 0).sum(), "screen only removes"
    assert (screened["Signal"] != 0).sum() > 0, "white-noise spread should pass a 10-bar screen"
    blocked_all = calculate_pairs_signals(price_a, price_b, window=30, max_half_life=0.0)[0]
    assert (blocked_all["Signal"] == 0).all(), "impossible threshold must block everything"

    fast = calculate_pairs_signals(price_a, price_b, window=30, z_window="half_life")[0]
    assert fast["Z_Window"].between(30, 120).all(), "window must stay inside its clip"
    assert (fast["Signal"][fast["Z_Score"].isna()] == 0).all()

    gross, net = spread_returns(df, 0.0), spread_returns(df, 5.0)
    print(f"rolling hedge ratio: median {hedge.median():.4f} (true beta 1.5)")
    print(f"bars flat: {(sig == 0).sum()}  long: {(sig == 1).sum()}  short: {(sig == -1).sum()}")
    print(f"round trips: {int(((sig != 0) & (sig != sig.shift(1))).sum())}")
    print(f"cumulative return  gross {gross.sum():+.2%}   net of 5bps {net.sum():+.2%}")
    try:
        print(f"co-integration p-value: {check_cointegration(price_a, price_b):.4g}")
    except ImportError:
        print("co-integration p-value: statsmodels not installed")
