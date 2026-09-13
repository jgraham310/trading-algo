"""Stan Weinstein 4-stage model + Mansfield RS, as in the Pine dashboard.

The Pine script is a scanner, not a strategy: the only tradeable statement in
it is the stage a symbol is in. So the rule backtested here is the obvious one
-- long in Stage 2, flat otherwise -- on weekly bars, held through the
following week's daily bars.

WATCH OUT: Stage 2 requires Mansfield RS > 0 against the benchmark. Backtesting
SPY against the script's default benchmark (SPY) makes RS identically 0, so
Stage 2 is unreachable. `rs_gate=False` scores the model without that leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import _stats
from .timesfm_signal import load_closes

MA, EMA, RS_MA, SLOPE_LAG = 30, 8, 52, 3  # the script's inputs, in weeks


def weekly(daily: pd.Series) -> pd.Series:
    """Last close of each week -- the bars the Pine script sees on "W"."""
    return daily.resample("W-FRI").last().dropna()


def stages(w: pd.Series, bench: pd.Series | None = None, rs_gate: bool = True,
           ma_len: int = MA, lag: int = SLOPE_LAG) -> pd.DataFrame:
    """Per-week stage 1..4 (0 = the script's "1>2 (Wait RS)" transition)."""
    ma = w.rolling(ma_len).mean()
    ma_old = w.shift(lag).rolling(ma_len).mean()  # == sma(close[3], 30)
    m52 = w.rolling(RS_MA).mean()

    if bench is None or bench is w:
        rs = pd.Series(0.0, index=w.index)
    else:
        b = bench.reindex(w.index)
        rs = ((w / b) / (m52 / b.rolling(RS_MA).mean()) - 1.0) * 100.0

    above, rising, falling = w > ma, ma > ma_old, ma < ma_old
    st = pd.Series(np.nan, index=w.index)
    st[above & rising] = 0.0 if rs_gate else 2.0       # "wait for RS" bucket
    if rs_gate:
        st[above & rising & (rs > 0)] = 2.0
    st[above & ~rising] = 1.0
    st[~above & ~falling] = 3.0
    st[~above & falling] = 4.0
    st[ma.isna() | ma_old.isna() | (rs_gate and m52.isna())] = np.nan

    return pd.DataFrame({"close": w, "ma30": ma, "ma30_prev": ma_old,
                         "ema8": w.ewm(span=EMA, adjust=False).mean(),
                         "rs": rs, "stage": st})


def positions(daily: pd.Series, bench: pd.Series | None = None, rs_gate: bool = True,
              long_stages=(2,), ma_len: int = MA, lag: int = SLOPE_LAG,
              ema_gate: bool = False) -> pd.Series:
    """Daily 0/1 exposure from the weekly stage, decided at the weekly close.

    The stage for week t is only actionable from week t+1, hence the shift --
    without it you trade on a close you could not have known you'd get.

    `ema_gate` adds the dashboard's "8W Mom = HOLD" column: close > 8-week EMA.
    The other two columns ("Vs 30W = ABOVE", "RS = POS") are already implied by
    Stage 2, so this is the only one that narrows anything.
    """
    df = stages(weekly(daily), bench, rs_gate, ma_len, lag)
    ok = df["stage"].isin(long_stages)
    if ema_gate:
        ok &= df["close"] > df["ema8"]
    pos = ok.astype(float).where(df["stage"].notna()).shift(1)
    return pos.reindex(daily.index, method="ffill")


def entry_exit(daily: pd.Series, bench: pd.Series | None = None, rs_gate: bool = True,
               entry_stages=(1,), exit_stages=(3, 4), ma_len: int = MA,
               lag: int = SLOPE_LAG, ema_gate: bool = True,
               rs_entry: bool = True, rs_exit: bool = False) -> pd.Series:
    """Hysteresis rule: enter on a stage/filter trigger, HOLD until an exit stage.

    Different animal from `positions`, which re-tests its condition every week
    and drops out the moment it fails. Here the entry gates are checked only
    when flat, so a position rides through Stage 2 -- and through the "1>2 wait
    RS" bucket -- until Stage 3 or 4 fires. That is what cuts the churn.
    """
    df = stages(weekly(daily), bench, rs_gate, ma_len, lag)
    enter = df["stage"].isin(entry_stages)
    if ema_gate:
        enter &= df["close"] > df["ema8"]
    if rs_entry:
        # The dashboard's "RS = POS" column. Stage 1 does NOT imply it -- only
        # Stage 2 has RS baked in -- so for a Stage 1 entry it must be explicit.
        enter &= df["rs"] > 0
    exit_ = df["stage"].isin(exit_stages)
    if rs_exit:
        # Leave when it stops outperforming, rather than refusing to enter until
        # it already does -- Stage 1 and positive RS rarely coincide.
        exit_ |= df["rs"] <= 0

    held, on = [], False
    for e, x, warm in zip(enter, exit_, df["stage"].notna()):
        if not warm:
            on = False
            held.append(np.nan)
            continue
        on = False if x else (True if e else on)  # exit wins; stages are exclusive anyway
        held.append(float(on))
    pos = pd.Series(held, index=df.index).shift(1)
    return pos.reindex(daily.index, method="ffill")


def run(ticker: str = "SPY", years: int = 20, benchmark: str = "SPY",
        cost_bps: float = 1.0) -> str:
    warm = pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 2)
    px = load_closes(ticker, start=str(warm.date()))
    bench = px if benchmark == ticker else weekly(load_closes(benchmark, start=str(warm.date())))

    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    nxt = (px.shift(-1) / px - 1.0)[start:].dropna()
    idx = nxt.index

    def score(pos, prefix):
        p = pos.reindex(idx)
        if p.isna().any():
            raise ValueError(f"{prefix}: signal not warmed up by {idx[0].date()}")
        return _stats(p.to_numpy(), nxt.to_numpy(), cost_bps, prefix)

    m = {
        **score(positions(px, bench, rs_gate=True), "rs_"),
        **score(positions(px, bench, rs_gate=False), "st2_"),
        **score(positions(px, bench, rs_gate=False, long_stages=(1, 2)), "st12_"),
        **score(pd.Series(1.0, index=idx), "bh_"),
    }
    dist = stages(weekly(px), bench)["stage"][start:].value_counts(normalize=True)
    names = {0: "1>2 wait RS", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "Stage 4"}

    def row(label, key, fmt):
        return f"  {label:<14}" + "".join(f"{format(m[p + key], fmt):>14}"
                                          for p in ("rs_", "st2_", "st12_", "bh_"))

    return f"""\
=== {ticker} weekly 4-stage model  {idx[0].date()} .. {idx[-1].date()}  ({len(idx)} daily bars) ===
benchmark for Mansfield RS: {benchmark}{'   <-- SAME AS TICKER: RS is identically 0' if benchmark == ticker else ''}

weeks per stage: """ + "  ".join(f"{names[k]} {v:.0%}" for k, v in sorted(dist.items())) + f"""

  {'':<14}{'STAGE2+RS':>14}{'STAGE2':>14}{'STAGE 1+2':>14}{'BUY&HOLD':>14}
{row('ann return', 'ann_return', '+.1%')}
{row('sharpe', 'sharpe', '.2f')}
{row('max drawdown', 'max_dd', '+.1%')}
{row('time in mkt', 'time_in_mkt', '.0%')}
{row('trades', 'trades', 'd')}
{row('hit rate', 'hit_rate', '.1%')}
{row('cost drag', 'cost_drag', '+.2%')}

Costs {cost_bps:.0f}bp per unit turnover, closes only, no slippage/borrow/taxes.
"""


def demo():
    """Self-check: stage classification and the no-look-ahead shift."""
    # 40 weeks up then 40 down -> must end Stage 4 having passed through 2.
    up = pd.Series(np.r_[np.arange(100, 140.0), np.arange(140, 100.0, -1)],
                   index=pd.date_range("2020-01-03", periods=80, freq="W-FRI"))
    st = stages(up, rs_gate=False)["stage"]
    assert st.iloc[:MA + SLOPE_LAG - 1].isna().all(), "warmup must be NaN, not a stage"
    assert st.iloc[MA + SLOPE_LAG] == 2, st.iloc[MA + SLOPE_LAG]
    assert st.iloc[-1] == 4, st.iloc[-1]
    assert set(st.dropna()) <= {1.0, 2.0, 3.0, 4.0}

    # RS gate: against itself nothing is ever Stage 2.
    assert (stages(up, up, rs_gate=True)["stage"] == 2).sum() == 0

    # The daily position on the week's own bars comes from the PREVIOUS week.
    daily = up.resample("D").ffill()
    pos = positions(daily, rs_gate=False)
    wk = st.index[MA + SLOPE_LAG]
    assert pos.loc[wk] == float(st.iloc[MA + SLOPE_LAG - 1] == 2), "look-ahead: used its own week"

    # The 8W EMA gate can only ever remove exposure, never add it.
    strict = positions(daily, rs_gate=False, ema_gate=True)
    assert ((strict <= pos) | pos.isna()).all(), "ema_gate must be a subset"
    assert strict.sum() < pos.sum(), "ema_gate never bound -- check the column"

    # Hysteresis: must hold through Stage 2 (which `positions` also holds) and
    # must trade strictly less often than the re-tested-every-week version.
    ee = entry_exit(daily, rs_gate=False, ema_gate=False)
    flips = lambda s: int((s.dropna().diff().abs() > 0).sum())
    assert flips(ee) <= flips(pos), (flips(ee), flips(pos))
    assert ee.dropna().isin([0.0, 1.0]).all()
    # once entered it stays on until an exit stage -- never flat while Stage 2
    st_d = st.shift(1).reindex(daily.index, method="ffill")
    assert not ((st_d == 2) & (ee == 0) & (ee.shift(1) == 1)).any(), "dropped out mid-Stage-2"
    print("ok")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        a = [x for x in sys.argv[1:] if not x.startswith("-")]
        print(run(a[0] if a else "SPY", int(a[1]) if len(a) > 1 else 20,
                  a[2] if len(a) > 2 else "SPY"))
