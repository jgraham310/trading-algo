"""Cross-sectional test: does the stage RANK names, even if it can't time one?

The Pine script is a scanner -- 39 tickers sorted by stage rank -- so its
implied use is selection, not timing: out of a theme, hold the Stage 2 names and
skip the Stage 4 ones. That is a different claim from "be long X while X is
Stage 2", and it survives even if the timing claim fails.

The control is equal-weighting the SAME eligible universe every week. If stage
sorts, picking a subset beats holding all of it. A second control holds the full
universe at the strategy's own average exposure, so a rule that sits in cash
isn't credited for merely being less invested.

SURVIVORSHIP: names are eligible only in weeks where they actually have prices,
so a name that delists drops out on its delisting date rather than being
silently backfilled. What CANNOT be fixed here is that the script's ticker lists
are today's -- names that were in these themes historically and got dropped are
absent. That biases the stock universes UP. Read `SECTORS` as the clean test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import _ann, _max_dd, _sharpe
from .stage import EMA, MA, SLOPE_LAG, stages, weekly

WEEKS = 52

# Survivorship-free: sector SPDRs are not selected on past performance.
SECTORS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]

# The script's own theme lists, verbatim. Today's tickers -> survivorship bias.
THEMES = {
    "Macro ETFs & World Indexes": [
        "XLK", "XLV", "XLY", "XLP", "XLE", "XLF", "XLI", "XLB", "XLRE", "XLU", "XLC",
        "SMH", "BOTZ", "ROBO", "UFO", "ITA", "QTUM", "CIBR", "SKYY", "ICLN", "URA",
        "LIT", "COPX", "ARKK", "XBI", "SPY", "QQQ", "IWM", "DIA", "FXI", "EWY",
        "EWG", "EWJ", "EWU", "INDA", "EWZ", "EWW", "EFA", "EEM"],
    "AI & Semiconductors": [
        "NVDA", "AMD", "TSM", "AVGO", "MU", "INTC", "ASML", "AMAT", "LRCX", "KLAC",
        "ARM", "SNPS", "CDNS", "MRVL", "TXN", "QCOM", "SWKS", "NXPI", "MCHP", "ON",
        "WDC", "STX", "RMBS", "LSCC", "POWI", "SYNA", "CRUS", "DIOD", "SLAB", "MPWR",
        "OLED", "TER", "UMC", "HIMX", "AOSL", "ACLS", "COHR", "MKSI", "VECO"],
}


def panel(tickers, years: int = 20, benchmark: str = "SPY"):
    """Weekly closes for the universe plus the benchmark, downloaded in one shot."""
    import yfinance as yf

    warm = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 2)).date())
    want = list(dict.fromkeys([*tickers, benchmark]))
    df = yf.download(want, start=warm, auto_adjust=True, progress=False, threads=True)["Close"]
    df = df.tz_localize(None) if df.index.tz else df
    w = df.resample("W-FRI").last()
    return w[[c for c in tickers if c in w]], w[benchmark]


def select(px_w: pd.DataFrame, bench_w: pd.Series, long_stages=(2,), rs_gate: bool = True,
           ma_len: int = MA, lag: int = SLOPE_LAG, ema_gate: bool = False):
    """Per-name weekly stage, and the boolean 'hold this one' matrix."""
    st = {}
    for tk in px_w:
        s = px_w[tk].dropna()
        if len(s) < ma_len + lag + WEEKS:
            continue
        d = stages(s, bench_w, rs_gate, ma_len, lag)
        ok = d["stage"].isin(long_stages)
        if ema_gate:
            ok &= d["close"] > d["ema8"]
        st[tk] = pd.DataFrame({"stage": d["stage"], "hold": ok.where(d["stage"].notna())})
    stage = pd.DataFrame({k: v["stage"] for k, v in st.items()}).reindex(px_w.index)
    hold = pd.DataFrame({k: v["hold"] for k, v in st.items()}).reindex(px_w.index)
    return stage, hold.fillna(False).astype(bool)


def run(tickers, years: int = 20, benchmark: str = "SPY", cost_bps: float = 5.0,
        long_stages=(2,), label: str = "", **kw) -> dict:
    """Equal-weight the selected names each week; cash when nothing qualifies."""
    px_w, bench_w = panel(tickers, years, benchmark)
    stage, hold = select(px_w, bench_w, long_stages, **kw)

    fwd = px_w.shift(-1) / px_w - 1.0           # week t -> t+1; decided at t's close
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    keep = (stage.index >= start) & fwd.notna().any(axis=1)
    eligible = stage.notna() & fwd.notna()       # in the universe AND scoreable
    stage, hold, fwd, eligible = (x[keep] for x in (stage, hold, fwd, eligible))

    sel = hold & eligible
    w = _norm(sel)                               # equal weight, 0 rows when empty
    ew = _norm(eligible)                         # hold the whole theme
    exposure = w.sum(axis=1)
    matched = ew.mul(exposure.mean(), axis=0)    # same average exposure, no picking

    out = {"label": label or f"{len(px_w.columns)} names", "weeks": int(len(fwd)),
           "names": int(eligible.sum(axis=1).mean()), "held": float(sel.sum(axis=1).mean()),
           "exposure": float(exposure.mean())}
    for name, wt in (("sel", w), ("ew", ew), ("matched", matched)):
        r = (wt * fwd).sum(axis=1)
        to = wt.diff().abs().sum(axis=1).fillna(wt.abs().sum(axis=1))
        net = (r - to * cost_bps / 1e4).to_numpy()
        out |= {f"{name}_ann": _ann(net, len(net), WEEKS), f"{name}_sh": _sharpe(net, WEEKS),
                f"{name}_dd": _max_dd(net), f"{name}_to": float(to.sum() / len(to) * WEEKS)}

    # The direct question, with no portfolio construction in the way: do the
    # selected names out-return the universe in the following week?
    spread = ((sel * fwd).sum(axis=1) / sel.sum(axis=1).replace(0, np.nan)
              - (eligible * fwd).sum(axis=1) / eligible.sum(axis=1))
    sp = spread.dropna()
    out |= {"spread_wk": float(sp.mean()), "spread_ann": float(sp.mean() * WEEKS),
            "spread_t": float(sp.mean() / sp.std(ddof=1) * np.sqrt(len(sp))), "spread_n": len(sp)}
    out["by_stage"] = {int(k): (float(v.mean() * WEEKS), int(len(v))) for k, v in
                       fwd.where(eligible).stack().groupby(stage.stack()) if len(v) > 50}
    return out


def _norm(mask: pd.DataFrame) -> pd.DataFrame:
    n = mask.sum(axis=1)
    return mask.astype(float).div(n.where(n > 0), axis=0).fillna(0.0)


def report(rows) -> str:
    head = (f"{'universe':<28}{'wks':>5}{'names':>6}{'held':>6}{'expo':>6}"
            f"{'SEL':>8}{'EWall':>8}{'MATCH':>8}{'SELsh':>7}{'EWsh':>7}{'MTCHsh':>7}"
            f"{'spread':>8}{'t':>6}")
    body = []
    for m in rows:
        body.append(f"{m['label']:<28}{m['weeks']:>5}{m['names']:>6.0f}{m['held']:>6.1f}"
                    f"{m['exposure']:>6.0%}{m['sel_ann']:>+8.1%}{m['ew_ann']:>+8.1%}"
                    f"{m['matched_ann']:>+8.1%}{m['sel_sh']:>7.2f}{m['ew_sh']:>7.2f}"
                    f"{m['matched_sh']:>7.2f}{m['spread_ann']:>+8.1%}{m['spread_t']:>6.2f}")
    tail = []
    for m in rows:
        bs = "  ".join(f"St{k} {v[0]:+.1%} (n={v[1]})" for k, v in sorted(m["by_stage"].items()))
        tail.append(f"  {m['label']}: annualized fwd return by stage -- {bs}")
    return head + "\n" + "\n".join(body) + "\n\n" + "\n".join(tail)


def demo():
    """Self-check: weighting, cash rows, and no look-ahead in the fwd alignment."""
    idx = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    px = pd.DataFrame({"A": [10, 11, 12, 13, 14, 15.0], "B": [10, 9, 8, 7, 6, 5.0]}, index=idx)
    fwd = px.shift(-1) / px - 1.0
    assert abs(fwd["A"].iloc[0] - 0.1) < 1e-12, "fwd[t] must be the t->t+1 return"

    sel = pd.DataFrame([[True, False]] * 6, index=idx, columns=["A", "B"])
    w = _norm(sel & fwd.notna())
    assert (w.sum(axis=1).iloc[:5] == 1.0).all() and w.sum(axis=1).iloc[-1] == 0.0
    assert abs((w * fwd).sum(axis=1).iloc[0] - 0.1) < 1e-12, "portfolio != picked name's return"

    none = _norm(pd.DataFrame(False, index=idx, columns=["A", "B"]))
    assert (none.sum(axis=1) == 0).all(), "empty selection must be cash, not NaN"
    print("ok")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        rows = [run(SECTORS, label="SECTORS (no surv. bias)")]
        rows += [run(v, label=k + " *") for k, v in THEMES.items()]
        print("CROSS-SECTIONAL  hold Stage 2, equal weight, weekly rebalance, 5bp\n")
        print(report(rows))
        print("\n  * survivorship-biased: the script's lists are TODAY's tickers.")
