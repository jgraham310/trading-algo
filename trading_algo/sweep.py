"""Parameter sweep for the 4-stage model, scored against the control that matters.

The benchmark for a sometimes-in-the-market rule is NOT buy & hold -- a rule
that is flat 30% of the time SHOULD earn less. It is the same asset held at
constant exposure equal to the rule's own time-in-market: identical average
risk, zero timing. `edge` is the rule minus that control, so a positive edge
means the signal picked WHICH days to be in, not merely HOW MANY.

Sector SPDRs, deliberately: the set has no survivorship bias, unlike any basket
of today's winners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import _ann, _max_dd, _sharpe, _stats
from .stage import positions, weekly
from .timesfm_signal import load_closes

BASKET = ["SPY", "XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU"]
MAS = range(10, 51, 5)
LAGS = range(1, 9)


def _bars(tickers, years: int, benchmark: str):
    """Daily next-bar returns per ticker plus the weekly benchmark, downloaded once."""
    warm = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 2)).date())
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    px = {tk: load_closes(tk, start=warm) for tk in dict.fromkeys([*tickers, benchmark])}
    bench = weekly(px[benchmark])
    return {tk: (px[tk], (px[tk].shift(-1) / px[tk] - 1.0)[start:].dropna()) for tk in tickers}, bench


def sweep(tickers=BASKET, years: int = 20, benchmark: str = "SPY", mas=MAS, lags=LAGS,
          rs_gates=(False, True), cost_bps: float = 1.0, span=None) -> pd.DataFrame:
    """One row per (ticker, rs_gate, ma_len, lag). `span` slices the scored bars."""
    data, bench = _bars(tickers, years, benchmark)
    rows = []
    for tk, (px, nxt) in data.items():
        if span is not None:
            nxt = nxt[span[0]:span[1]]
        idx, r = nxt.index, nxt.to_numpy()
        for gate in rs_gates:
            for ma_len in mas:
                for lag in lags:
                    p = positions(px, bench, gate, ma_len=ma_len, lag=lag).reindex(idx)
                    if p.isna().any():
                        raise ValueError(f"{tk} ma={ma_len} lag={lag}: not warmed up by {idx[0].date()}")
                    p = p.to_numpy()
                    s = _stats(p, r, cost_bps)
                    tim = s["time_in_mkt"]
                    # same average exposure, no timing decisions at all
                    c = _stats(np.full(len(r), tim), r, cost_bps)
                    rows.append({
                        "ticker": tk, "rs": gate, "ma": ma_len, "lag": lag, "tim": tim,
                        "ann": s["ann_return"], "sharpe": s["sharpe"], "trades": s["trades"],
                        "edge_ann": s["ann_return"] - c["ann_return"],
                        "edge_sharpe": s["sharpe"] - c["sharpe"],
                    })
    return pd.DataFrame(rows)


def grid(df: pd.DataFrame, rs: bool = False, metric: str = "edge_sharpe") -> pd.DataFrame:
    """Basket-mean `metric` as an ma x lag grid."""
    return df[df.rs == rs].pivot_table(index="ma", columns="lag", values=metric)


def report(df: pd.DataFrame, metric: str = "edge_sharpe") -> str:
    out = []
    n_cfg = df.groupby("rs")[["ma", "lag"]].apply(lambda d: len(d.drop_duplicates())).iloc[0]
    for gate in sorted(df.rs.unique()):
        g = grid(df, gate, metric)
        best = g.stack().idxmax()
        cfg = df[(df.rs == gate) & (df.ma == best[0]) & (df.lag == best[1])]
        out.append(
            f"\nRS gate {'ON ' if gate else 'OFF'}   basket-mean {metric}, {n_cfg} configs\n"
            + g.round(2).to_string()
            + f"\n  configs with positive basket-mean: {int((g > 0).sum().sum())}/{g.size}"
            + f"   best ma={best[0]} lag={best[1]} ({g.loc[best]:+.2f})"
            + f", positive on {int((cfg[metric] > 0).sum())}/{len(cfg)} tickers"
        )
    return "\n".join(out)


def holdout(tickers=BASKET, years: int = 20, benchmark: str = "SPY", metric: str = "edge_sharpe",
            **kw) -> str:
    """Pick the best config on the first half. Measure it on the second half.

    This is the only number in the sweep that isn't selected on itself.
    """
    mid = pd.Timestamp.today().normalize() - pd.DateOffset(years=years // 2)
    a = sweep(tickers, years, benchmark, span=(None, mid), **kw)
    b = sweep(tickers, years, benchmark, span=(mid, None), **kw)
    lines = []
    for gate in sorted(a.rs.unique()):
        ma, lag = grid(a, gate, metric).stack().idxmax()
        sel = b[(b.rs == gate) & (b.ma == ma) & (b.lag == lag)]
        allb = grid(b, gate, metric).stack()
        lines.append(
            f"  RS {'ON ' if gate else 'OFF'}  in-sample pick ma={ma} lag={lag}"
            f" ({grid(a, gate, metric).loc[ma, lag]:+.2f})"
            f" -> out-of-sample {sel[metric].mean():+.2f}"
            f"  (best available OOS {allb.max():+.2f}, median config {allb.median():+.2f})"
            f"  positive on {int((sel[metric] > 0).sum())}/{len(sel)} tickers")
    return (f"HOLDOUT  train {years}y..{years // 2}y ago   test {years // 2}y ago..now\n"
            + "\n".join(lines))


if __name__ == "__main__":
    df = sweep()
    df.to_csv("sweep_basket.csv", index=False)
    print(f"=== 4-stage sweep  basket={' '.join(BASKET)} ===")
    print(report(df))
    print()
    print(report(df, "edge_ann"))
    print()
    print(holdout())


def strict(tickers=None, years: int = 20, benchmark: str = "SPY", cost_bps: float = 1.0) -> str:
    """The dashboard's all-green rule: Stage 2 AND above 30W AND above 8W EMA AND RS>0.

    "Vs 30W = ABOVE" and "RS = POS" are already inside Stage 2, so the 8W EMA is
    the only column that narrows anything. Scored against constant exposure at
    the STRICT rule's own time-in-market -- the same-risk, no-timing control.
    """
    tickers = tickers or BASKET
    data, bench = _bars(tickers, years, benchmark)
    rows = []
    for tk, (px, nxt) in data.items():
        idx, r = nxt.index, nxt.to_numpy()

        def sc(**kw):
            return _stats(positions(px, bench, **kw).reindex(idx).to_numpy(), r, cost_bps)

        f = sc(rs_gate=True, ema_gate=True)   # all four columns green
        s = sc(rs_gate=True, ema_gate=False)  # Stage 2 + RS only
        c = _stats(np.full(len(r), f["time_in_mkt"]), r, cost_bps)  # matched exposure
        b = _stats(np.ones(len(r)), r, cost_bps)
        rows.append({"ticker": tk, "tim": f["time_in_mkt"], "trades": f["trades"],
                     "full": f["ann_return"], "full_sh": f["sharpe"], "full_dd": f["max_dd"],
                     "stage2rs": s["ann_return"], "s2_sh": s["sharpe"],
                     "ctrl": c["ann_return"], "ctrl_sh": c["sharpe"],
                     "bh": b["ann_return"], "bh_sh": b["sharpe"],
                     "edge_ann": f["ann_return"] - c["ann_return"],
                     "edge_sh": f["sharpe"] - c["sharpe"]})
    d = pd.DataFrame(rows).set_index("ticker")
    head = (f"{'':<7}{'tim':>6}{'trd':>5}{'FULL':>8}{'+RS only':>10}{'CTRL':>8}{'B&H':>8}"
            f"{'edge':>8}{'FULLsh':>8}{'CTRLsh':>8}{'edgesh':>8}{'maxDD':>8}")
    body = "\n".join(
        f"{tk:<7}{x.tim:>6.0%}{int(x.trades):>5d}{x.full:>+8.1%}{x.stage2rs:>+10.1%}{x.ctrl:>+8.1%}"
        f"{x.bh:>+8.1%}{x.edge_ann:>+8.1%}{x.full_sh:>8.2f}{x.ctrl_sh:>8.2f}"
        f"{x.edge_sh:>+8.2f}{x.full_dd:>+8.1%}" for tk, x in d.iterrows())
    n = len(d)
    return (f"ALL-GREEN RULE  (Stage 2 + above 30W + above 8W EMA + RS>0)  vs {benchmark}\n"
            + head + "\n" + body + "\n"
            + f"\n  mean edge: {d.edge_ann.mean():+.2%}/yr, {d.edge_sh.mean():+.2f} sharpe"
            + f"   positive on {int((d.edge_sh > 0).sum())}/{n} (sharpe),"
            + f" {int((d.edge_ann > 0).sum())}/{n} (return)"
            + f"\n  beats buy & hold on sharpe: {int((d.full_sh > d.bh_sh).sum())}/{n}"
            + f"   the 8W EMA leg vs Stage2+RS alone: {(d.full - d.stage2rs).mean():+.2%}/yr\n")


def hyst(tickers=None, years: int = 20, benchmark: str = "SPY", cost_bps: float = 1.0,
         entry_stages=(1,), exit_stages=(3, 4), rs_gate: bool = True,
         ema_gate: bool = True, rs_entry: bool = True, rs_exit: bool = False) -> str:
    """Enter on Stage 1 + 8W EMA + RS>0, hold through Stage 2, exit on Stage 3/4."""
    from .stage import entry_exit

    tickers = tickers or BASKET
    data, bench = _bars(tickers, years, benchmark)
    rows = []
    for tk, (px, nxt) in data.items():
        idx, r = nxt.index, nxt.to_numpy()
        p = entry_exit(px, bench, rs_gate, entry_stages, exit_stages,
                       ema_gate=ema_gate, rs_entry=rs_entry, rs_exit=rs_exit).reindex(idx)
        if p.isna().any():
            raise ValueError(f"{tk}: not warmed up by {idx[0].date()}")
        h = _stats(p.to_numpy(), r, cost_bps)
        s = _stats(positions(px, bench, rs_gate, ema_gate=ema_gate).reindex(idx).to_numpy(),
                   r, cost_bps)
        c = _stats(np.full(len(r), h["time_in_mkt"]), r, cost_bps)
        b = _stats(np.ones(len(r)), r, cost_bps)
        rows.append({"ticker": tk, "tim": h["time_in_mkt"], "trades": h["trades"],
                     "hyst": h["ann_return"], "h_sh": h["sharpe"], "h_dd": h["max_dd"],
                     "allgreen": s["ann_return"], "ag_trd": s["trades"],
                     "ctrl": c["ann_return"], "c_sh": c["sharpe"],
                     "bh": b["ann_return"], "b_sh": b["sharpe"], "b_dd": b["max_dd"],
                     "edge_ann": h["ann_return"] - c["ann_return"],
                     "edge_sh": h["sharpe"] - c["sharpe"]})
    d = pd.DataFrame(rows).set_index("ticker")
    head = (f"{'':<7}{'tim':>6}{'trd':>5}{'HYST':>8}{'ALLGRN':>8}{'(trd)':>7}{'CTRL':>8}"
            f"{'B&H':>8}{'edge':>8}{'HYSTsh':>8}{'CTRLsh':>8}{'edgesh':>8}{'maxDD':>8}{'bhDD':>8}")
    body = "\n".join(
        f"{tk:<7}{x.tim:>6.0%}{int(x.trades):>5d}{x.hyst:>+8.1%}{x.allgreen:>+8.1%}"
        f"{int(x.ag_trd):>7d}{x.ctrl:>+8.1%}{x.bh:>+8.1%}{x.edge_ann:>+8.1%}{x.h_sh:>8.2f}"
        f"{x.c_sh:>8.2f}{x.edge_sh:>+8.2f}{x.h_dd:>+8.1%}{x.b_dd:>+8.1%}"
        for tk, x in d.iterrows())
    n = len(d)
    rs_in = "+RS>0" if rs_entry else ""
    rs_out = " or RS<=0" if rs_exit else ""
    return (f"HYSTERESIS  enter Stage{entry_stages}+8W EMA{rs_in}, hold, "
            f"exit Stage{exit_stages}{rs_out}  vs {benchmark}\n" + head + "\n" + body + "\n"
            + f"\n  mean edge: {d.edge_ann.mean():+.2%}/yr, {d.edge_sh.mean():+.2f} sharpe"
            + f"   positive on {int((d.edge_sh > 0).sum())}/{n} (sharpe),"
            + f" {int((d.edge_ann > 0).sum())}/{n} (return)"
            + f"\n  beats buy & hold on sharpe: {int((d.h_sh > d.b_sh).sum())}/{n}"
            + f"   median trades {int(d.trades.median())} vs {int(d.ag_trd.median())} all-green\n")


# Beta spans roughly 0.5 (staples) to 1.6 (semis, miners, country funds).
BETA_ETFS = ["XLP", "XLU", "XLV", "XLK", "XLE", "XLF", "XLI", "XLY", "XLB", "SPY", "QQQ",
             "IWM", "DIA", "EFA", "EEM", "EWZ", "EWJ", "EWY", "EWG", "EWU", "FXI", "IYR",
             "XBI", "IBB", "IGV", "SOXX", "GDX", "KRE", "XHB", "XRT", "XME", "ITB", "OIH"]
BETA_STOCKS = ["NVDA", "AMD", "MU", "LRCX", "AAPL", "NFLX", "AMZN", "MSFT", "GOOGL",
               "JNJ", "PG", "KO", "WMT", "XOM", "VZ", "PFE", "MRK", "CVX", "MCD", "T"]


def beta_scan(tickers=None, years: int = 20, benchmark: str = "SPY", cost_bps: float = 1.0,
              rs_gate: bool = False, ema_gate: bool = False, label: str = "") -> pd.DataFrame:
    """Per-ticker timing edge vs its own matched-exposure control, sorted by beta.

    The edge is internally controlled -- strategy vs the SAME asset held at the
    same average exposure -- so survivorship bias, which inflates both sides
    equally, largely cancels. That is what makes this comparable across a set
    of names picked for their beta rather than at random.
    """
    tickers = list(tickers or BETA_ETFS)
    warm = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 2)).date())
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    spy = load_closes(benchmark, start=warm)
    bench = weekly(spy)
    # FORWARD returns on both legs -- `nxt` is forward, so a trailing market
    # return here would be off by a bar and beta would come out near zero.
    spy_r = (spy.shift(-1) / spy - 1.0)[start:]

    rows, skipped = [], []
    for tk in tickers:
        px = load_closes(tk, start=warm)
        if px.empty or px.index[0] > pd.Timestamp(warm) + pd.Timedelta(days=120):
            skipped.append(tk)
            continue
        nxt = (px.shift(-1) / px - 1.0)[start:].dropna()
        idx, r = nxt.index, nxt.to_numpy()
        p = positions(px, bench, rs_gate, ema_gate=ema_gate).reindex(idx)
        if p.isna().any():
            skipped.append(tk)
            continue
        s = _stats(p.to_numpy(), r, cost_bps)
        c = _stats(np.full(len(r), s["time_in_mkt"]), r, cost_bps)
        b = _stats(np.ones(len(r)), r, cost_bps)
        d = pd.concat([nxt.rename("x"), spy_r.rename("m")], axis=1).dropna()
        rows.append({"ticker": tk, "beta": float(d.x.cov(d.m) / d.m.var()),
                     "vol": float(nxt.std() * np.sqrt(252)), "tim": s["time_in_mkt"],
                     "ann": s["ann_return"], "sh": s["sharpe"], "dd": s["max_dd"],
                     "bh_ann": b["ann_return"], "bh_sh": b["sharpe"], "bh_dd": b["max_dd"],
                     "edge_ann": s["ann_return"] - c["ann_return"],
                     "edge_sh": s["sharpe"] - c["sharpe"],
                     # +ve = the rule's drawdown is that many points SHALLOWER
                     "dd_cut": s["max_dd"] - b["max_dd"],
                     "beats_bh": float(s["sharpe"] > b["sharpe"])})
    df = pd.DataFrame(rows).set_index("ticker").sort_values("beta")
    df.attrs["skipped"] = skipped
    df.attrs["label"] = label or f"{len(df)} names"
    return df


def beta_report(df: pd.DataFrame) -> str:
    head = (f"{'':<8}{'beta':>6}{'vol':>7}{'tim':>6}{'STRAT':>8}{'B&H':>8}{'edge':>8}"
            f"{'STRATsh':>9}{'B&Hsh':>7}{'edgesh':>8}{'maxDD':>8}{'bhDD':>8}{'ddcut':>7}")
    body = "\n".join(
        f"{tk:<8}{x.beta:>6.2f}{x.vol:>7.0%}{x.tim:>6.0%}{x.ann:>+8.1%}{x.bh_ann:>+8.1%}"
        f"{x.edge_ann:>+8.1%}{x.sh:>9.2f}{x.bh_sh:>7.2f}{x.edge_sh:>+8.2f}"
        f"{x.dd:>+8.1%}{x.bh_dd:>+8.1%}{x.dd_cut:>+7.1%}" for tk, x in df.iterrows())

    q = pd.qcut(df.beta, 3, labels=["low beta", "mid beta", "high beta"])
    g = df.groupby(q, observed=True)
    terc = g[["beta", "vol", "edge_ann", "edge_sh", "dd_cut", "tim"]].mean()
    terc["n"] = g.size()
    terc["pos_sh"] = g.edge_sh.apply(lambda s: (s > 0).sum())
    terc["beat_bh"] = g.beats_bh.sum().astype(int)

    lines = [f"{df.attrs['label']}   ({len(df)} names"
             + (f", skipped {' '.join(df.attrs['skipped'])}" if df.attrs["skipped"] else "") + ")",
             head, body, "",
             "  by beta tercile:", terc.round(3).to_string(), ""]
    for x in ("beta", "vol"):
        for m in ("edge_sh", "edge_ann", "dd_cut"):
            r = df[x].corr(df[m])
            t = r * np.sqrt((len(df) - 2) / max(1e-9, 1 - r ** 2))
            lines.append(f"  corr({x:<4}, {m:<9}) = {r:+.2f}   t = {t:+.2f}")
    # These names co-move hard, so the effective sample is far below len(df) and
    # the t's above are optimistic. Read them as direction, not as a p-value.
    lines.append(f"  edge_sh positive on {int((df.edge_sh > 0).sum())}/{len(df)}"
                 f"   mean {df.edge_sh.mean():+.2f}")
    return "\n".join(lines)


# Broad ETF cross-section. Names enter the test only once they have a full
# lookback of history, so late launches are handled point-in-time. What this
# CANNOT fix is ETFs that closed and are absent from the list entirely.
PIT_ETFS = BETA_ETFS + ["XBI", "GDX", "KRE", "XHB", "XRT", "XME", "ITB", "SMH", "ICLN",
                        "URA", "LIT", "COPX", "TLT", "GLD", "SLV", "IJR", "IJH", "VNQ",
                        "EWC", "EWA", "EWH", "EWS", "EWT", "EWM", "EPP", "ILF", "XOP",
                        "PBW", "IYT", "IHI", "IGE", "RSP", "MDY", "EZU", "IEV"]


def vol_terciles(tickers=None, years: int = 20, benchmark: str = "SPY", cost_bps: float = 1.0,
                 lookback: int = 252, rs_gate: bool = False, ema_gate: bool = False,
                 n_groups: int = 3) -> pd.DataFrame:
    """Rank the universe by TRAILING vol each January; hold each tercile for a year.

    Point-in-time on both legs: the vol ranking uses only returns strictly
    before the ranking date, and the stage signal is already lagged a week. No
    name is chosen with knowledge of how it did afterwards -- which is the whole
    difference between this and hand-picking NVDA because it trended.
    """
    import yfinance as yf

    warm = str((pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 2)).date())
    want = list(dict.fromkeys([*(tickers or PIT_ETFS), benchmark]))
    px = yf.download(want, start=warm, auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.tz_localize(None) if px.index.tz else px
    bench = weekly(px[benchmark])
    rets = px.pct_change()
    fwd = px.shift(-1) / px - 1.0

    pos = pd.DataFrame({tk: positions(px[tk].dropna(), bench, rs_gate, ema_gate=ema_gate)
                        for tk in px}).reindex(px.index)

    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    masks = {g: pd.DataFrame(False, index=px.index, columns=px.columns)
             for g in range(n_groups)}
    picks = []
    for y in range(start.year, px.index[-1].year + 1):
        rank_on = px.index[px.index < max(start, pd.Timestamp(f"{y}-01-01"))]
        if not len(rank_on):
            continue
        rank_on = rank_on[-1]
        hist = rets.loc[:rank_on].tail(lookback)
        ok = hist.notna().sum() >= lookback - 5                 # enough history AT the date
        v = (hist.std() * np.sqrt(252))[ok].dropna().drop(benchmark, errors="ignore")
        if len(v) < n_groups * 3:
            continue
        grp = pd.qcut(v.rank(method="first"), n_groups, labels=False)
        window = (px.index > rank_on) & (px.index <= pd.Timestamp(f"{y}-12-31")) & (px.index >= start)
        for g in range(n_groups):
            names = v.index[grp == g]
            masks[g].loc[window, names] = True
            picks.append({"year": y, "grp": g, "n": len(names), "vol": float(v[names].mean())})

    rows, series = [], {}
    for g in range(n_groups):
        m = masks[g] & pos.notna() & fwd.notna()
        w = pos.where(m, 0.0).div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        keep = m.any(axis=1)
        w, ew = w[keep], m.astype(float).div(m.sum(axis=1), axis=0).fillna(0.0)[keep]
        f = fwd[keep]
        expo = w.sum(axis=1).mean()
        for name, wt in (("sel", w), ("ctrl", ew * expo), ("full", ew)):
            r = (wt * f).sum(axis=1)
            to = wt.diff().abs().sum(axis=1).fillna(0.0)
            net = (r - to * cost_bps / 1e4).to_numpy()
            series[(g, name)] = pd.Series(net, index=f.index)
            rows.append({"grp": g, "leg": name, "days": len(net), "expo": expo,
                         "ann": _ann(net, len(net)), "sh": _sharpe(net), "dd": _max_dd(net),
                         "to": float(to.sum() / len(to) * 252)})
    out = pd.DataFrame(rows).pivot(index="grp", columns="leg")
    p = pd.DataFrame(picks).groupby("grp").agg(names=("n", "mean"), tvol=("vol", "mean"))
    out[("meta", "tvol")], out[("meta", "names")] = p.tvol, p.names
    out.attrs["series"] = series  # net daily returns per (tercile, leg), for further tests
    return out


def vol_report(d: pd.DataFrame, n_groups: int = 3) -> str:
    lab = ["low vol", "mid vol", "high vol"] if n_groups == 3 else [f"q{i}" for i in range(n_groups)]
    head = (f"{'tercile':<10}{'tvol':>7}{'names':>7}{'expo':>6}{'SEL':>8}{'CTRL':>8}{'FULL':>8}"
            f"{'SELsh':>7}{'CTRLsh':>8}{'FULLsh':>8}{'edgesh':>8}{'SELdd':>8}{'FULLdd':>8}")
    body = []
    for g, r in d.iterrows():
        body.append(
            f"{lab[g]:<10}{r[('meta','tvol')]:>7.0%}{r[('meta','names')]:>7.0f}"
            f"{r[('expo','sel')]:>6.0%}{r[('ann','sel')]:>+8.1%}{r[('ann','ctrl')]:>+8.1%}"
            f"{r[('ann','full')]:>+8.1%}{r[('sh','sel')]:>7.2f}{r[('sh','ctrl')]:>8.2f}"
            f"{r[('sh','full')]:>8.2f}{r[('sh','sel')] - r[('sh','ctrl')]:>+8.2f}"
            f"{r[('dd','sel')]:>+8.1%}{r[('dd','full')]:>+8.1%}")
    return head + "\n" + "\n".join(body)
