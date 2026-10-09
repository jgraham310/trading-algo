"""Weekly Bollinger Band (20, 2) screen over all NYSE + NASDAQ common stocks.

A hit is a weekly close above the upper band or below the lower band, with the
bands computed from the last 20 weekly closes INCLUDING the current one (the
standard definition) and population std (ddof=0, what charting packages use).
Run after Friday's close; a holiday Friday falls back to that week's last bar.

Floors (default tier): last close >= $5 and 20-week average daily dollar
volume >= $5M. Pure research screen -- not a trade signal.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trading_algo.flag import download, weekly

SYMDIR = "https://www.nasdaqtrader.com/dynamic/SymDir/"
# Not common stock. ponytail: name regex, ceiling is the odd oddly-named issue.
NOT_COMMON = re.compile(r"\b(?:Warrants?|Rights?|Units?|Preferred|Notes?|Debentures?)\b", re.I)


@dataclass(frozen=True)
class Params:
    window: int = 20
    k: float = 2.0
    min_price: float = 5.0
    min_dollar_vol: float = 5e6   # average DAILY $ volume over `window` weeks


def universe() -> list[str]:
    """NASDAQ + NYSE common stocks, Yahoo-style tickers, from NASDAQ Trader's symbol files."""
    def fetch(name):
        raw = urllib.request.urlopen(urllib.request.Request(SYMDIR + name, headers={"User-Agent": "Mozilla/5.0"}),
                                     timeout=30).read().decode()
        return pd.read_csv(io.StringIO(raw), sep="|", dtype=str,
                           keep_default_na=False).iloc[:-1]  # "NA" is a real ticker, not a null  # last row = file timestamp

    nq = fetch("nasdaqlisted.txt")
    nq = nq[(nq["Test Issue"] == "N") & (nq["ETF"] == "N") & nq["Financial Status"].isin(["N", ""])]
    ot = fetch("otherlisted.txt")
    ot = ot[(ot["Exchange"] == "N") & (ot["Test Issue"] == "N") & (ot["ETF"] == "N")]  # N = NYSE
    syms = pd.concat([nq[["Symbol", "Security Name"]],
                      ot.rename(columns={"ACT Symbol": "Symbol"})[["Symbol", "Security Name"]]])
    syms = syms[~syms["Security Name"].str.contains(NOT_COMMON) & ~syms["Symbol"].str.contains(r"\$")]
    return sorted({s.replace(".", "-") for s in syms["Symbol"]})


def bands(w: pd.DataFrame, p: Params = Params()) -> dict | None:
    """Bands as of the LAST row of weekly OHLCV `w`, or None if it fails the floors/history."""
    if len(w) < p.window:
        return None
    c = w["Close"].iloc[-p.window:]
    last = c.iloc[-1]
    daily_dollar = (w["Volume"].iloc[-p.window:] * c).mean() / 5
    if last < p.min_price or not daily_dollar >= p.min_dollar_vol:
        return None
    mid, sd = c.mean(), c.std(ddof=0)
    lo, hi = mid - p.k * sd, mid + p.k * sd
    if not hi > lo or (lo <= last <= hi):
        return None
    side = "above" if last > hi else "below"
    return {"week": w.index[-1].date().isoformat(), "side": side,
            "close": round(last, 2), "upper": round(hi, 2), "mid": round(mid, 2), "lower": round(lo, 2),
            "pct_b": round((last - lo) / (hi - lo), 3),    # >1 above upper, <0 below lower
            "vs_band_pct": round((last / (hi if last > hi else lo) - 1) * 100, 1),
            "avg_daily_dollar_vol_m": round(daily_dollar / 1e6, 1),
            "bandwidth": round((hi - lo) / mid, 3),        # band width / mid: how volatile the last 20 weeks were
            "ret_1w": round(last / w["Close"].iloc[-2] - 1, 3),
            "streak": _streak(w["Close"], side, p)}        # consecutive weeks outside on this side, incl. this one


def _streak(close: pd.Series, side: str, p: Params, cap: int = 26) -> int:
    n = 0
    for e in range(len(close), p.window - 1, -1):
        if n >= cap:
            break
        c = close.iloc[e - p.window:e]
        mid, sd = c.mean(), c.std(ddof=0)
        if not (c.iloc[-1] > mid + p.k * sd if side == "above" else c.iloc[-1] < mid - p.k * sd):
            break
        n += 1
    return n


def scan(data: dict[str, pd.DataFrame], p: Params = Params(), asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Screen every ticker as of its latest bar (or `asof`); stale (halted/delisted) names are dropped."""
    if asof is not None:
        data = {tk: d.loc[:asof] for tk, d in data.items()}
    data = {tk: d for tk, d in data.items() if len(d)}
    latest = max(d.index[-1] for d in data.values())
    rows = []
    for tk, d in data.items():
        if d.index[-1] < latest:
            continue
        hit = bands(weekly(d), p)
        if hit:
            rows.append({"ticker": tk, **hit})
    cols = ["ticker", "week", "side", "close", "upper", "mid", "lower", "pct_b", "vs_band_pct",
            "avg_daily_dollar_vol_m", "bandwidth", "ret_1w", "streak"]
    out = pd.DataFrame(rows, columns=cols)
    # Furthest outside first within each side.
    out["_rank"] = out["pct_b"].where(out["side"] == "below", -out["pct_b"])
    out = out.sort_values(["side", "_rank"], ascending=[True, True], ignore_index=True).drop(columns="_rank")
    return out


def render(hits: pd.DataFrame, n_scanned: int, friday: str, top: int = 100) -> str:
    """Markdown: two sections, furthest outside first, capped at `top` rows each (CSV has all)."""
    above, below = hits[hits.side == "above"], hits[hits.side == "below"]
    out = [f"**Weekly Bollinger Bands (20, 2) — week ending {friday}** · "
           f"{len(above)} above, {len(below)} below, of {n_scanned} scanned\n"]
    cols = ["ticker", "close", "lower", "upper", "pct_b", "vs_band_pct", "avg_daily_dollar_vol_m"]
    for title, t in (("Closed above the upper band", above), ("Closed below the lower band", below)):
        out.append(f"\n### {title} ({len(t)})\n")
        if t.empty:
            out.append("None.\n")
            continue
        out += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.head(top).iterrows()]
        if len(t) > top:
            out.append(f"\n_Top {top} shown; full list in the CSV._")
    out.append("\n`pct_b` = (close − lower)/(upper − lower); `vs_band_pct` = % beyond the breached band. "
               "Floors: close ≥ $5, avg daily $ volume ≥ $5M.\n")
    return "\n".join(out)


HORIZONS = (1, 2, 4, 8)   # weeks ahead
FWD = [f"fwd_{n}w" for n in HORIZONS] + [f"spy_fwd_{n}w" for n in HORIZONS]


def update_history(path: str | Path, hits: pd.DataFrame, weeks: list[str], data: dict[str, pd.DataFrame],
                   spy: pd.DataFrame | None) -> pd.DataFrame:
    """Upsert `hits` for `weeks` into the CSV at `path`, then fill any forward returns now knowable.

    Forward return = weekly close n weeks later / close at the signal week, both from the SAME freshly
    downloaded adjusted series (stored `close` is not used, dividends would skew it). Blank = not yet
    known, or the ticker has since vanished (delisted names drop out: survivorship bias on old rows).
    ponytail: the last weekly bar is trusted to be complete, so only write history from post-close runs.
    """
    path = Path(path)
    old = pd.read_csv(path, dtype={"ticker": str}, keep_default_na=False, na_values=[""]) if path.exists() \
        else pd.DataFrame()
    h = pd.concat([old[~old["week"].isin(weeks)] if len(old) else old, hits], ignore_index=True)
    for c in FWD:
        if c not in h:
            h[c] = float("nan")
    closes = {}

    def wclose(tk):
        if tk not in closes:
            d = spy if tk == "SPY" else data.get(tk)
            closes[tk] = weekly(d)["Close"] if d is not None and len(d) else None
        return closes[tk]

    for i in h.index[h[FWD].isna().any(axis=1)]:
        t0 = pd.Timestamp(h.at[i, "week"])
        for prefix, tk in (("", h.at[i, "ticker"]), ("spy_", "SPY")):
            s = wclose(tk)
            if s is None or t0 not in s.index:
                continue
            pos = s.index.get_loc(t0)
            for n in HORIZONS:
                col = f"{prefix}fwd_{n}w"
                if pd.isna(h.at[i, col]) and pos + n < len(s):
                    h.at[i, col] = round(s.iloc[pos + n] / s.iloc[pos] - 1, 4)
    h = h.sort_values(["week", "side", "ticker"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(path, index=False)
    return h


def week_complete(friday: date, now: datetime) -> bool:
    """Is the W-FRI week labelled `friday` over (today's close has settled if it is that Friday)?"""
    return friday < now.date() or (friday == now.date() and now.hour * 60 + now.minute >= 16 * 60 + 15)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tickers", nargs="*", help="default: all NYSE + NASDAQ common stocks")
    ap.add_argument("--asof", help="screen as of a past date (YYYY-MM-DD)")
    ap.add_argument("--out", help="write the markdown report here")
    ap.add_argument("--csv", help="write hits as CSV here")
    ap.add_argument("--history", help="upsert hits into this running CSV (use after the close) and fill forward returns")
    ap.add_argument("--backfill", type=int, metavar="N",
                    help="with --history: also seed the last N past weeks (max ~30) from the same download")
    a = ap.parse_args(argv)

    asof = pd.Timestamp(a.asof) if a.asof else None
    if a.history and asof is not None and asof.weekday() != 4:
        raise SystemExit("--history with --asof needs a Friday (a mid-week asof would store a partial week)")
    tickers = [t.upper() for t in a.tickers] or universe()
    data = download(tickers + ["SPY"], period="1y")
    spy = data.pop("SPY", None)
    if len(data) < 0.8 * len(tickers):
        # Don't send "nothing outside the bands" when the truth is "no data".
        raise SystemExit(f"only {len(data)}/{len(tickers)} tickers downloaded -- aborting")
    if asof is not None:   # behave as if run then: no later (or partial) bars may leak into forward returns
        data = {tk: d.loc[:asof] for tk, d in data.items()}
        data = {tk: d for tk, d in data.items() if len(d)}
        spy = spy.loc[:asof]
    hits = scan(data)
    week = asof or max(d.index[-1] for d in data.values())
    friday = week.to_period("W-FRI").end_time.date().isoformat()
    if a.history and not week_complete(date.fromisoformat(friday), datetime.now(ZoneInfo("America/New_York"))):
        # ponytail: calendar rule only; a Friday holiday is fine (the label is still that Friday).
        raise SystemExit(f"week ending {friday} is not over (needs ~4:15 PM ET on its last session) "
                         "-- refusing to store a partial week in the history")
    body = render(hits, len(data), friday)
    print(body)
    if a.out:
        Path(a.out).write_text(body)
    if a.csv:
        hits.to_csv(a.csv, index=False)
    if a.history:
        past = []
        if a.backfill:
            fridays = [f for f in weekly(spy).index if f < pd.Timestamp(friday)][-a.backfill:]
            past = [scan(data, asof=f) for f in fridays]
        allhits = pd.concat([*past, hits], ignore_index=True)
        weeks = [f.date().isoformat() for f in fridays] + [friday] if a.backfill else [friday]
        h = update_history(a.history, allhits, weeks, data, spy)
        print(f"history: {len(h)} rows across {h.week.nunique()} weeks -> {a.history}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"above={(hits.side == 'above').sum()}\nbelow={(hits.side == 'below').sum()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
