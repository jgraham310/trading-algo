"""Weekly bull-flag breakout screener.

A hit at week t needs all four, using only bars up to and including t:

  pole    a run from a swing low to the highest high of the setup, gaining at
          least `min_pole` (default 20%) in 1-6 weeks.
  flag    the 2-6 weeks after the pole top, no longer than the pole: no new high, the deepest
          low gives back at most `max_retrace` of the pole, and closes drift
          sideways-to-down (a rising channel is not a flag).
  break   week t closes above every high printed during the flag.
  volume  week t's volume >= `min_vol` x the average of the prior `vol_avg`
          weeks (10 weeks ~ the 50-day average).

Run intraday on a Friday, week t is the PARTIAL current week. Its volume is
missing the last hour (the heaviest of the day, closing auction included), so
the volume test is conservative: a name that clears 1.0x at 3 PM clears it at
the close. Names just under will be missed -- lower `min_vol` if you'd rather
see them. The breakout itself can also fail by the close; that is the price of
alerting before it.

Nothing here is a tested edge. It finds the pattern; whether the pattern pays
is a separate question (see README).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

UNIVERSE = Path(__file__).with_name("universe.txt")


@dataclass(frozen=True)
class Params:
    min_pole: float = 0.20        # pole gain, low -> top
    pole_weeks: tuple = (1, 6)    # weeks from pole low to pole top -- a pole is steep
    flag_weeks: tuple = (2, 6)    # weeks of consolidation after the top, and never
                                  # longer than the pole: past that it's a base
    max_retrace: float = 0.50     # of the pole's height
    max_flag_slope: float = 0.005  # closes' regression slope, fraction of price per week
    vol_avg: int = 10             # weeks in the volume average (excludes week t)
    min_vol: float = 1.0          # "average or better"
    min_price: float = 5.0
    min_dollar_vol: float = 5e6   # avg weekly $ volume floor -- skip the illiquid


def weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV -> W-FRI OHLCV. The current week is whatever has printed so far."""
    w = daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                     "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Close"])


def detect(w: pd.DataFrame, p: Params = Params()) -> dict | None:
    """Does the LAST row of weekly OHLCV `w` break out of a bull flag?

    Returns the setup (pole, flag, breakout level, volume ratio) or None.
    Every flag length is tried and the first valid one (shortest) is returned.
    """
    if len(w) < p.vol_avg + p.pole_weeks[1] + p.flag_weeks[1] + 1:
        return None
    hi, lo, cl, vol = (w[c].to_numpy(float) for c in ("High", "Low", "Close", "Volume"))
    t = len(w) - 1

    avg_vol = vol[t - p.vol_avg:t].mean()
    if not avg_vol > 0 or cl[t] < p.min_price or avg_vol * cl[t] < p.min_dollar_vol:
        return None
    vol_ratio = vol[t] / avg_vol
    if vol_ratio < p.min_vol:
        return None

    for n in range(p.flag_weeks[0], p.flag_weeks[1] + 1):
        top = t - n - 1                       # pole top week; flag is top+1 .. t-1
        flag = slice(top + 1, t)
        if hi[flag].max() >= hi[top]:         # flag made a new high: top is not the top
            continue
        resistance = hi[flag].max()
        if cl[t] <= resistance:
            continue
        # Pole: lowest low in the pole window before the top, and it must be a
        # real run -- the top is the high of everything from the low onward.
        lo_win = slice(top - p.pole_weeks[1], top - p.pole_weeks[0] + 1)
        w_lo = lo[lo_win]                     # LAST lowest low: ties must not stretch the pole
        base = lo_win.stop - 1 - int(np.argmin(w_lo[::-1]))
        pole = hi[top] - lo[base]
        if lo[base] <= 0 or hi[top] / lo[base] - 1 < p.min_pole or n > top - base:
            continue
        if hi[base:top].max(initial=-np.inf) > hi[top]:
            continue
        flag_low = lo[flag].min()
        retrace = (hi[top] - flag_low) / pole
        if retrace > p.max_retrace:
            continue
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, cl[flag], 1)[0] / cl[top] if n > 1 else 0.0
        if slope > p.max_flag_slope:
            continue
        return {
            "week": w.index[t].date().isoformat(),
            "close": round(cl[t], 2),
            "breakout_level": round(resistance, 2),
            "above_pole_high": bool(cl[t] > hi[top]),
            "pole_gain": round(hi[top] / lo[base] - 1, 3),
            "pole_weeks": top - base,
            "flag_weeks": n,
            "retrace": round(retrace, 3),
            "flag_vol_vs_pole": round(vol[flag].mean() / vol[base:top + 1].mean(), 2),
            "vol_ratio": round(vol_ratio, 2),
            "stop": round(flag_low, 2),        # below the flag = pattern failed
        }
    return None


def download(tickers: list[str], period: str = "2y", chunk: int = 200) -> dict[str, pd.DataFrame]:
    """Daily OHLCV per ticker. Today's row is the live, partial session."""
    import yfinance as yf

    out = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        df = yf.download(batch, period=period, interval="1d", auto_adjust=True,
                         progress=False, group_by="ticker", threads=True)
        for tk in batch:
            if tk not in df.columns.get_level_values(0):
                continue
            d = df[tk].dropna(subset=["Close"])
            if len(d):
                out[tk] = d.tz_localize(None) if d.index.tz else d
    return out


def scan(data: dict[str, pd.DataFrame], p: Params = Params(),
         asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Screen every ticker as of its latest bar (or as of `asof`).

    Tickers whose last bar is older than the freshest bar in the set are
    dropped: a halted or delisted name's stale "current week" is not this week.
    """
    latest = max(d.index[-1] for d in data.values())
    rows = []
    for tk, d in data.items():
        if asof is not None:
            d = d.loc[:asof]
        elif d.index[-1] < latest:
            continue
        if not len(d):
            continue
        hit = detect(weekly(d), p)
        if hit:
            rows.append({"ticker": tk, **hit})
    cols = ["ticker", "week", "close", "breakout_level", "above_pole_high", "pole_gain",
            "pole_weeks", "flag_weeks", "retrace", "flag_vol_vs_pole", "vol_ratio", "stop"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("vol_ratio", ascending=False, ignore_index=True)


def load_universe(path: str | Path = UNIVERSE) -> list[str]:
    lines = Path(path).read_text().split()
    return [t.strip().upper() for t in lines if t.strip() and not t.startswith("#")]


def refresh_universe(path: str | Path = UNIVERSE) -> int:
    """S&P 500 + 400 + 600 from Wikipedia, Yahoo-style tickers (BRK.B -> BRK-B)."""
    import io
    import urllib.request

    syms = set()
    for idx in ("500", "400", "600"):
        url = f"https://en.wikipedia.org/wiki/List_of_S%26P_{idx}_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=30).read().decode()
        syms |= set(pd.read_html(io.StringIO(html))[0]["Symbol"].astype(str))
    syms = sorted(s.strip().replace(".", "-") for s in syms)
    Path(path).write_text("\n".join(syms) + "\n")
    return len(syms)


def render(hits: pd.DataFrame, n_scanned: int, asof: str, partial: bool = True) -> str:
    """Markdown body for the alert."""
    head = f"**Weekly bull-flag breakouts — week ending {asof}** · {len(hits)} hit(s) of {n_scanned} scanned\n"
    if hits.empty:
        return head + "\nNo breakouts this week.\n"
    t = hits.assign(pole_gain=(hits.pole_gain * 100).round(0).astype(int).astype(str) + "%",
                    retrace=(hits.retrace * 100).round(0).astype(int).astype(str) + "%",
                    vol_ratio=hits.vol_ratio.map("{:.2f}x".format))
    cols = ["ticker", "close", "breakout_level", "stop", "vol_ratio", "pole_gain",
            "pole_weeks", "flag_weeks", "retrace", "above_pole_high"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.iterrows()]
    note = "\n`stop` = flag low."
    if partial:
        note += " Partial week: close and volume are as of the run, not the Friday close."
    return head + "\n" + "\n".join(lines) + "\n" + note + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tickers", nargs="*", help="default: the bundled universe")
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--min-pole", type=float, default=Params.min_pole)
    ap.add_argument("--min-vol", type=float, default=Params.min_vol)
    ap.add_argument("--asof", help="screen as of a past date (YYYY-MM-DD)")
    ap.add_argument("--out", help="write the markdown report here")
    ap.add_argument("--csv", help="write hits as CSV here")
    ap.add_argument("--refresh-universe", action="store_true")
    a = ap.parse_args(argv)

    if a.refresh_universe:
        print(f"{refresh_universe(a.universe)} tickers -> {a.universe}")
        return 0

    tickers = [t.upper() for t in a.tickers] or load_universe(a.universe)
    data = download(tickers)
    if len(data) < 0.8 * len(tickers):
        # Don't send "no breakouts" when the truth is "no data".
        raise SystemExit(f"only {len(data)}/{len(tickers)} tickers downloaded -- aborting")
    p = Params(min_pole=a.min_pole, min_vol=a.min_vol)
    asof = pd.Timestamp(a.asof) if a.asof else None
    hits = scan(data, p, asof)
    week = asof or max(d.index[-1] for d in data.values())
    friday = week.to_period("W-FRI").end_time.date().isoformat()
    body = render(hits, len(data), friday, partial=asof is None)
    print(body)
    if a.out:
        Path(a.out).write_text(body)
    if a.csv:
        hits.to_csv(a.csv, index=False)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"hits={len(hits)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
