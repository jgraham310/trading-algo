"""Weekly Bollinger screen on synthetic weekly bars -- no download."""

import numpy as np
import pandas as pd

from trading_algo.bband import Params, bands, render, scan, week_complete


def bars(closes, vol=1e6):
    c = np.array(closes, float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(len(c), vol)},
                        index=pd.date_range("2024-01-05", periods=len(c), freq="W-FRI"))


FLAT = [50.0, 51.0] * 10  # 20 weeks, mean 50.5, std 0.5 -> bands 49.5..51.5 once the last close joins


def test_above_below_inside():
    up = bands(bars(FLAT[:-1] + [60.0]))
    assert up and up["side"] == "above" and up["pct_b"] > 1
    dn = bands(bars(FLAT[:-1] + [40.0]))
    assert dn and dn["side"] == "below" and dn["pct_b"] < 0
    assert bands(bars(FLAT)) is None  # close 51 is inside the bands


def test_population_std_and_window():
    w = bars([10.0] * 19 + [20.0])  # window includes the last close; ddof=0
    c = w["Close"]
    assert bands(w, Params(min_price=0, min_dollar_vol=0))["upper"] == round(c.mean() + 2 * c.std(ddof=0), 2)


def test_floors_and_short_history():
    assert bands(bars(FLAT[:-1] + [3.0] * 1), Params(min_price=5)) is None   # under $5
    assert bands(bars(FLAT[:-1] + [60.0], vol=1e3)) is None                  # illiquid
    assert bands(bars((FLAT[:-1] + [60.0])[-19:])) is None                   # < 20 weeks


def test_scan_drops_stale_and_sorts_by_distance():
    data = {"UP1": bars(FLAT[:-1] + [60.0]), "UP2": bars(FLAT[:-1] + [70.0]),
            "DN": bars(FLAT[:-1] + [40.0]), "OLD": bars(FLAT[:-1] + [60.0])[:-1]}
    hits = scan(data)
    assert list(hits.ticker) == ["UP2", "UP1", "DN"]  # OLD is stale; furthest-out first
    md = render(hits, 4, "2024-05-24")
    assert md.index("above the upper") < md.index("below the lower")


def test_streak_counts_consecutive_weeks_outside():
    assert bands(bars(FLAT + [60.0]))["streak"] == 1
    assert bands(bars(FLAT + [60.0, 70.0]))["streak"] == 2


def test_history_upsert_and_forward_fill(tmp_path):
    from trading_algo.bband import update_history

    full = bars(FLAT[:-1] + [60.0, 61.0, 62.0, 63.0, 64.0])   # signal at week index 19, 5 weeks of future
    data, spy = {"UP": full}, bars([100.0] * 25)
    wk = full.index[19]
    hits = scan(data, asof=wk)
    path = tmp_path / "h.csv"
    h = update_history(path, hits, [wk.date().isoformat()], data, spy)
    r = h.iloc[0]
    assert len(h) == 1 and r["fwd_1w"] == round(61 / 60 - 1, 4) and r["fwd_4w"] == round(64 / 60 - 1, 4)
    assert pd.isna(r["fwd_8w"]) and r["spy_fwd_1w"] == 0
    # Re-running the same week replaces, not duplicates; a stored fwd value survives a later run.
    h2 = update_history(path, hits, [wk.date().isoformat()], data, spy)
    assert len(h2) == 1 and h2.iloc[0]["fwd_4w"] == r["fwd_4w"]


def test_week_complete_guard():
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    fri = date(2026, 10, 9)
    assert not week_complete(fri, datetime(2026, 10, 9, 12, 30, tzinfo=ny))   # Friday midday
    assert not week_complete(fri, datetime(2026, 10, 7, 20, 0, tzinfo=ny))    # Wednesday night
    assert week_complete(fri, datetime(2026, 10, 9, 17, 30, tzinfo=ny))       # Friday after close
    assert week_complete(fri, datetime(2026, 10, 10, 9, 0, tzinfo=ny))        # Saturday


def test_event_price_tags_separate_gap_from_grind():
    from trading_algo.events import price_tags

    idx = pd.bdate_range("2024-01-01", periods=60)             # ends Fri 2024-03-22
    friday = idx[-1]

    def daily(closes, gap_day=None, vol=None):
        c = np.array(closes, float)
        o = np.r_[c[0], c[:-1]]
        if gap_day is not None:
            o[gap_day] = c[gap_day - 1] * 1.15                 # 15% overnight gap
        v = np.full(len(c), 1e6) if vol is None else vol
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.005, "Low": np.minimum(o, c) * 0.995,
                             "Close": c, "Volume": v}, index=idx)

    base = np.full(55, 100.0)
    grind = daily(np.r_[base, 101, 102, 103, 104, 105])
    gap = daily(np.r_[base, 100, 100, 100, 100, 115], gap_day=59, vol=np.r_[np.full(59, 1e6), 5e6])
    spy = daily(np.tile([100.0, 101.0], 30))
    g, e = price_tags(grind, spy, friday), price_tags(gap, spy, friday)
    assert e["gap_atr"] > 5 * g["gap_atr"] and e["day_conc"] > 0.9 > g["day_conc"] and e["vol_spike"] == 5.0
    assert g["resid_1w"] == 0.05                               # stock never co-moved with SPY (beta 0) -> all +5% is idiosyncratic


def test_filing_tags_window_and_unknown():
    from trading_algo.events import tag

    class Fake:
        def filings(self, tk):
            if tk == "NOCIK":
                return None
            return pd.DataFrame({"date": pd.to_datetime(["2024-03-12", "2024-03-01"]),
                                 "form": ["8-K", "8-K"], "items": ["2.02,9.01", "1.01"]})

    h = pd.DataFrame({"week": ["2024-03-15", "2024-03-15"], "ticker": ["A", "NOCIK"]})
    out = tag(h, {}, None, edgar=Fake())
    a, n = out.iloc[0], out.iloc[1]
    assert a.filing_event == 1 and a.earnings_event == 1 and a.filing_items == "2.02;9.01"  # 03-01 is outside 14d
    assert pd.isna(n.filing_event)                                                          # unknown, not 0


def test_daily_forward_returns_and_mae():
    from trading_algo.bband import _daily_fwd

    idx = pd.bdate_range("2024-03-18", periods=7)              # Mon..Fri, then Mon, Tue
    d = pd.DataFrame({"Open": [10, 10, 10, 10, 10, 11.0, 11.0], "High": 12.0, "Close": [10, 10, 10, 10, 10, 12.0, 13.0],
                      "Low": [10, 10, 10, 10, 10, 10.5, 9.9], "Volume": 1}, index=idx)
    r = _daily_fwd(d, idx[4])                                  # signal Friday close 10; Mon open 11, close 12
    assert r["fwd_1d"] == 0.2 and r["fwd_2d"] == 0.3           # close-to-close from Friday's 10
    assert r["ofwd_1d"] == round(12 / 11 - 1, 4) and r["ofwd_2d"] == round(13 / 11 - 1, 4)
    assert "fwd_3d" not in r and "mae_3d" not in r             # only 2 sessions exist yet -> not knowable
