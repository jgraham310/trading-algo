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
