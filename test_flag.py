"""Bull-flag detector on synthetic weekly bars -- no download."""

import numpy as np
import pandas as pd

from trading_algo.flag import Params, detect, render, scan, weekly

# 20 flat weeks, 4-week pole +40%, 3-week drifting flag, then the breakout week.
BASE = [50.0] * 20
POLE = [55.0, 60.0, 65.0, 70.0]
FLAG = [67.0, 65.0, 64.0]


def bars(closes, vols=None, breakout_close=72.0, breakout_vol=1.5e6):
    c = np.array(closes + [breakout_close])
    v = np.append(vols if vols is not None else [1e6] * len(closes), breakout_vol)
    hi, lo = c * 1.01, c * 0.99
    return pd.DataFrame({"Open": c, "High": hi, "Low": lo, "Close": c, "Volume": v},
                        index=pd.date_range("2024-01-05", periods=len(c), freq="W-FRI"))


def test_textbook_flag_breaks_out():
    hit = detect(bars(BASE + POLE + FLAG))
    assert hit is not None
    assert hit["flag_weeks"] == 3 and hit["pole_weeks"] >= 4
    assert hit["breakout_level"] == round(67.0 * 1.01, 2)  # highest flag high
    assert hit["stop"] == round(64.0 * 0.99, 2)            # flag low
    assert hit["vol_ratio"] == 1.5 and hit["above_pole_high"]


def test_below_average_volume_is_rejected():
    assert detect(bars(BASE + POLE + FLAG, breakout_vol=0.99e6)) is None
    assert detect(bars(BASE + POLE + FLAG, breakout_vol=1.0e6)) is not None  # "average" counts


def test_no_breakout_yet():
    assert detect(bars(BASE + POLE + FLAG, breakout_close=66.0)) is None


def test_deep_retrace_is_not_a_flag():
    # gives back ~80% of a 50 -> 70 pole
    assert detect(bars(BASE + POLE + [62.0, 56.0, 54.0], breakout_close=63.0)) is None


def test_no_pole_no_flag():
    # a +10% drift is not a pole
    assert detect(bars(BASE + [51.0, 52.0, 53.0, 55.0] + [54.0, 53.5, 53.0],
                       breakout_close=56.0)) is None


def test_rising_channel_is_not_a_flag():
    # "flag" weeks that keep climbing without a new high above the top are
    # still a rising channel once the slope is meaningful
    top = BASE + POLE[:-1] + [75.0]
    assert detect(bars(top + [66.0, 69.0, 72.0], breakout_close=76.0)) is None


def test_flag_longer_than_pole_is_a_base():
    # 1-week pole (gap), then 4 weeks of drift: too long relative to the pole
    gap = BASE + [70.0]
    assert detect(bars(gap + [68.0, 67.0, 66.0, 65.0], breakout_close=72.0)) is None
    assert detect(bars(gap + [68.0], breakout_close=72.0)) is None  # flag < 2 weeks
    assert detect(bars(BASE + [60.0, 70.0] + [68.0, 66.0], breakout_close=72.0)) is not None


def test_detect_only_reads_the_last_bar():
    """Appending future weeks must not change a past verdict (no lookahead)."""
    w = bars(BASE + POLE + FLAG)
    later = pd.concat([w, bars([10.0] * 5).set_index(
        pd.date_range(w.index[-1] + pd.Timedelta(weeks=1), periods=6, freq="W-FRI"))])
    assert detect(later.iloc[: len(w)]) == detect(w)


def test_weekly_resample_keeps_the_partial_week():
    d = pd.DataFrame({"Open": 1.0, "High": [2.0, 3.0, 2.5], "Low": [0.5, 0.4, 0.6],
                      "Close": [1.5, 2.5, 2.0], "Volume": [10, 20, 30]},
                     index=pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-15"]))
    w = weekly(d)
    assert list(w.index.strftime("%F")) == ["2024-01-12", "2024-01-19"]
    assert w.iloc[0].tolist() == [1.0, 3.0, 0.4, 2.5, 30]
    assert w.iloc[1]["Volume"] == 30  # Monday-only week is still a row


def test_scan_drops_stale_tickers():
    # one bar per Friday resamples to itself, so the weekly fixture doubles as daily
    fresh = bars(BASE + POLE + FLAG)
    stale = fresh.set_axis(fresh.index - pd.Timedelta(days=7))  # stopped trading a week ago
    assert scan({"NEW": fresh, "OLD": stale})["ticker"].tolist() == ["NEW"]


def test_render():
    assert "No breakouts" in render(scan({"X": bars(BASE * 2)}), 1, "2024-06-07")
    body = render(scan({"FLAG": bars(BASE + POLE + FLAG)}), 1, "2024-06-07")
    assert "| FLAG |" in body and "1.50x" in body


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
