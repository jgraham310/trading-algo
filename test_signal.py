"""Money-path checks. No model download: forecast() is stubbed."""

import numpy as np

from trading_algo import timesfm_signal as ts


def fake(last, end_q):
    """Forecast whose horizon-end quantiles are `end_q` (9 values, price space)."""
    end_q = np.asarray(end_q, dtype=float)
    return ts.Forecast(last=last, path=np.array([end_q[4]]), quantiles=end_q[None, :])


def test():
    # q0.1 .. q0.9 all above last -> confident long, +2% er at scale 2% = full size
    f = fake(100.0, [100.5, 100.8, 101, 101.4, 102, 102.6, 103, 103.5, 104])
    assert abs(f.expected_return - 0.02) < 1e-9
    assert ts.signal(None, _forecast=f) == 1.0

    # same median, but q0.3 below last -> not confident -> flat
    f = fake(100.0, [95, 96, 99.5, 101.4, 102, 102.6, 103, 103.5, 104])
    assert f.expected_return > 0
    assert ts.signal(None, _forecast=f) == 0.0

    # confident short
    f = fake(100.0, [96, 96.5, 97, 97.4, 98, 98.6, 99, 99.3, 99.5])
    assert ts.signal(None, _forecast=f) == -1.0

    # short with q0.7 above last -> flat
    f = fake(100.0, [96, 96.5, 97, 97.4, 98, 98.6, 100.5, 103, 105])
    assert f.expected_return < 0
    assert ts.signal(None, _forecast=f) == 0.0

    # sizing scales with edge and clips
    f = fake(100.0, [100.1] * 4 + [101.0] + [101.5] * 4)
    assert abs(ts.signal(None, _forecast=f) - 0.5) < 1e-9  # 1% / 2% scale
    f = fake(100.0, [100.1] * 4 + [110.0] + [111.0] * 4)
    assert ts.signal(None, _forecast=f) == 1.0  # 10% edge clips at 1

    # quantile lookup picks the nearest declared quantile
    f = fake(100.0, [91, 92, 93, 94, 95, 96, 97, 98, 99])
    assert abs(f.quantile_return(0.1) - -0.09) < 1e-9
    assert abs(f.quantile_return(0.9) - -0.01) < 1e-9

    # no look-ahead: signal passes through exactly the bars it was given,
    # and the position it returns is for the bar AFTER prices[-1].
    px = list(np.linspace(100.0, 110.0, 200))
    seen = {}
    real = ts.forecast

    def spy(p, **k):
        p = np.asarray(p).ravel()
        seen["n"] = len(p)
        seen["last"] = p[-1]
        return fake(float(p[-1]), [float(p[-1])] * 9)

    ts.forecast = spy
    try:
        ts.signal(px)
    finally:
        ts.forecast = real
    assert seen["n"] == len(px), seen
    assert seen["last"] == px[-1], seen

    # too-short context is rejected, not silently padded
    try:
        real([1.0] * 31)
    except ValueError:
        pass
    else:
        raise AssertionError("short context should raise")

    print("ok")


if __name__ == "__main__":
    test()
