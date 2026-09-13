"""Leakage checks. These are the ones worth having."""

import numpy as np
import pandas as pd

from trading_algo import backtest as bt
from trading_algo.sources import align, calendar, future_index
from test_backtest import stub

IDX = pd.bdate_range("2024-01-01", periods=300)


def test_publication_lag_hides_the_value():
    # A print stamped Monday but published 3 days later must not exist before then.
    macro = pd.Series([7.0], index=pd.to_datetime(["2024-01-08"]))
    col = align(IDX, m=(macro, 3))["m"]
    assert col.loc[: "2024-01-10"].isna().all(), "value visible before publication"
    assert col.loc["2024-01-11"] == 7.0 and col.loc["2024-02-01"] == 7.0  # then held


def test_lag_zero_is_same_bar():
    vix = pd.Series(range(300), index=IDX, dtype=float)
    assert align(IDX, v=vix)["v"].equals(vix.rename("v"))


def test_align_never_reads_right():
    """Every aligned value must equal the last source value at or before that bar."""
    src = pd.Series(np.arange(60.0), index=pd.bdate_range("2024-01-01", freq="5D", periods=60))
    got = align(IDX, s=(src, 0))["s"]
    for t, v in got.dropna().items():
        assert v == src[src.index <= t].iloc[-1], f"{t}: leaked a future value"


def test_stale_values_expire():
    src = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))
    assert align(IDX, s=src, ffill_limit=3)["s"].notna().sum() == 4  # pub bar + 3

    # the limit counts TARGET bars, not source rows: a calendar-daily source
    # must not have its budget eaten by weekends that are not trading bars.
    daily = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=1))
    assert align(IDX, s=daily, ffill_limit=10)["s"].notna().sum() == 11


def test_future_covariates_are_horizon_longer():
    fi = future_index(IDX, 5)
    assert len(fi) == len(IDX) + 5 and fi[: len(IDX)].equals(IDX) and (fi[len(IDX):] > IDX[-1]).all()
    assert calendar(fi).notna().all().all()


def test_covariate_windows_align_to_the_decision_bar():
    """The window handed to the model must end at bar t -- same as the price."""
    px = pd.Series(100 + np.arange(300.0), index=IDX)
    past = pd.DataFrame({"a": np.arange(300.0) * 10}, index=IDX)
    fut = pd.DataFrame({"c": np.arange(305.0) * 100}, index=future_index(IDX, 5))

    seen = []
    stub(lambda c: c[-1])
    real = bt._model()

    class Spy:
        def predict_batch(self, contexts, horizon, **kw):
            seen.extend(zip(contexts, kw.get("past_only_covariates"), kw.get("past_future_covariates")))
            return real.predict_batch(contexts, horizon, **kw)

    bt._model = lambda: Spy()
    df = bt.walk_forward(px, horizon=5, context=100, past=past, future=fut)

    assert len(seen) == len(df)
    for (ctx, p, f), t in zip(seen, df.index):
        j = IDX.get_loc(t)
        assert p.shape == (1, 100) and f.shape == (1, 105)
        assert ctx[-1] == np.float32(px.iloc[j])
        assert p[0, -1] == np.float32(past["a"].iloc[j]), "past covariate must end at t"
        assert f[0, 99] == np.float32(fut["c"].iloc[j]), "future cov must be aligned at t"
        assert f[0, -1] == np.float32(fut["c"].iloc[j + 5]), "and run exactly horizon past t"


def test_bad_covariates_are_rejected_not_guessed():
    px = pd.Series(100 + np.arange(300.0), index=IDX)
    stub(lambda c: c[-1])
    for bad, why in [
        (pd.DataFrame({"a": np.arange(299.0)}, index=IDX[:299]), "wrong length"),
        (pd.DataFrame({"a": [np.nan] * 300}, index=IDX), "NaN"),
        (pd.DataFrame({"a": np.arange(300.0)}, index=pd.bdate_range("2020-01-01", periods=300)), "wrong dates"),
    ]:
        try:
            bt.walk_forward(px, horizon=5, context=100, past=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted covariates with {why}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
