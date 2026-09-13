"""Backtest checks. Model is stubbed -- these test alignment and metrics."""

import numpy as np
import pandas as pd

from trading_algo import backtest as bt


class Out:
    def __init__(self, f, q):
        self.forecast, self.quantiles = f, q


def stub(fn):
    """Patch the model with `fn(context) -> horizon-end price`."""
    class M:
        def predict_batch(self, contexts, horizon, **kw):
            for c in contexts:
                p = fn(np.asarray(c))
                yield Out(np.full(horizon, p), np.tile(np.linspace(p * 0.97, p * 1.03, 9), (horizon, 1)))
    bt._model = lambda: M()


PX = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300))),
               index=pd.bdate_range("2024-01-01", periods=300))
H, C = 5, 100


def test_alignment():
    """The single most important property: nothing after bar t reaches bar t."""
    seen = []
    stub(lambda c: (seen.append(c.copy()), c[-1])[1])
    df = bt.walk_forward(PX, horizon=H, context=C)

    assert len(df) == len(PX) - C - H + 1 == len(seen)
    for i, (t, row) in enumerate(df.iterrows()):
        j = PX.index.get_loc(t)
        assert len(seen[i]) == C
        assert seen[i][-1] == np.float32(PX.iloc[j]), "context must END at the decision bar"
        assert np.allclose(seen[i], PX.iloc[j - C + 1 : j + 1].to_numpy(np.float32))
        assert row["last"] == np.float32(PX.iloc[j])
        assert row["actual"] == np.float32(PX.iloc[j + H]), "target must be t+horizon"
        assert abs(row["next_ret"] - (np.float32(PX.iloc[j + 1]) / np.float32(PX.iloc[j]) - 1)) < 1e-6
    assert df.index[-1] <= PX.index[-1 - H], "cannot decide on a bar with no future to score"


def test_naive_forecast_scores_as_naive():
    stub(lambda c: c[-1])  # "price stays put"
    m = bt.evaluate(bt.walk_forward(PX, horizon=H, context=C))
    assert abs(m["mae_ratio"] - 1.0) < 1e-6, m["mae_ratio"]
    assert m["trades"] == 0 and m["time_in_mkt"] == 0.0  # flat forecast -> no position
    assert m["ann_return"] == 0.0


def test_perfect_foresight_makes_money():
    """Sanity: if the forecast were perfect, the harness must show a big edge."""
    px = PX.to_numpy(np.float32)
    fut = {float(px[i]): float(px[i + H]) for i in range(len(px) - H)}
    stub(lambda c: fut[float(c[-1])])
    m = bt.evaluate(bt.walk_forward(PX, horizon=H, context=C))
    assert m["mae_ratio"] < 1e-6
    assert m["dir_acc"] > 0.99
    assert m["sharpe"] > 3, m["sharpe"]
    assert m["hit_rate"] > 0.6


def test_costs_bite():
    px = PX.to_numpy(np.float32)
    fut = {float(px[i]): float(px[i + H]) for i in range(len(px) - H)}
    stub(lambda c: fut[float(c[-1])])
    df = bt.walk_forward(PX, horizon=H, context=C)
    free, dear = bt.evaluate(df, cost_bps=0), bt.evaluate(df, cost_bps=50)
    assert free["cost_drag"] == 0.0
    assert dear["cost_drag"] > 0.01
    assert dear["ann_return"] < free["ann_return"]
    assert free["ann_turnover"] == dear["ann_turnover"]  # costs must not change trading


def test_metric_math():
    r = np.array([0.10, -0.05, 0.10, -0.05] * 63)  # 252 bars
    assert abs(bt._ann(r, 252) - (np.prod(1 + r) - 1)) < 1e-9
    assert bt._max_dd(np.array([0.1, -0.5, 0.1])) < -0.49
    assert bt._max_dd(np.array([0.01, 0.01])) == 0.0
    assert np.isnan(bt._sharpe(np.array([0.01] * 10)))  # zero vol -> undefined, not 1e15
    assert abs(bt._sharpe(np.array([0.01, -0.01] * 50))) < 1e-9  # zero mean -> zero




def test_ma_positions_have_no_lookahead():
    """MA at bar t must use only closes <= t, and imply a position for t+1."""
    px = pd.Series([1.0] * 10 + [100.0] * 10)
    pos = bt.ma_positions(px, window=5)
    assert pos.iloc[:4].isna().all(), "warmup must be NaN, not a silent flat"
    assert pos.iloc[9] == 0.0, "still at/below MA on the bar before the jump"
    assert pos.iloc[10] == 1.0, "goes long on the jump bar, earning it from t+1"
    # recomputing on a truncated series must not change any earlier value
    for cut in (12, 15, 18):
        assert bt.ma_positions(px[:cut], 5).equals(pos.iloc[:cut])


def test_ma_benchmark_is_scored_by_the_same_code():
    stub(lambda c: c[-1])
    df = bt.walk_forward(PX, horizon=H, context=C)
    m = bt.evaluate(df, prices=PX, ma_window=50)  # 300 bars, decisions start at 99
    assert 0.0 < m["ma_time_in_mkt"] < 1.0, "an MA that never trades is not a benchmark"

    # window=1 -> close is never strictly above its own 1-bar mean -> always flat
    flat = bt.evaluate(df, prices=PX, ma_window=1)
    assert flat["ma_time_in_mkt"] == 0.0 and flat["ma_ann_return"] == 0.0

    # buy&hold column must be exactly what the shared scorer gives always-long
    hand = bt._stats(np.ones(len(df)), df["next_ret"].to_numpy(), 1.0)
    for k in ("ann_return", "sharpe", "max_dd"):
        assert abs(hand[k] - m["bh_" + k]) < 1e-12, k


def test_ma_warmup_is_enforced_not_faked():
    stub(lambda c: c[-1])
    df = bt.walk_forward(PX, horizon=H, context=C)
    try:
        bt.evaluate(df, prices=PX, ma_window=290)  # not enough history before bar 0
    except ValueError:
        pass
    else:
        raise AssertionError("silently scored an MA that was never warmed up")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
