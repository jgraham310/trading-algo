"""Synthetic fixtures only: no claim these were observed AFPR signals."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest

from trading_algo.afpr import Observation, capture, event_study
from trading_algo.engine import run, stats
from trading_algo.backtest import _max_dd
from trading_algo.intake import weekly_levels


def observation():
    return Observation('fixture', 'XYZ', '2026-11-13T21:01:00Z',
        '2026-11-13T21:00:00Z', 'https://example.com/synthetic', 'reconstruction',
        'daily', 'fixture', 100, 90, 120, 'close confirmed by source')


def bars():
    days = ['2026-11-13', '2026-11-16', '2026-11-17', '2026-11-18']
    return pd.DataFrame({'open_at': [x+'T14:30:00Z' for x in days],
        'close_at': [x+'T21:00:00Z' for x in days],
        'open': [99., 101., 80., 110.], 'close': [101., 89., 110., 112.]})


def test_next_open_gap_exit_and_two_sided_cost():
    r = event_study(observation(), bars())
    assert r['entry_at'] == '2026-11-16T14:30:00+00:00'
    assert r['exit'] == 80  # not the invalidation level 90
    assert r['exit_reason'] == 'invalidation'
    assert r['net_return'] == pytest.approx((80*.999)/(101*1.001)-1)


def test_pending_missing_and_benchmark_alignment():
    r = event_study(observation(), bars().iloc[:2])
    assert r['status'] == 'pending_exit' and r['net_return'] is None
    r = event_study(observation(), bars(), benchmark=bars().iloc[:1])
    assert r['benchmark_return'] is None
    b = bars(); b.loc[2, 'open'] = np.nan
    with pytest.raises(ValueError): event_study(observation(), b)
    r = event_study(observation(), bars(), benchmark=bars())
    assert r['excess_return'] == pytest.approx(0)


def test_aware_and_strict_entry():
    with pytest.raises(ValueError):
        event_study(replace(observation(), observed_at='2026-11-13'), bars())
    r = event_study(replace(observation(), observed_at='2026-11-16T14:30:00Z'), bars())
    assert r['status'] == 'skipped_gap' and r['entry'] == 80


def test_capture_revisions(tmp_path):
    first = capture(tmp_path, observation())
    contents = first.read_text()
    assert json.loads(contents)['recorded_at']
    with pytest.raises(ValueError): capture(tmp_path, observation())
    capture(tmp_path, replace(observation(), revision=2, observed_at='2026-11-16T21:01:00Z', target=130))
    assert first.read_text() == contents
    with pytest.raises(ValueError): capture(tmp_path, replace(observation(), signal_id='../bad'))


def test_first_loss_drawdown_and_initial_entry_cost():
    assert stats(pd.Series([-.1, .01]))['max_dd'] == pytest.approx(-.1)
    assert _max_dd(np.array([-.1, .01])) == pytest.approx(-.1)
    x = pd.Series([0., 0., 0.]); w = pd.Series([1., 1., 0.])
    assert run(w, x, 10).tolist() == pytest.approx([-.001, 0.])


def test_weekly_decision_not_delayed_extra_week():
    idx = pd.bdate_range('2024-03-01', '2024-03-18')
    p = pd.DataFrame({'A': np.arange(len(idx), dtype=float)}, index=idx)
    s = weekly_levels(p, 1)
    assert s.loc['2024-03-08','A'] == 1
    assert s.shift(1).loc['2024-03-11','A'] == 1


def test_holiday_prefix_invariance():
    idx = pd.to_datetime(['2024-03-15','2024-03-18','2024-03-19',
                         '2024-03-20','2024-03-22','2024-03-28','2024-04-01'])
    p = pd.DataFrame({'A':[1.,2.,3.,4.,5.,.5,11.]}, index=idx)
    pd.testing.assert_frame_equal(weekly_levels(p,1).iloc[:-1], weekly_levels(p.iloc[:-1],1))


def test_capture_rejected_bad_vendor_levels_and_cutoff(tmp_path):
    capture(tmp_path, replace(observation(), disposition='rejected',
                             invalidation=-1, reason='invalid vendor geometry'))
    with pytest.raises(ValueError):
        replace(observation(), data_as_of='2026-11-14T21:00:00Z').validate()


def test_pending_and_excursions():
    assert event_study(observation(), bars().iloc[:1])['status'] == 'pending_entry'
    b = bars(); b['close'] = [101., 105., 110., 115.]; b['open'] = [99., 101., 106., 111.]
    assert event_study(observation(), b)['status'] == 'pending_horizon'
    assert event_study(observation(), b, horizon=1)['close_mae'] == 0
    b['close'] = [101., 99., 98., 97.]; b['open'] = [99., 101., 98., 97.]
    assert event_study(observation(), b, horizon=1)['close_mfe'] == 0
    before = replace(observation(), observed_at='2026-11-16T14:29:00Z')
    assert event_study(before, b)['entry'] == 101
