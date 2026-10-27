"""Cutoff and label logic of src/panel.py on a small synthetic inspection table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import panel  # noqa: E402

T = pd.Timestamp('2020-01-01')


def make_ins() -> pd.DataFrame:
    rows = [
        # id, date, purpose type, n_high
        ('A', '2019-06-01', 'routine', 0),      # known, routine history
        ('A', '2020-03-01', 'complaint', 3),    # after t but not routine
        ('A', '2020-05-01', 'routine', 2),      # first routine after t, serious
        ('B', '2019-12-02', 'routine', 1),      # exactly 30 days before t: not known
        ('B', '2021-01-01', 'routine', 0),      # day 366 after t: outside 12 months
        ('C', '2016-06-01', 'routine', 0),      # older than the 3-year look-back
        ('D', '2018-02-01', 'routine', 0),      # inside the look-back, no inspection after t
    ]
    ins = pd.DataFrame(rows, columns=['id', 'date', 'type', 'n_high'])
    ins['date'] = pd.to_datetime(ins.date)
    ins['routine'] = ins.type.eq('routine')
    ins['encounter'] = np.arange(len(ins)).astype(str)
    ins['description'] = 'Restaurant without Liquor'
    ins['status'] = '1'
    ins['municipal'] = 'X'
    ins['zip'] = '15213'
    ins['bus_st_date'] = '2010-03-15'
    ins['purpose'] = ins.type.map({'routine': 'Initial', 'complaint': 'Complaint'})
    ins['placard_st'] = '1'
    ins['reispt_cd'] = '2'
    ins['n_critical'] = 0
    return ins


def test_known_excludes_the_lag_boundary():
    ins = make_ins()
    k = panel.known(ins, T)
    assert 'B' not in set(k.id)
    assert 'A' in set(k.id)


def test_candidates_apply_the_lookback():
    ins = make_ins()
    c = panel.candidates(ins, T)
    assert set(c.index) == {'A', 'D'}


def test_first_routine_skips_complaints():
    ins = make_ins()
    lab = panel.labels(ins, pd.Index(['A', 'D']), T, pd.Timestamp('2022-01-01'))
    assert lab.loc['A', 'first_routine_date'] == pd.Timestamp('2020-05-01')
    assert lab.loc['A', 'n_high_first'] == 2
    assert np.isnan(lab.loc['D', 'gap_days'])


def test_horizon_and_coverage():
    ins = make_ins()
    lab = panel.labels(ins, pd.Index(['A', 'B']), T, pd.Timestamp('2020-07-01'))
    ins12 = panel.inspected(lab, 12)
    assert bool(pd.isna(ins12['A']))                  # only 6 months covered: unknown
    lab = panel.labels(ins, pd.Index(['A', 'B']), T, pd.Timestamp('2022-01-01'))
    ins12 = panel.inspected(lab, 12)
    assert bool(ins12['A']) and not bool(ins12['B'])  # B's next routine is on day 366
    assert bool(panel.serious(lab, 12)['A'])


def test_selection_h_tier_and_calendar_year():
    import selection_features
    ins = make_ins()
    ins['n_critical'] = [3, 0, 0, 0, 0, 0, 0]
    ins['reispt_cd'] = ['1', '2', '2', '2', '2', '2', '2']
    roster = pd.DataFrame({'seat_count': [50.0], 'sq_feet': [np.nan], 'fdo': ['2010-03-15']}, index=pd.Index(['A'], name='id'))
    cand = panel.candidates(ins, T)
    f = selection_features.build(ins, roster, cand, T)
    assert f.loc['A', 'h_tier'] == 1 and f.loc['D', 'h_tier'] == 0
    assert f.loc['A', 'days_to_due'] == (pd.Timestamp('2019-06-01') + pd.Timedelta(days=180) - T).days
    assert f.loc['A', 'routine_this_year'] == 0 and f.loc['A', 'cal_years_since_routine'] == 1
    assert f.loc['A', 'months_to_anniversary'] == 2
    assert f.loc['A', 'seat_count'] == 50
