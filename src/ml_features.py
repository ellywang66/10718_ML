"""Point-in-time features for the ML models: the baseline history features plus a few more.

Every feature for a (restaurant, scoring date) pair uses only inspections whose report is known before the
scoring date (inspection date + 30 days < scoring date), the same rule as the baselines.
"""
import numpy as np
import pandas as pd

import features
from config import REPORT_LAG_DAYS

BASE = ['history_rate', 'history_rate_3y', 'recent_weighted_rate', 'mean_high_count', 'last_result',
        'days_since_last_routine', 'overdue', 'n_routine']
EXTRA = ['n_routine_high', 'rate_last3', 'max_high', 'last_med', 'last_low', 'mean_med', 'mean_low',
         'n_complaint_1y', 'n_complaint_all', 'n_reinspection_1y', 'high_nonroutine_1y',
         'days_since_any', 'years_on_record', 'has_liquor', 'is_chain']
FEATURES = BASE + EXTRA


def extra_features(g, cutoff):
    """g: one restaurant's inspections (all purposes) sorted by date; cutoff: pd.Timestamp."""
    g = g[g.date + pd.Timedelta(days=REPORT_LAG_DAYS) < cutoff]
    nan = np.nan
    f = dict.fromkeys(EXTRA, nan)
    if len(g) == 0:
        return f
    age = (cutoff - g.date).dt.days.to_numpy()
    purpose = g.purpose.fillna('')
    last_year = age <= 365
    complaint = purpose.str.contains('Complaint').to_numpy()
    reinsp = purpose.str.contains('Reinspection').to_numpy()
    routine = g.routine.to_numpy()
    f['n_complaint_1y'] = int((complaint & last_year).sum())
    f['n_complaint_all'] = int(complaint.sum())
    f['n_reinspection_1y'] = int((reinsp & last_year).sum())
    f['high_nonroutine_1y'] = int(g.n_high.to_numpy()[~routine & last_year].sum())
    f['days_since_any'] = float(age.min())
    f['years_on_record'] = float(age.max()) / 365.0
    desc = str(g.description.iloc[-1])  # category on the latest known record
    f['has_liquor'] = int('with Liquor' in desc)
    f['is_chain'] = int(desc.startswith('Chain'))
    r = g[g.routine]
    if len(r):
        hi = r.n_high.to_numpy()
        f['n_routine_high'] = int((hi > 0).sum())
        f['rate_last3'] = float((hi[-3:] > 0).mean())
        f['max_high'] = int(hi.max())
        f['last_med'] = int(r.n_med.iloc[-1])
        f['last_low'] = int(r.n_low.iloc[-1])
        f['mean_med'] = float(r.n_med.mean())
        f['mean_low'] = float(r.n_low.mean())
    return f


def build(ins, ids, cutoff):
    """Feature table (one row per id) at one scoring date, plus the municipality used only for subgroup analysis."""
    cutoff = pd.Timestamp(cutoff)
    base = features.build(ins, ids, cutoff)
    groups = dict(tuple(ins[ins.id.isin(set(ids))].groupby('id', sort=False)))
    empty = ins.iloc[:0]
    extra = pd.DataFrame([extra_features(groups.get(i, empty), cutoff) for i in ids], index=base.index)
    known = ins[ins.date + pd.Timedelta(days=REPORT_LAG_DAYS) < cutoff]
    muni = known.drop_duplicates('id', keep='last').set_index('id').municipal.reindex(base.index)
    out = pd.concat([base, extra], axis=1)[FEATURES].astype(float)
    out['municipal'] = muni.fillna('unknown').to_numpy()
    return out
