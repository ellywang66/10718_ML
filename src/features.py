"""Point-in-time history features for (restaurant, scoring date) pairs.

Only inspections whose report is known before the scoring date (inspection date + 30 days < scoring date)
are visible. All rates use a Beta(0.6, 1.4) prior (mean 0.3), so a restaurant with s high-risk routine
inspections out of n has rate (s + 0.6) / (n + 2); a restaurant with no routine inspection yet gets 0.3.
"""
import numpy as np
import pandas as pd
from config import REPORT_LAG_DAYS

PRIOR_A, PRIOR_B = 0.6, 1.4


def _rate(s, n):
    return (s + PRIOR_A) / (n + PRIOR_A + PRIOR_B)


def restaurant_features(g, cutoff):
    """g: one restaurant's inspections sorted by date; cutoff: pd.Timestamp."""
    g = g[g.date + pd.Timedelta(days=REPORT_LAG_DAYS) < cutoff]
    r = g[g.routine]
    f = {}
    if len(r) == 0:
        # no routine history: the rates fall back to the prior mean (0.3); count-based keys are missing
        prior = _rate(0, 0)
        return dict(history_rate=prior, history_rate_3y=prior, recent_weighted_rate=prior, mean_high_count=np.nan,
                    last_result=np.nan, days_since_last_routine=np.nan, overdue=np.nan, n_routine=0)
    age = (cutoff - r.date).dt.days.to_numpy()
    hi = (r.n_high.to_numpy() > 0).astype(float)
    recent = age <= 1095
    w = 0.5 ** (age / 365.0)
    gaps = np.diff((r.date - pd.Timestamp('2000-01-01')).dt.days.to_numpy())
    f['history_rate'] = _rate(hi.sum(), len(hi))
    f['history_rate_3y'] = _rate(hi[recent].sum(), int(recent.sum()))
    f['recent_weighted_rate'] = (np.dot(w, hi) + PRIOR_A) / (w.sum() + PRIOR_A + PRIOR_B)
    f['mean_high_count'] = float(r.n_high.mean())
    f['last_result'] = float(r.n_high.iloc[-1])
    f['days_since_last_routine'] = float(age[-1])
    mean_gap = gaps[-4:].mean() if len(gaps) else np.nan
    f['overdue'] = f['days_since_last_routine'] / mean_gap if len(gaps) and mean_gap > 0 else np.nan
    f['n_routine'] = len(r)
    return f


def build(ins, ids, cutoff):
    """Features for every id at one scoring date. ins: output of history.load_inspections()."""
    cutoff = pd.Timestamp(cutoff)
    groups = dict(tuple(ins[ins.id.isin(set(ids))].groupby('id', sort=False)))
    empty = ins.iloc[:0]
    rows = [restaurant_features(groups.get(i, empty), cutoff) for i in ids]
    return pd.DataFrame(rows, index=pd.Index(ids, name='id'))
