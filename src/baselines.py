"""Non-ML baselines: each turns a restaurant's inspection history into a score; higher scores are visited first.

Every rule is a total order: its score, then the high-risk count at the last routine inspection, then how
overdue the restaurant is, then facility ID. A missing value sorts last on its key, except for the score of the
two schedule rules: a restaurant with no usable inspection interval is treated as due and sorts first.
"""
import numpy as np

RULES = {  # name: (feature used as the score, description)
    'history_rate': ('history_rate', 'share of past routine inspections with a high-risk violation'),
    'recency_weighted_rate': ('recent_weighted_rate', 'same, each inspection weighted by 0.5^(years ago)'),
    'mean_high_count': ('mean_high_count', 'mean number of high-risk violations per past routine inspection'),
    'history_rate_3y': ('history_rate_3y', 'share of routine inspections in the past three years with a high-risk violation'),
    'last_result': ('last_result', 'number of high-risk violations at the last routine inspection'),
    'rotation': ('days_since_last_routine', 'days since the last routine inspection'),
    'most_overdue': ('overdue', 'days since the last routine inspection / mean of its last four gaps (current practice)'),
}
TIE_BREAK = ('last_result', 'overdue')
MISSING_FIRST = {'rotation', 'most_overdue'}  # schedule rules: no known interval = due now


def order(f, score_col, missing_first=False):
    """Row positions of f, best first."""
    keys = [f[c].fillna(np.inf if (missing_first and c == score_col) else -np.inf).to_numpy()
            for c in (score_col,) + TIE_BREAK]
    ids = f.index.astype(str).to_numpy()
    # np.lexsort sorts by the last key first; negate numeric keys for descending order, ids ascending
    return np.lexsort([ids] + [-k for k in keys[::-1]])


def order_random_ties(score, rng, missing_first=False):
    """Order by the score alone, ties broken at random."""
    s = np.nan_to_num(np.asarray(score, float), nan=np.inf if missing_first else -np.inf)
    return np.lexsort((rng.random(len(s)), -s))
