"""Sampling noise of recall at K on the test set.

Bootstraps the test restaurants (with replacement) and recomputes recall at K = top half for every
baseline ranking in results/test_rankings.csv. Reports the standard error of each rule's recall and of
the recall difference between the chosen rule and every other rule. Writes results/recall_noise.json.
"""
import json
import numpy as np
import pandas as pd

from config import RESULTS, SEED

CHOSEN = 'history_rate'
DRAWS = 4000


def recall_top_half(rank, y):
    k = int(np.ceil(len(y) / 2))
    top = np.argsort(rank, kind='stable')[:k]
    return y[top].sum() / y.sum()


def main():
    d = pd.read_csv(RESULTS / 'test_rankings.csv')
    y = d.serious_violator.values
    ranks = {c[len('rank_'):]: d[c].values for c in d.columns if c.startswith('rank_')}
    rng = np.random.default_rng(SEED)
    draws = {r: [] for r in ranks}
    for _ in range(DRAWS):
        idx = rng.integers(0, len(d), len(d))
        for r, rank in ranks.items():
            draws[r].append(100 * recall_top_half(rank[idx], y[idx]))
    draws = {r: np.array(v) for r, v in draws.items()}
    out = {'n': len(d), 'serious_violators': int(y.sum()), 'draws': DRAWS, 'recall': {}, 'difference_from_chosen': {}}
    for r, v in draws.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out['recall'][r] = dict(recall=round(100 * recall_top_half(ranks[r], y), 1), se=round(v.std(), 2),
                                ci95=[round(lo, 1), round(hi, 1)])
        if r != CHOSEN:
            diff = draws[CHOSEN] - v
            lo, hi = np.percentile(diff, [2.5, 97.5])
            out['difference_from_chosen'][r] = dict(mean=round(diff.mean(), 1), se=round(diff.std(), 2),
                                                    ci95=[round(lo, 1), round(hi, 1)])
    (RESULTS / 'recall_noise.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
