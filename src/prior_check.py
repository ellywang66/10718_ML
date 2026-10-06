"""Where the smoothing constants of the history rate come from, and how much they matter.

The history rate is (s + m*w) / (n + w): a Beta prior with mean m and weight w (in pseudo-inspections).
The baselines use m = 0.3 and w = 2, i.e. (s + 0.6) / (n + 2). This script uses only data known before the
first scoring date (Jan 1, 2023) for the estimates, and only the development years for the sensitivity
check. The test set is not read. Writes results/prior_check.json.
"""
import json
import numpy as np
import pandas as pd

import features
import history
import splits
from baselines import order
from config import DEV_YEARS, REPORT_LAG_DAYS, RESULTS, SERIOUS

MEANS = (0.10, 0.20, 0.26, 0.28, 0.30, 0.34, 0.40, 0.50)
WEIGHTS = (0.5, 1, 2, 3, 5, 10)


def prior_from_history(ins, cutoff):
    """Base rate and method-of-moments Beta fit from routine inspections known before the cutoff."""
    r = ins[ins.routine & (ins.date + pd.Timedelta(days=REPORT_LAG_DAYS) < pd.Timestamp(cutoff))]
    g = (r.n_high > 0).astype(float).groupby(r.id).agg(['sum', 'count'])
    mean = g['sum'].sum() / g['count'].sum()
    both = (g['sum'] * (g['sum'] - 1)).sum() / (g['count'] * (g['count'] - 1)).sum()  # P(two inspections both high)
    weight = mean * (1 - mean) / (both - mean ** 2) - 1
    return dict(routine_inspections=int(len(r)), base_rate=round(mean, 3), fitted_mean=round(mean, 3),
                fitted_weight=round(weight, 2))


def dev_recalls(ins):
    """Recall in the top half on each development year for every (mean, weight)."""
    out = {}
    for year in DEV_YEARS:
        lab = splits.dev_set(ins, year)
        f = features.build(ins, lab.id.tolist(), f'{year}-01-01')
        y = (lab.set_index('id').n_high.reindex(f.index) >= SERIOUS).astype(int).to_numpy()
        n = f.n_routine.to_numpy()
        s = np.where(n > 0, np.round(f.history_rate.to_numpy() * (n + 2) - 0.6), 0.0)
        k = int(np.ceil(0.5 * len(y)))
        out[year] = dict(n=int(len(y)), no_routine_history=int((n == 0).sum()), recall={})
        for m in MEANS:
            for w in WEIGHTS:
                g = f.copy()
                g['history_rate'] = (s + m * w) / (n + w)
                out[year]['recall'][f'{m},{w}'] = round(100 * y[order(g, 'history_rate')[:k]].sum() / y.sum(), 1)
    return out


def main():
    ins = history.load_inspections()
    res = dict(prior_from_pre_2023_history=prior_from_history(ins, f'{DEV_YEARS[0]}-01-01'), development=dev_recalls(ins))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'prior_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['prior_from_pre_2023_history'], indent=1))
    for year, d in res['development'].items():
        v = d['recall']
        print(year, 'used (0.3, 2):', v['0.3,2'], '| range over all means and weights:', min(v.values()), 'to', max(v.values()),
              '| weight 0.5..10 at mean 0.3:', sorted({v[f'0.3,{w}'] for w in WEIGHTS}))


if __name__ == '__main__':
    main()
