"""Simple ML baseline promised in the baselines assignment: logistic regression on five history features.

Same training rows, development dates, grid, selection criterion and test evaluation as train_outcome.py,
restricted to the logistic-regression family and five features. Writes results/ml_lr5_results.json.
"""
import json

import numpy as np

from src import panel
from src.models import train_outcome as tm
from src.config import DEV_YEARS, RESULTS, TEST_SCORING_DATE

X5 = ['history_rate', 'last_result', 'days_since_last_routine', 'n_routine', 'n_complaint_1y']


def main() -> None:
    p = panel.load()
    rows = tm.training_rows(p)
    dev_dates = [f'{y}-01-01' for y in DEV_YEARS]
    dev_cand = {d: panel.at(p, d) for d in dev_dates}
    _, make = tm.GRID['logistic_regression']
    runs = []
    for _, params, _ in tm.configs({'logistic_regression': tm.GRID['logistic_regression']}):
        r = dict(params=params)
        for d in dev_dates:
            tr = tm.usable(rows, d)
            m = make(params).fit(tr[X5], tr.y)
            r[f'dev{d[:4]}'] = tm.score_candidates(m.predict_proba(dev_cand[d][X5])[:, 1], dev_cand[d])[1]
        r['dev_mean_selection'] = float(np.mean([r[f'dev{d[:4]}'][tm.SELECTION_METRIC] for d in dev_dates]))
        runs.append(r)
    chosen = max(runs, key=lambda r: r['dev_mean_selection'])

    test_cand = panel.at(p, TEST_SCORING_DATE)
    tr = tm.usable(rows, TEST_SCORING_DATE)
    m = make(chosen['params']).fit(tr[X5], tr.y)
    _, test_m = tm.score_candidates(m.predict_proba(test_cand[X5])[:, 1], test_cand, bootstrap=True)
    _, test_b = tm.score_candidates(test_cand.history_rate.to_numpy(), test_cand, bootstrap=True)
    names = list(m[0].get_feature_names_out(X5))
    out = dict(features=X5, chosen=chosen['params'], q_selection_metric=tm.SELECTION_METRIC,
               dev={d: chosen[f'dev{d[:4]}'] for d in dev_dates},
               test=dict(lr5=test_m, history_rate_rule=test_b),
               standardized_coef=dict(zip(names, [round(float(c), 3) for c in m[-1].coef_[0]])), runs=runs)
    (RESULTS / 'ml_lr5_results.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ('chosen', 'test', 'standardized_coef')}, indent=1, default=str))


if __name__ == '__main__':
    main()
