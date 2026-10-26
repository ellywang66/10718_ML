"""Simple ML baseline promised in the baselines assignment: logistic regression on five history features.

Features: historical high-risk rate, high-risk count at the last routine inspection, days since the last routine
inspection, number of past routine inspections, and complaints in the past year. Same rows, split, grid, selection
rule and test evaluation as train_models.py (logistic regression family only). Writes results/ml_lr5_results.json.
"""
import json

import numpy as np
import pandas as pd

import history
import splits
import train_models as tm
from config import DEV_YEARS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE
import ml_features

X5 = ['history_rate', 'last_result', 'days_since_last_routine', 'n_routine', 'n_complaint_1y']


def main():
    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    grid, make = tm.GRID['logistic_regression']
    runs = []
    for family, params, mk in tm.configs():
        if family != 'logistic_regression':
            continue
        r = dict(params=params)
        for v in DEV_YEARS:
            tr, dv = tm.train_rows(rows, f'{v}-01-01'), rows[rows.year == v]
            m = make(params).fit(tr[X5], tr.y)
            r[f'dev{v}'] = tm.rank_eval(m.predict_proba(dv[X5])[:, 1], dv, dv.y.to_numpy())[1]
        r['dev_mean_recall_at_50'] = float(np.mean([r[f'dev{v}']['recall_at_50'] for v in DEV_YEARS]))
        r['dev_mean_auc'] = float(np.mean([r[f'dev{v}']['auc'] for v in DEV_YEARS]))
        runs.append(r)
        print(f"{params} dev recall@50 {r['dev_mean_recall_at_50']:.3f}")
    chosen = max(runs, key=lambda r: (r['dev_mean_recall_at_50'], r['dev_mean_auc']))

    test_lab = splits.test_set(ins)
    ft = ml_features.build(ins, test_lab.id.tolist(), TEST_SCORING_DATE)
    yt = (test_lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    tr = tm.train_rows(rows, TEST_SCORING_DATE)
    m = make(chosen['params']).fit(tr[X5], tr.y)
    o_m, test_m = tm.rank_eval(m.predict_proba(ft[X5])[:, 1], ft, yt)
    o_b, test_b = tm.rank_eval(ft.history_rate.to_numpy(), ft, yt)

    rng = np.random.default_rng(SEED)
    k = int(np.ceil(len(yt) / 2))
    top_m, top_b = (np.argsort(o_m) + 1) <= k, (np.argsort(o_b) + 1) <= k
    diffs = []
    for _ in range(4000):
        i = rng.integers(0, len(yt), len(yt))
        diffs.append(100 * ((yt[i] * top_m[i]).sum() - (yt[i] * top_b[i]).sum()) / yt[i].sum())
    diffs = np.array(diffs)

    coef = m[-1].coef_[0]
    names = list(m[0].get_feature_names_out(X5))
    out = dict(features=X5, chosen=chosen['params'],
               dev={str(v): chosen[f'dev{v}'] for v in DEV_YEARS},
               test=dict(lr5=test_m, history_rate_rule=test_b),
               difference_vs_baseline=dict(points=round(100 * (test_m['recall_at_50'] - test_b['recall_at_50']), 1),
                                           bootstrap_se=round(float(diffs.std()), 2),
                                           ci95=[round(float(x), 1) for x in np.percentile(diffs, [2.5, 97.5])]),
               standardized_coef=dict(zip(names, [round(float(c), 3) for c in coef])),
               runs=runs)
    (RESULTS / 'ml_lr5_results.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ('chosen', 'test', 'difference_vs_baseline', 'standardized_coef')},
                     indent=1, default=str))


if __name__ == '__main__':
    main()
