"""Line 2: who is likely not to be inspected, and does the violation model flag any of them?

Propensity e(x) = P(routine inspection in calendar year y | features at Jan 1 of y), over every candidate
(splits.candidates) on Jan 1 of y, with the ML features (ml_features.py). Fit on y = 2016..2022; logistic regression
(L2, C in {0.01, 0.1, 1}) and histogram gradient boosting; selected on 2023 AUC; reported on 2024 and at the test
scoring date (Jan 1, 2025), where inspected = in the linked test set. The test set covers only Comprehensive
inspections Aug 11, 2025 - Jul 31, 2026 that could be linked to an old-system ID, so it undercounts inspections.

At Jan 1, 2025, predicted-not-inspected = candidates whose e is below the threshold that makes the predicted-inspected
count equal the test inspected count. They are scored with the chosen violation model q (results/ml_results.json,
fit as in train_models.py); flagged = q at or above the score of the K-th restaurant (K = 50%) in the test set.
q is trained only on inspected restaurants. Writes results/line2_propensity.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import history
import ml_features
import splits
import train_models as tm
from config import RESULTS, SEED, SERIOUS, TEST_SCORING_DATE

FIT_YEARS = range(2016, 2023)
SELECT_YEAR, REPORT_YEAR = 2023, 2024
CANDIDATE_MODELS = {f'logistic_regression_C{c}': tm.GRID['logistic_regression'][1](dict(penalty='l2', C=c))
                    for c in (0.01, 0.1, 1)}
CANDIDATE_MODELS['gradient_boosting'] = HistGradientBoostingClassifier(random_state=SEED)


def year_rows(ins, year):
    cand = splits.candidates(ins, f'{year}-01-01')
    f = ml_features.build(ins, cand, f'{year}-01-01')
    done = set(ins[ins.routine & (ins.date.dt.year == year)].id)
    f['inspected'] = f.index.isin(done).astype(int)
    f['year'] = year
    return f


def q_stats(g, cutoff):
    """Recall and precision of q at the cutoff within labeled restaurants g (columns q, y)."""
    flag = g.q >= cutoff; pos = int(g.y.sum())
    return dict(n=int(len(g)), serious=pos, flagged=int(flag.sum()),
                recall=round(float((flag & g.y.eq(1)).sum() / pos), 4) if pos else None,
                precision=round(float(g.y[flag].mean()), 4) if flag.any() else None)


def main():
    ins = history.load_inspections()
    X = tm.X
    rows = pd.concat([year_rows(ins, y) for y in [*FIT_YEARS, SELECT_YEAR, REPORT_YEAR]])
    fit = rows[rows.year.isin(FIT_YEARS)]
    sel, rep = rows[rows.year == SELECT_YEAR], rows[rows.year == REPORT_YEAR]

    lab = splits.test_set(ins).set_index('id')
    cand = splits.candidates(ins, TEST_SCORING_DATE)
    fc = ml_features.build(ins, cand, TEST_SCORING_DATE)
    fc['inspected'] = fc.index.isin(lab.index).astype(int)

    runs = {}
    for name, m in CANDIDATE_MODELS.items():
        m.fit(fit[X], fit.inspected)
        runs[name] = dict(auc_2023=round(roc_auc_score(sel.inspected, m.predict_proba(sel[X])[:, 1]), 4),
                          auc_2024=round(roc_auc_score(rep.inspected, m.predict_proba(rep[X])[:, 1]), 4),
                          auc_test=round(roc_auc_score(fc.inspected, m.predict_proba(fc[X])[:, 1]), 4))
    best = max(runs, key=lambda k: runs[k]['auc_2023'])
    fc['e'] = CANDIDATE_MODELS[best].predict_proba(fc[X])[:, 1]

    # violation model q, fit as in train_models.py, scored on every candidate
    chosen = json.loads((RESULTS / 'ml_results.json').read_text())['chosen']
    vrows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    tr = tm.train_rows(vrows, TEST_SCORING_DATE)
    q = tm.GRID[chosen['family']][1](chosen['params']).fit(tr[X], tr.y)
    fc['q'] = q.predict_proba(fc[X])[:, 1]
    fc['y'] = (lab.n_high.reindex(fc.index) >= SERIOUS).astype(float).where(fc.inspected.eq(1))
    n_insp = int(fc.inspected.sum())
    k = int(np.ceil(n_insp / 2))
    cutoff = float(np.sort(fc.q[fc.inspected.eq(1)].to_numpy())[::-1][k - 1])

    order = fc.e.sort_values(ascending=False).index
    fc['pred_inspected'] = fc.index.isin(order[:n_insp])
    pni = fc[~fc.pred_inspected]
    flagged = pni.q >= cutoff
    split = {}
    for name, mask in (('no routine history', pni.n_routine.eq(0)), ('1+ past routine inspections', pni.n_routine.gt(0))):
        split[name] = dict(n=int(mask.sum()), flagged=int((flagged & mask).sum()))
    labeled = fc[fc.inspected.eq(1)]
    out = dict(
        propensity=dict(fit_years=list(FIT_YEARS), features=X, runs=runs, selected=best,
                        rows={int(y): dict(n=int(len(g)), inspected_rate=round(float(g.inspected.mean()), 3))
                              for y, g in rows.groupby('year')},
                        test_note='inspected = in the linked test set (Comprehensive, Aug 11 2025 - Jul 31 2026, linked '
                                  'to an old-system ID); this undercounts inspections, so test AUC is approximate'),
        test_scoring_date=TEST_SCORING_DATE, candidates=int(len(fc)), test_inspected=n_insp, K=k,
        violation_model=chosen, q_cutoff=round(cutoff, 5),
        predicted_not_inspected=dict(n=int(len(pni)), flagged=int(flagged.sum()),
                                     flagged_share=round(float(flagged.mean()), 4), by_history=split,
                                     actually_inspected=int(pni.inspected.sum())),
        q_at_cutoff=dict(predicted_not_inspected_but_inspected=q_stats(labeled[~labeled.pred_inspected], cutoff),
                         predicted_inspected=q_stats(labeled[labeled.pred_inspected], cutoff)),
        notes=['e is weak (see AUC); predicted-not-inspected is a ranking of e, not a confident prediction',
               'q is trained only on inspected restaurants; its scores for never-inspected restaurants are extrapolation'])
    (RESULTS / 'line2_propensity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
