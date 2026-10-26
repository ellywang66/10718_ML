"""Train and select ML models on the same split and metric as the non-ML baselines.

Training rows: one per (restaurant, scoring year t), t = 2016..2024. Features at Jan 1 of t (ml_features.py);
label = serious violator (2+ high-risk violations) at the restaurant's first routine inspection in calendar year t,
the same label as the development years. A row is usable for a model scored on date T only if its label report
was known before T (inspection date + 30 days < T).

Selection: for each development year V in (2023, 2024), fit on rows with t < V whose label was known before
Jan 1 of V, and rank that year's development set. The configuration with the best mean recall in the top 50%
over the two years (ties: mean AUC) is the final model. It is refit on every row known before Jan 1, 2025 and
scored once on the held-out test set, together with the history-rate baseline. Writes results/ml_*.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import history
import ml_features
import splits
from baselines import order
from config import DEV_YEARS, REPORT_LAG_DAYS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE
from run_baselines import evaluate

TRAIN_YEARS = range(2016, 2025)   # 2014-2015 serve only as history
LAG = pd.Timedelta(days=REPORT_LAG_DAYS)
X = ml_features.FEATURES


def imputed(model):
    """Median imputation with missing-value indicators, then the model."""
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), model)


GRID = {
    'logistic_regression': (
        dict(penalty=['l2', 'l1'], C=[0.001, 0.01, 0.1, 1, 10]),
        lambda p: make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                LogisticRegression(solver='liblinear', max_iter=5000, **p))),
    'decision_tree': (
        dict(max_depth=[3, 5, 8], min_samples_leaf=[50, 200]),
        lambda p: imputed(DecisionTreeClassifier(random_state=SEED, **p))),
    'random_forest': (
        dict(max_depth=[5, 10, None], min_samples_leaf=[10, 50], max_features=['sqrt', 0.5]),
        lambda p: imputed(RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED, **p))),
    'gradient_boosting': (
        dict(learning_rate=[0.03, 0.1], max_leaf_nodes=[15, 31], max_iter=[100, 300], l2_regularization=[0, 1]),
        lambda p: HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **p)),
}


def configs():
    for family, (grid, make) in GRID.items():
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            yield family, dict(zip(keys, values)), make


def labeled(ins, year):
    lab = splits.dev_set(ins, year)
    f = ml_features.build(ins, lab.id.tolist(), f'{year}-01-01')
    f['y'] = (lab.set_index('id').n_high.reindex(f.index) >= SERIOUS).astype(int).to_numpy()
    f['label_known'] = (lab.set_index('id').date.reindex(f.index) + LAG).to_numpy()
    f['year'] = year
    return f


def train_rows(rows, scoring_date):
    t = pd.Timestamp(scoring_date)
    return rows[(rows.year < t.year) & (rows.label_known < t)]


def rank_eval(score, f, y):
    g = f[['last_result', 'overdue']].copy()
    g['score'] = score
    o = order(g, 'score')
    return o, evaluate(o, y)


def main():
    t0 = time.time()
    ins = history.load_inspections()
    rows = pd.concat([labeled(ins, y) for y in TRAIN_YEARS])
    print('training rows by year:', rows.groupby('year').y.agg(['size', 'mean']).round(3).to_dict('index'))

    # model selection on the development years
    runs = []
    for family, params, make in configs():
        r = dict(family=family, params=params)
        for v in DEV_YEARS:
            tr, dv = train_rows(rows, f'{v}-01-01'), rows[rows.year == v]
            m = make(params).fit(tr[X], tr.y)
            _, ev = rank_eval(m.predict_proba(dv[X])[:, 1], dv, dv.y.to_numpy())
            r[f'dev{v}'] = ev
        r['dev_mean_recall_at_50'] = float(np.mean([r[f'dev{v}']['recall_at_50'] for v in DEV_YEARS]))
        r['dev_mean_auc'] = float(np.mean([r[f'dev{v}']['auc'] for v in DEV_YEARS]))
        runs.append(r)
        print(f"{family:20s} {params} dev recall@50 {r['dev_mean_recall_at_50']:.3f}")
    key = lambda r: (r['dev_mean_recall_at_50'], r['dev_mean_auc'])
    chosen = max(runs, key=key)
    family_best = {fam: max((r for r in runs if r['family'] == fam), key=key) for fam in GRID}

    # baseline on the development years, computed the same way
    base_dev = {v: rank_eval(rows[rows.year == v].history_rate.to_numpy(), rows[rows.year == v],
                             rows[rows.year == v].y.to_numpy())[1] for v in DEV_YEARS}

    # held-out test: refit each family's dev-best config on everything known before the test scoring date
    test_lab = splits.test_set(ins)
    ft = ml_features.build(ins, test_lab.id.tolist(), TEST_SCORING_DATE)
    yt = (test_lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    month = test_lab.set_index('id').date.reindex(ft.index)
    tr = train_rows(rows, TEST_SCORING_DATE)
    make = {fam: mk for fam, (_, mk) in GRID.items()}
    ranks = pd.DataFrame(index=ft.index)
    test = {}
    o, test['history_rate_rule'] = rank_eval(ft.history_rate.to_numpy(), ft, yt)
    ranks['rank_history_rate_rule'] = np.argsort(o) + 1
    for fam, r in family_best.items():
        m = make[fam](r['params']).fit(tr[X], tr.y)
        o, test[fam] = rank_eval(m.predict_proba(ft[X])[:, 1], ft, yt)
        ranks[f'rank_{fam}'] = np.argsort(o) + 1
    ranks['serious_violator'] = yt

    # bootstrap the recall difference, chosen model minus the history-rate rule
    rng = np.random.default_rng(SEED)
    k = int(np.ceil(len(yt) / 2))
    top_m = (ranks[f"rank_{chosen['family']}"] <= k).to_numpy()
    top_b = (ranks['rank_history_rate_rule'] <= k).to_numpy()
    diffs = []
    for _ in range(4000):
        i = rng.integers(0, len(yt), len(yt))
        diffs.append(100 * ((yt[i] * top_m[i]).sum() - (yt[i] * top_b[i]).sum()) / yt[i].sum())
    diffs = np.array(diffs)

    # where does the model do worse? recall within subgroups of the test set (same global top K)
    ft['_y'], ft['_m'], ft['_b'] = yt, top_m, top_b
    nr = ft.n_routine
    groups = {
        'label in Aug-Dec 2025': month.dt.year.eq(2025).to_numpy(),
        'label in Jan-Jul 2026': month.dt.year.eq(2026).to_numpy(),
        'City of Pittsburgh': ft.municipal.str.contains('Pittsburgh', case=False).to_numpy(),
        'outside Pittsburgh': ~ft.municipal.str.contains('Pittsburgh', case=False).to_numpy(),
        'chain': ft.is_chain.eq(1).to_numpy(), 'independent': ft.is_chain.eq(0).to_numpy(),
        'with liquor': ft.has_liquor.eq(1).to_numpy(), 'without liquor': ft.has_liquor.eq(0).to_numpy(),
        'no routine history': nr.eq(0).to_numpy(), '1-3 past routine inspections': nr.between(1, 3).to_numpy(),
        '4+ past routine inspections': (nr >= 4).to_numpy(),
    }
    sub = {}
    for name, mask in groups.items():
        g = ft[mask]
        pos = int(g._y.sum())
        sub[name] = dict(n=int(mask.sum()), serious=pos,
                         model_recall=float((g._y & g._m).sum() / pos) if pos else None,
                         baseline_recall=float((g._y & g._b).sum() / pos) if pos else None,
                         model_share_in_top=float(g._m.mean()), baseline_share_in_top=float(g._b.mean()))

    # inspected vs. not inspected: how the test set differs from the full Jan 1, 2025 candidate list
    cand = splits.candidates(ins, TEST_SCORING_DATE)
    fc = ml_features.build(ins, cand, TEST_SCORING_DATE)
    insp = fc.index.isin(ft.index)
    cols = ['history_rate', 'last_result', 'overdue', 'days_since_last_routine', 'n_routine']
    inspected_vs_not = {'n_inspected': int(insp.sum()), 'n_not_inspected': int((~insp).sum()),
                        'inspected_mean': fc[insp][cols].mean().round(3).to_dict(),
                        'not_inspected_mean': fc[~insp][cols].mean().round(3).to_dict()}

    out = dict(
        train_rows={int(y): dict(n=int(len(g)), serious_rate=round(float(g.y.mean()), 3)) for y, g in rows.groupby('year')},
        n_test_train_rows=int(len(tr)), features=X, n_configs=len(runs),
        chosen=dict(family=chosen['family'], params=chosen['params']),
        family_best={f: dict(params=r['params'], dev2023=r['dev2023'], dev2024=r['dev2024']) for f, r in family_best.items()},
        baseline_dev={str(v): e for v, e in base_dev.items()},
        test=test,
        difference_vs_baseline=dict(points=round(100 * (test[chosen['family']]['recall_at_50'] - test['history_rate_rule']['recall_at_50']), 1),
                                    bootstrap_se=round(float(diffs.std()), 2),
                                    ci95=[round(float(x), 1) for x in np.percentile(diffs, [2.5, 97.5])]),
        subgroups=sub, inspected_vs_not=inspected_vs_not, runtime_s=round(time.time() - t0))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'ml_results.json').write_text(json.dumps(out, indent=1, default=str))
    (RESULTS / 'ml_runs.json').write_text(json.dumps(runs, indent=1, default=str))
    ranks.to_csv(RESULTS / 'ml_test_rankings.csv')
    print(json.dumps({k: out[k] for k in ('chosen', 'test', 'difference_vs_baseline')}, indent=1, default=str))


if __name__ == '__main__':
    main()
