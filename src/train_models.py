"""Train and select ML outcome models q(x) = P(serious | inspected, x) on the panel, and score the candidate set.

Training rows: panel rows with inspected = 1 at the Jan 1 scoring dates (2016 on). A row is usable for a model
applied at date T only if its label was known before T (first routine inspection date + 30 days < T).

Selection: for each development date V, fit on usable rows with scoring date before V and rank V's candidates.
The criterion is the legacy conditional metric (recall among inspected candidates at K = 50% of them), recorded
in the output as `q_selection_metric`; it is kept for continuity, not chosen as the task metric. The chosen
configuration is refit on everything usable before the test date and scored once on the test candidates.
Writes results/ml_results.json, results/ml_runs.json, results/ml_test_rankings.csv, and results/q_scores.parquet
(the chosen model's q for every candidate at the development and test dates, read by train_propensity.py).
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

import evaluate
import ml_features
import panel
from baselines import ranks
from config import DEV_YEARS, REPORT_LAG_DAYS, RESULTS, SEED, TEST_SCORING_DATE
from run_baselines import BOOTSTRAP_FIELDS, SELECTION_METRIC, legacy_recall_at_50

LAG = pd.Timedelta(days=REPORT_LAG_DAYS)
X = ml_features.FEATURES
Q_SCORES = RESULTS / 'q_scores.parquet'


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


def configs(families=GRID):
    """Every (family, params, constructor) in the grid."""
    for family, (grid, make) in families.items():
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            yield family, dict(zip(keys, values)), make


def training_rows(p: pd.DataFrame) -> pd.DataFrame:
    """Inspected rows at the Jan 1 scoring dates with their label-known date."""
    jan = p[p.scoring_date.dt.month.eq(1) & p.scoring_date.dt.day.eq(1)]
    parts = [panel.at(jan, t) for t in sorted(jan.scoring_date.unique())]
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows.inspected].copy()
    rows['label_known'] = rows.first_routine_date + LAG
    rows['y'] = rows.serious.astype(int)
    return rows


def usable(rows: pd.DataFrame, scoring_date) -> pd.DataFrame:
    """Training rows whose scoring date precedes T and whose label was known before T."""
    t = pd.Timestamp(scoring_date)
    return rows[(rows.scoring_date < t) & (rows.label_known < t)]


def score_candidates(score: np.ndarray, cand: pd.DataFrame, bootstrap: bool = False) -> tuple[np.ndarray, dict]:
    """Rank all candidates by a score and compute every metric at C = number inspected."""
    f = cand.set_index('id')[['last_result', 'overdue']].copy()
    f['score'] = score
    r = ranks(f, 'score')
    ins, ser = cand.inspected.to_numpy(), cand.serious.to_numpy()
    c = int(ins.sum())
    m = evaluate.to_dict(evaluate.metrics(r, ins, ser, c))
    tmp = cand.copy(); tmp['score'] = score
    m[SELECTION_METRIC] = legacy_recall_at_50(tmp, 'score', False)
    if bootstrap:
        m['bootstrap'] = {k: evaluate.bootstrap_metric(k, r, ins, ser, c, seed=SEED) for k in BOOTSTRAP_FIELDS}
    return r, m


def subgroups(cand: pd.DataFrame, top_m: np.ndarray, top_b: np.ndarray) -> dict:
    """Where the model and the baseline differ: recall among inspected and share of the top C, by subgroup."""
    nr = cand.n_routine
    groups = {
        'City of Pittsburgh': cand.municipal.str.contains('Pittsburgh', case=False).to_numpy(),
        'outside Pittsburgh': ~cand.municipal.str.contains('Pittsburgh', case=False).to_numpy(),
        'chain': cand.is_chain.eq(1).to_numpy(), 'independent': cand.is_chain.eq(0).to_numpy(),
        'with liquor': cand.has_liquor.eq(1).to_numpy(), 'without liquor': cand.has_liquor.eq(0).to_numpy(),
        'no routine history': nr.eq(0).to_numpy(), '1-3 past routine inspections': nr.between(1, 3).to_numpy(),
        '4+ past routine inspections': (nr >= 4).to_numpy(),
    }
    out = {}
    y = cand.serious.to_numpy()
    for name, mask in groups.items():
        pos = int(y[mask].sum())
        out[name] = dict(n_candidates=int(mask.sum()), n_inspected=int(cand.inspected.to_numpy()[mask].sum()), serious=pos,
                         model_recall=float((y & top_m)[mask].sum() / pos) if pos else None,
                         baseline_recall=float((y & top_b)[mask].sum() / pos) if pos else None,
                         model_share_in_top=float(top_m[mask].mean()), baseline_share_in_top=float(top_b[mask].mean()))
    return out


def main() -> None:
    t0 = time.time()
    p = panel.load()
    rows = training_rows(p)
    dev_dates = [f'{y}-01-01' for y in DEV_YEARS]
    print('training rows by scoring date:', rows.groupby(rows.scoring_date.dt.year).y.agg(['size', 'mean']).round(3).to_dict('index'))

    # model selection on the development dates
    dev_cand = {d: panel.at(p, d) for d in dev_dates}
    runs, dev_scores = [], {}
    for family, params, make in configs():
        r = dict(family=family, params=params)
        for d in dev_dates:
            tr = usable(rows, d)
            m = make(params).fit(tr[X], tr.y)
            q = m.predict_proba(dev_cand[d][X])[:, 1]
            r[f'dev{d[:4]}'] = score_candidates(q, dev_cand[d])[1]
            dev_scores[(family, json.dumps(params, sort_keys=True), d)] = q
        r['dev_mean_selection'] = float(np.mean([r[f'dev{d[:4]}'][SELECTION_METRIC] for d in dev_dates]))
        r['dev_mean_lower_bound_yield'] = float(np.mean([r[f'dev{d[:4]}']['lower_bound_yield'] for d in dev_dates]))
        runs.append(r)
        print(f"{family:20s} {params} dev {SELECTION_METRIC} {r['dev_mean_selection']:.3f}")
    key = lambda r: (r['dev_mean_selection'], r['dev_mean_lower_bound_yield'])
    chosen = max(runs, key=key)
    family_best = {fam: max((r for r in runs if r['family'] == fam), key=key) for fam in GRID}

    # the history-rate rule on the same candidate sets
    base_dev = {d: score_candidates(dev_cand[d].history_rate.to_numpy(), dev_cand[d])[1] for d in dev_dates}

    # held-out test: refit each family's dev-best config on everything usable before the test date
    test_cand = panel.at(p, TEST_SCORING_DATE)
    tr = usable(rows, TEST_SCORING_DATE)
    make = {fam: mk for fam, (_, mk) in GRID.items()}
    rank_table = test_cand[['id', 'inspected', 'serious']].copy()
    test, test_scores = {}, {}
    r_b, test['history_rate_rule'] = score_candidates(test_cand.history_rate.to_numpy(), test_cand, bootstrap=True)
    rank_table['rank_history_rate_rule'] = r_b
    for fam, r in family_best.items():
        m = make[fam](r['params']).fit(tr[X], tr.y)
        test_scores[fam] = m.predict_proba(test_cand[X])[:, 1]
        r_m, test[fam] = score_candidates(test_scores[fam], test_cand, bootstrap=(fam == chosen['family']))
        rank_table[f'rank_{fam}'] = r_m

    # q scores of the chosen configuration for every candidate at the development and test dates
    q_parts = []
    for d in dev_dates:
        q_parts.append(pd.DataFrame({'scoring_date': pd.Timestamp(d), 'id': dev_cand[d].id,
                                     'q': dev_scores[(chosen['family'], json.dumps(chosen['params'], sort_keys=True), d)]}))
    q_parts.append(pd.DataFrame({'scoring_date': pd.Timestamp(TEST_SCORING_DATE), 'id': test_cand.id, 'q': test_scores[chosen['family']]}))
    pd.concat(q_parts, ignore_index=True).to_parquet(Q_SCORES, index=False)

    # bootstrap of the chosen model minus the rule on the test date
    c = int(test_cand.inspected.sum())
    r_m = rank_table[f"rank_{chosen['family']}"].to_numpy()
    ins, ser = test_cand.inspected.to_numpy(), test_cand.serious.to_numpy()
    diff = {}
    for field in BOOTSTRAP_FIELDS:
        def stat(idx, field=field):
            return (getattr(evaluate.metrics(r_m[idx], ins[idx], ser[idx], c), field)
                    - getattr(evaluate.metrics(r_b[idx], ins[idx], ser[idx], c), field))
        diff[field] = dict(point=float(test[chosen['family']][field] - test['history_rate_rule'][field]),
                           **evaluate.bootstrap(stat, len(r_m), seed=SEED))

    out = dict(
        train_rows={str(y): dict(n=int(len(g)), serious_rate=round(float(g.y.mean()), 3)) for y, g in rows.groupby(rows.scoring_date.dt.year)},
        n_test_train_rows=int(len(tr)), features=X, n_configs=len(runs), q_selection_metric=SELECTION_METRIC,
        chosen=dict(family=chosen['family'], params=chosen['params']),
        family_best={f: {k: r[k] for k in r if k != 'family'} for f, r in family_best.items()},
        baseline_dev=base_dev, test=test, test_C=c, difference_vs_baseline=diff,
        subgroups=subgroups(test_cand, evaluate.top_mask(r_m, c), evaluate.top_mask(r_b, c)),
        runtime_s=round(time.time() - t0))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'ml_results.json').write_text(json.dumps(out, indent=1, default=str))
    (RESULTS / 'ml_runs.json').write_text(json.dumps(runs, indent=1, default=str))
    rank_table.to_csv(RESULTS / 'ml_test_rankings.csv', index=False)
    print(json.dumps({k: out[k] for k in ('chosen', 'test', 'difference_vs_baseline')}, indent=1, default=str))


if __name__ == '__main__':
    main()
