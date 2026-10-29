"""Models judged by the share of the ranked list needed to reach a fixed recall of serious violators.

For a ranking of N inspected restaurants with P serious violators, list_share_at_R is the smallest k / N such that
the top k contain at least ceil(R * P) serious violators (lower is better). The primary target is R = 90%; 80% is
also reported. Same training rows, grids and development years as train_models.py, plus the five-feature
logistic regression (train_lr5.py). Selection: lowest mean development list_share_at_90 (ties: higher mean AUC).
Each family's best configuration and the chosen model are refit on rows known before the test scoring date and
scored once on the test set, with the history-rate rule and most-overdue-first. Writes results/recall_targets.json.
"""
import json

import numpy as np
import pandas as pd

import history
import ml_features
import splits
import train_models as tm
from baselines import order
from run_baselines import evaluate
from config import DEV_YEARS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE
from train_lr5 import X5

TARGETS = (0.8, 0.9)
N_BOOT = 4000


def list_share(o, y, r):
    hits = np.cumsum(y[o])
    return float((np.searchsorted(hits, np.ceil(r * y.sum())) + 1) / len(y))


def metrics(o, y):
    out = {f'list_share_at_{int(r * 100)}': list_share(o, y, r) for r in TARGETS}
    out.update(evaluate(o, y))
    return out


def rank(score, f):
    g = f[['last_result', 'overdue']].copy()
    g['score'] = score
    return order(g, 'score')


def rule_orders(f):
    g = f[['last_result', 'overdue']].copy()
    g['score'] = f.overdue.to_numpy()
    return {'history_rate_rule': rank(f.history_rate.to_numpy(), f),
            'most_overdue_first': order(g, 'score', missing_first=True)}


def candidates():
    for family, params, make in tm.configs():
        yield family, params, make, tm.X
    _, make = tm.GRID['logistic_regression']
    for family, params, _ in tm.configs():
        if family == 'logistic_regression':
            yield 'lr_5_features', params, make, X5


def main():
    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])

    runs = []
    for family, params, make, X in candidates():
        r = dict(family=family, params=params)
        for v in DEV_YEARS:
            tr, dv = tm.train_rows(rows, f'{v}-01-01'), rows[rows.year == v]
            m = make(params).fit(tr[X], tr.y)
            r[f'dev{v}'] = metrics(rank(m.predict_proba(dv[X])[:, 1], dv), dv.y.to_numpy())
        for k in ('list_share_at_90', 'list_share_at_80', 'recall_at_50', 'auc'):
            r[f'dev_mean_{k}'] = float(np.mean([r[f'dev{v}'][k] for v in DEV_YEARS]))
        runs.append(r)
        print(f"{family:20s} {params} dev share@90 {r['dev_mean_list_share_at_90']:.3f}")
    key = lambda r: (-r['dev_mean_list_share_at_90'], r['dev_mean_auc'])
    family_best = {fam: max((r for r in runs if r['family'] == fam), key=key) for fam in dict.fromkeys(r['family'] for r in runs)}
    chosen = max((r for r in runs if r['family'] != 'lr_5_features'), key=key)

    rules_dev = {}
    for v in DEV_YEARS:
        dv = rows[rows.year == v]
        rules_dev[str(v)] = {k: metrics(o, dv.y.to_numpy()) for k, o in rule_orders(dv).items()}

    test_lab = splits.test_set(ins)
    ft = ml_features.build(ins, test_lab.id.tolist(), TEST_SCORING_DATE)
    yt = (test_lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    tr = tm.train_rows(rows, TEST_SCORING_DATE)
    make = {fam: mk for fam, (_, mk) in tm.GRID.items()}
    make['lr_5_features'] = make['logistic_regression']
    feats = {fam: (X5 if fam == 'lr_5_features' else tm.X) for fam in family_best}
    orders = rule_orders(ft)
    for fam, r in family_best.items():
        m = make[fam](r['params']).fit(tr[feats[fam]], tr.y)
        orders[fam] = rank(m.predict_proba(ft[feats[fam]])[:, 1], ft)
    test = {k: metrics(o, yt) for k, o in orders.items()}

    # bootstrap by restaurant: list share saved by the chosen model against each rule (positive = model needs less)
    rng = np.random.default_rng(SEED)
    pos = {k: np.argsort(o) for k, o in orders.items()}   # position of each restaurant in each ranking
    diffs = {(b, r): [] for b in ('history_rate_rule', 'most_overdue_first') for r in TARGETS}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt))
        y = yt[i]
        for b in ('history_rate_rule', 'most_overdue_first'):
            for r in TARGETS:
                diffs[(b, r)].append(100 * (list_share(np.argsort(pos[b][i], kind='stable'), y, r)
                                            - list_share(np.argsort(pos[chosen['family']][i], kind='stable'), y, r)))
    saved = {f'{b}_at_{int(r * 100)}': dict(
        points=round(100 * (test[b][f'list_share_at_{int(r * 100)}'] - test[chosen['family']][f'list_share_at_{int(r * 100)}']), 1),
        bootstrap_se=round(float(np.std(d)), 2), ci95=[round(float(x), 1) for x in np.percentile(d, [2.5, 97.5])])
        for (b, r), d in diffs.items()}

    out = dict(targets=list(TARGETS), n_test=int(len(yt)), serious_test=int(yt.sum()),
               chosen=dict(family=chosen['family'], params=chosen['params']),
               family_best={f: dict(params=r['params'], **{k: r[k] for k in r if k.startswith('dev_mean_')})
                            for f, r in family_best.items()},
               rules_dev=rules_dev, test=test, list_share_saved_by_chosen=saved, runs=runs)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'recall_targets.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ('chosen', 'list_share_saved_by_chosen')}, indent=1, default=str))
    for k, v in test.items():
        print(f"{k:20s} share@80 {v['list_share_at_80']:.3f} share@90 {v['list_share_at_90']:.3f} "
              f"recall@50 {v['recall_at_50']:.3f} auc {v['auc']:.3f}")


if __name__ == '__main__':
    main()
