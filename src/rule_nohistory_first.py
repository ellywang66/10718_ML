"""Rule variant: restaurants with no routine history first, then the history-rate rule.

Within the no-history group and among the rest, the order is the history-rate rule's, with its tie-break (high-risk
count at the last routine inspection, then overdue ratio). Scored on the same development years and test set as
run_baselines.py and train_models.py; paired bootstrap differences (4,000 restaurant resamples, same seed as
report_numbers.py) against the plain history-rate rule and the chosen model (results/ml_test_rankings.csv).
Writes results/rule_nohistory_first.json.
"""
import json

import numpy as np
import pandas as pd

import history
import ml_features
import splits
import train_models as tm
from config import DEV_YEARS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE
from report_numbers import lead, pct


def variant_score(f):
    """History rate, lifted above every restaurant with history when there is no routine history."""
    return f.history_rate.to_numpy() + 10 * f.n_routine.eq(0).to_numpy()


def ranks(o):
    return np.argsort(o) + 1


def main():
    chosen = json.loads((RESULTS / 'ml_results.json').read_text())['chosen']
    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    out = dict(rule='no-routine-history restaurants first, then history rate (rule tie-break)', dev={})
    for v in DEV_YEARS:
        dv = rows[rows.year == v]
        out['dev'][str(v)] = dict(variant=pct(tm.rank_eval(variant_score(dv), dv, dv.y.to_numpy())[1]['recall_at_50']),
                                  history_rate_rule=pct(tm.rank_eval(dv.history_rate.to_numpy(), dv, dv.y.to_numpy())[1]['recall_at_50']),
                                  no_history=int(dv.n_routine.eq(0).sum()), n=int(len(dv)))
    out['dev']['mean_variant'] = round(float(np.mean([out['dev'][str(v)]['variant'] for v in DEV_YEARS])), 1)
    out['dev']['mean_history_rate_rule'] = round(float(np.mean([out['dev'][str(v)]['history_rate_rule'] for v in DEV_YEARS])), 1)

    lab = splits.test_set(ins)
    ft = ml_features.build(ins, lab.id.tolist(), TEST_SCORING_DATE)
    y = (lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    o_v, ev = tm.rank_eval(variant_score(ft), ft, y)
    o_b, _ = tm.rank_eval(ft.history_rate.to_numpy(), ft, y)
    rk = pd.read_csv(RESULTS / 'ml_test_rankings.csv', dtype={'id': str}).set_index('id').reindex(ft.index)
    k = int(np.ceil(len(y) / 2))
    top_v, top_b = ranks(o_v) <= k, ranks(o_b) <= k
    top_m = (rk[f"rank_{chosen['family']}"] <= k).to_numpy()
    rng = np.random.default_rng(SEED)
    out['test'] = dict(n=int(len(y)), serious=int(y.sum()), K=k, R_at_K=pct(ev['recall_at_50']), P_at_K=pct(ev['precision_at_50']),
                       R_at_K30=pct(ev['recall_at_30']), auc=round(ev['auc'], 3), found_at_K=int(y[top_v].sum()),
                       variant_minus_history_rate_rule=lead(top_v, top_b, y, rng),
                       chosen_model_minus_variant=lead(top_m, top_v, y, rng), chosen_model=chosen)
    nr = ft.n_routine
    sub = {}
    for name, mask in (('no routine history', nr.eq(0)), ('1-3 past routine inspections', nr.between(1, 3)),
                       ('4+ past routine inspections', nr >= 4)):
        m = mask.to_numpy(); pos = int(y[m].sum())
        sub[name] = dict(n=int(m.sum()), serious=pos, variant_found=int(y[m & top_v].sum()),
                         variant_recall=pct(y[m & top_v].sum() / pos), rule_recall=pct(y[m & top_b].sum() / pos),
                         model_recall=pct(y[m & top_m].sum() / pos))
    out['test']['subgroups'] = sub
    (RESULTS / 'rule_nohistory_first.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
