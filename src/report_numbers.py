"""Every number the report quotes, recomputed from the results files of the current pipeline.

Reads results/baseline_results.json, ml_runs.json, ml_results.json, ml_lr5_results.json, test_rankings.csv and
ml_test_rankings.csv (run run_baselines.py, train_models.py and train_lr5.py first); refits the chosen model once
for its feature importances. Leads are the chosen model's recall minus the history-rate rule's, in points, with
4,000 bootstrap resamples by restaurant (same seed as train_models.py). Writes results/report_numbers.json.
"""
import json

import numpy as np
import pandas as pd

import history
import ml_features
import splits
import train_models as tm
from config import DEV_YEARS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE

FAMILIES = {'logistic_regression': 'LR', 'random_forest': 'RF', 'gradient_boosting': 'GB', 'decision_tree': 'DT'}
RULES = ('history_rate', 'last_result', 'most_overdue')


def pct(x):
    return round(100 * float(x), 1)


def lead(top_m, top_b, y, rng):
    point = 100 * ((y * top_m).sum() - (y * top_b).sum()) / y.sum()
    diffs = []
    for _ in range(4000):
        i = rng.integers(0, len(y), len(y))
        diffs.append(100 * ((y[i] * top_m[i]).sum() - (y[i] * top_b[i]).sum()) / y[i].sum())
    return dict(points=round(float(point), 1), bootstrap_se=round(float(np.std(diffs)), 2),
                ci95=[round(float(x), 1) for x in np.percentile(diffs, [2.5, 97.5])])


def main():
    load = lambda f: json.loads((RESULTS / f).read_text())
    base, runs, ml, lr5 = load('baseline_results.json'), load('ml_runs.json'), load('ml_results.json'), load('ml_lr5_results.json')
    chosen = ml['chosen']; fam = chosen['family']
    out = {}

    n = base['test']['n']; P = base['test']['serious_violators']
    K, K30 = int(np.ceil(n / 2)), int(np.ceil(0.3 * n))
    out['1_test'] = dict(n=n, serious=P, K=K, K30=K30)

    rule_dev = float(np.mean([base[f'dev{v}']['rules']['history_rate']['recall_at_50'] for v in DEV_YEARS]))
    part4 = {}
    for f, short in FAMILIES.items():
        means = [r['dev_mean_recall_at_50'] for r in runs if r['family'] == f]
        part4[short] = dict(n_configs=len(means), worst_mean_dev=pct(min(means)), best_mean_dev=pct(max(means)))
    lr5_means = [r['dev_mean_recall_at_50'] for r in lr5['runs']]
    part4['LR5'] = dict(n_configs=len(lr5_means), worst_mean_dev=pct(min(lr5_means)), best_mean_dev=pct(max(lr5_means)))
    lr_means = [r['dev_mean_recall_at_50'] for r in runs if r['family'] == 'logistic_regression']
    out['2_dev_selection'] = dict(families=part4, rule_mean_dev=pct(rule_dev),
                                  lr_configs_below_rule=int(sum(m < rule_dev for m in lr_means)))

    key = lambda r: (r['dev_mean_recall_at_50'], r['dev_mean_auc'])
    ranked = sorted(runs, key=key, reverse=True)
    other_family = next(r for r in ranked if r['family'] != fam)
    out['3_chosen'] = dict(family=fam, params=chosen['params'], dev2023=pct(ranked[0]['dev2023']['recall_at_50']),
                           dev2024=pct(ranked[0]['dev2024']['recall_at_50']), mean_dev=pct(ranked[0]['dev_mean_recall_at_50']),
                           runner_up=dict(family=ranked[1]['family'], params=ranked[1]['params'],
                                          mean_dev=pct(ranked[1]['dev_mean_recall_at_50'])),
                           best_other_family=dict(family=other_family['family'], params=other_family['params'],
                                                  mean_dev=pct(other_family['dev_mean_recall_at_50'])))

    row = lambda d23, d24, t: dict(dev2023=pct(d23['recall_at_50']), dev2024=pct(d24['recall_at_50']),
                                   test_R_at_K=pct(t['recall_at_50']), test_P_at_K=pct(t['precision_at_50']),
                                   test_R_at_K30=pct(t['recall_at_30']), test_auc=round(t['auc'], 3))
    main_table = {}
    for f, short in FAMILIES.items():
        fb = ml['family_best'][f]
        main_table[short] = dict(params=fb['params'], **row(fb['dev2023'], fb['dev2024'], ml['test'][f]))
    main_table['LR5'] = dict(params=lr5['chosen'], **row(lr5['dev']['2023'], lr5['dev']['2024'], lr5['test']['lr5']))
    for r in RULES:
        main_table[f'rule_{r}'] = row(base['dev2023']['rules'][r], base['dev2024']['rules'][r], base['test']['rules'][r])
    out['4_main_table'] = main_table

    rk = pd.read_csv(RESULTS / 'ml_test_rankings.csv', dtype={'id': str}).set_index('id')
    rb = pd.read_csv(RESULTS / 'test_rankings.csv', dtype={'id': str}).set_index('id').reindex(rk.index)
    y = rk.serious_violator.to_numpy()
    top_m, top_b = (rk[f'rank_{fam}'] <= K).to_numpy(), (rk.rank_history_rate_rule <= K).to_numpy()
    top_m30, top_b30 = (rk[f'rank_{fam}'] <= K30).to_numpy(), (rk.rank_history_rate_rule <= K30).to_numpy()
    rng = np.random.default_rng(SEED)
    out['5_lead'] = dict(model_found_at_K=int(y[top_m].sum()), rule_found_at_K=int(y[top_b].sum()),
                         lead_at_K=lead(top_m, top_b, y, rng),
                         model_found_at_K30=int(y[top_m30].sum()), rule_found_at_K30=int(y[top_b30].sum()),
                         lead_at_K30_points=round(100 * float(y[top_m30].sum() - y[top_b30].sum()) / y.sum(), 1),
                         lr5_lead_at_K=lr5['difference_vs_baseline'],
                         check_train_models_lead=ml['difference_vs_baseline'])

    ins = history.load_inspections()
    lab = splits.test_set(ins)
    ft = ml_features.build(ins, lab.id.tolist(), TEST_SCORING_DATE).reindex(rk.index)
    month = lab.set_index('id').date.reindex(rk.index)
    nr = ft.n_routine
    groups = {
        'inspection Aug-Dec 2025': month.dt.year.eq(2025), 'inspection Jan-Jul 2026': month.dt.year.eq(2026),
        'City of Pittsburgh': ft.municipal.str.contains('Pittsburgh', case=False),
        'outside Pittsburgh': ~ft.municipal.str.contains('Pittsburgh', case=False),
        'chain': ft.is_chain.eq(1), 'independent': ft.is_chain.eq(0),
        'no routine history': nr.eq(0), '1-3 past routine inspections': nr.between(1, 3), '4+ past routine inspections': nr >= 4,
    }
    sub = {}
    for name, mask in groups.items():
        m = mask.to_numpy(); pos = int(y[m].sum())
        sub[name] = dict(n=int(m.sum()), serious=pos, model_found=int(y[m & top_m].sum()), rule_found=int(y[m & top_b].sum()),
                         model_recall=pct(y[m & top_m].sum() / pos) if pos else None,
                         rule_recall=pct(y[m & top_b].sum() / pos) if pos else None)
    no_hist, hist = nr.eq(0).to_numpy(), nr.gt(0).to_numpy()
    out['6_subgroups'] = dict(
        groups=sub,
        no_history_share_of_top_half=dict(model=pct(no_hist[top_m].mean()), rule=pct(no_hist[top_b].mean())),
        no_history_in_top_half=dict(model=int(no_hist[top_m].sum()), rule=int(no_hist[top_b].sum())),
        serious_found_any_history=dict(model=int(y[hist & top_m].sum()), rule=int(y[hist & top_b].sum())),
        serious_found_chains=dict(model=sub['chain']['model_found'], rule=sub['chain']['rule_found']))

    top_o = (rb.rank_most_overdue <= K).to_numpy()
    out['7_most_overdue_found_at_K'] = int(y[top_o].sum())

    tr = ml['train_rows']
    out['8_training_rows'] = dict(total=int(sum(v['n'] for v in tr.values())), years=f'{min(tr)}-{max(tr)}',
                                  rows_per_year_min=min(v['n'] for v in tr.values()), rows_per_year_max=max(v['n'] for v in tr.values()),
                                  serious_rate_min=min(v['serious_rate'] for v in tr.values()),
                                  serious_rate_max=max(v['serious_rate'] for v in tr.values()),
                                  known_before_test=ml['n_test_train_rows'])

    iv = ml['inspected_vs_not']
    out['9_inspected_vs_not'] = dict(candidates=iv['n_inspected'] + iv['n_not_inspected'], inspected=iv['n_inspected'],
                                     not_inspected=iv['n_not_inspected'], inspected_mean=iv['inspected_mean'],
                                     not_inspected_mean=iv['not_inspected_mean'])

    rows = pd.concat([tm.labeled(ins, yr) for yr in tm.TRAIN_YEARS])
    trn = tm.train_rows(rows, TEST_SCORING_DATE)
    model = tm.GRID[fam][1](chosen['params']).fit(trn[tm.X], trn.y)
    est = model[-1]
    if hasattr(est, 'feature_importances_'):
        names = model[0].get_feature_names_out(tm.X)
        imp = pd.Series(est.feature_importances_, index=names).sort_values(ascending=False)
        out['10_top5_importances'] = {k: round(float(v), 4) for k, v in imp.head(5).items()}
    (RESULTS / 'report_numbers.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
