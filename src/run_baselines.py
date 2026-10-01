"""Score every non-ML baseline on the development years and the held-out test set.

Metric: recall of serious violators (2+ high-risk violations) among the top K inspected restaurants, with
K = 50% (main) and 30%; precision at the same K; AUC. Outputs in results/.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import features
import history
import splits
from baselines import MISSING_FIRST, RULES, order, order_random_ties
from config import BUDGETS, DEV_YEARS, RANDOM_TIE_DRAWS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE


def evaluate(o, y):
    n = len(y); out = {}
    for b in BUDGETS:
        top = o[:int(np.ceil(b * n))]
        out[f'recall_at_{int(b * 100)}'] = float(y[top].sum() / y.sum())
        out[f'precision_at_{int(b * 100)}'] = float(y[top].mean())
    rank_score = np.empty(n); rank_score[o] = -np.arange(n)
    out['auc'] = float(roc_auc_score(y, rank_score))
    return out


def score_set(ins, labels, scoring_date, rng):
    f = features.build(ins, labels.id.tolist(), scoring_date)
    y = (labels.set_index('id').n_high.reindex(f.index) >= SERIOUS).astype(int).to_numpy()
    y_any = (labels.set_index('id').n_high.reindex(f.index) >= 1).astype(int).to_numpy()
    res = {'n': len(y), 'serious_violators': int(y.sum()), 'any_high_risk': int(y_any.sum()), 'rules': {}}
    ranks = pd.DataFrame(index=f.index)
    for name, (col, _) in RULES.items():
        mf = name in MISSING_FIRST
        o = order(f, col, mf)
        r = evaluate(o, y)
        r['any_high_risk_recall_at_50'] = evaluate(o, y_any)['recall_at_50']
        k = int(np.ceil(0.5 * len(y)))
        caps = [y[order_random_ties(f[col], rng, mf)[:k]].sum() / y.sum() for _ in range(RANDOM_TIE_DRAWS)]
        r['random_ties_recall_at_50'] = {'p2.5': float(np.quantile(caps, .025)), 'mean': float(np.mean(caps)),
                                         'p97.5': float(np.quantile(caps, .975))}
        res['rules'][name] = r
        rank = np.empty(len(o), int); rank[o] = np.arange(1, len(o) + 1); ranks[f'rank_{name}'] = rank
    ranks['serious_violator'] = y
    return res, ranks


def table(results):
    lines = ['| Baseline | Dev 2023 recall@50% | Dev 2024 recall@50% | Test recall@50% | Test precision@50% | '
             'Test recall@30% | Test AUC | Test recall@50%, random ties |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in RULES:
        d23, d24, t = (results[s]['rules'][name] for s in ('dev2023', 'dev2024', 'test'))
        rt = t['random_ties_recall_at_50']
        lines.append(f"| {name} | {d23['recall_at_50']:.1%} | {d24['recall_at_50']:.1%} | {t['recall_at_50']:.1%} | "
                     f"{t['precision_at_50']:.1%} | {t['recall_at_30']:.1%} | {t['auc']:.3f} | {rt['p2.5']:.1%}-{rt['p97.5']:.1%} |")
    return '\n'.join(lines)


def main():
    rng = np.random.default_rng(SEED)
    ins = history.load_inspections()
    results = {'rules': {k: v[1] for k, v in RULES.items()}}
    for year in DEV_YEARS:
        results[f'dev{year}'], _ = score_set(ins, splits.dev_set(ins, year), f'{year}-01-01', rng)
    results['test'], ranks = score_set(ins, splits.test_set(ins), TEST_SCORING_DATE, rng)
    dev_mean = {k: np.mean([results[f'dev{y}']['rules'][k]['recall_at_50'] for y in DEV_YEARS]) for k in RULES}
    results['chosen_on_dev'] = max(dev_mean, key=dev_mean.get)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'baseline_results.json').write_text(json.dumps(results, indent=1))
    (RESULTS / 'results_table.md').write_text(table(results) + '\n')
    ranks.to_csv(RESULTS / 'test_rankings.csv')
    for s in ('dev2023', 'dev2024', 'test'):
        print(s, 'n', results[s]['n'], 'serious', results[s]['serious_violators'])
    print('chosen on development years:', results['chosen_on_dev'])
    print(table(results))


if __name__ == '__main__':
    main()
