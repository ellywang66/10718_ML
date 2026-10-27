"""Score every non-ML baseline rule over the candidate set at the development and test scoring dates.

Each rule ranks all candidates at a scoring date. Metrics come from evaluate.py at C = the number of
candidates ACHD inspected within the horizon (results/capacity.json). The legacy conditional metric
(recall among inspected restaurants at K = 50% of them) is kept under `legacy_recall_at_50_inspected`,
because it is still the selection criterion of the rules and of the ML outcome model; nothing here names a
primary metric. Outputs in results/.
"""
import json

import numpy as np
import pandas as pd

from src import evaluate
from src import panel
from src.models.baselines import MISSING_FIRST, RULES, ranks
from src.config import DEV_YEARS, RESULTS, SEED, TEST_SCORING_DATE

SELECTION_METRIC = 'legacy_recall_at_50_inspected'
BOOTSTRAP_FIELDS = ('lower_bound_yield', 'conditional_recall')


def legacy_recall_at_50(rows: pd.DataFrame, score_col: str, missing_first: bool) -> float:
    """Recall among inspected rows when the inspected rows alone are ranked and the top half is taken."""
    ins = rows[rows.inspected].set_index('id')
    r = ranks(ins, score_col, missing_first)
    k = int(np.ceil(len(ins) / 2))
    top = r <= k
    return float((ins.serious.to_numpy() & top).sum() / ins.serious.sum())


def score_date(rows: pd.DataFrame, bootstrap: bool) -> tuple[dict, pd.DataFrame]:
    """Metrics of every rule at one scoring date plus the rank table over candidates."""
    c = int(rows.inspected.sum())
    ins, ser = rows.inspected.to_numpy(), rows.serious.to_numpy()
    res = {'n_candidates': len(rows), 'C': c, 'serious_among_inspected': int(ser.sum()), 'rules': {}}
    rank_table = rows[['id']].copy()
    f = rows.set_index('id')
    for name, (col, _) in RULES.items():
        mf = name in MISSING_FIRST
        r = ranks(f, col, mf)
        m = evaluate.to_dict(evaluate.metrics(r, ins, ser, c))
        m[SELECTION_METRIC] = legacy_recall_at_50(rows, col, mf)
        if bootstrap:
            m['bootstrap'] = {k: evaluate.bootstrap_metric(k, r, ins, ser, c, seed=SEED) for k in BOOTSTRAP_FIELDS}
        res['rules'][name] = m
        rank_table[f'rank_{name}'] = r
    rank_table['inspected'] = ins
    rank_table['serious'] = ser
    return res, rank_table


def table(results: dict, dates: list[str]) -> str:
    """Markdown table of lower-bound yield, conditional recall, and the legacy metric per date."""
    head = '| Rule | ' + ' | '.join(f'{d} yield@C | {d} cond. recall@C | {d} legacy recall@50%' for d in dates) + ' |'
    lines = [head, '|---|' + '---:|' * (3 * len(dates))]
    for name in RULES:
        cells = []
        for d in dates:
            m = results[d]['rules'][name]
            cells += [f"{m['lower_bound_yield']:.3f}", f"{m['conditional_recall']:.1%}", f"{m[SELECTION_METRIC]:.1%}"]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    achd = ' | '.join(f"{results[d]['rules']['history_rate']['achd_yield']:.3f} (C={results[d]['C']})" for d in dates)
    lines.append(f'\nACHD observed yield at C: {achd}')
    return '\n'.join(lines)


def main() -> None:
    p = panel.load()
    dates = [f'{y}-01-01' for y in DEV_YEARS] + [TEST_SCORING_DATE]
    results = {'rules': {k: v[1] for k, v in RULES.items()}, 'selection_metric': SELECTION_METRIC,
               'test_C': 'linked_candidates (see results/capacity.json)'}
    for d in dates:
        results[d], rank_table = score_date(panel.at(p, d), bootstrap=(d == TEST_SCORING_DATE))
    dev_mean = {k: float(np.mean([results[f'{y}-01-01']['rules'][k][SELECTION_METRIC] for y in DEV_YEARS])) for k in RULES}
    results['chosen_on_dev'] = max(dev_mean, key=dev_mean.get)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'baseline_results.json').write_text(json.dumps(results, indent=1))
    (RESULTS / 'results_table.md').write_text(table(results, dates) + '\n')
    rank_table.to_csv(RESULTS / 'test_rankings.csv', index=False)
    print('chosen on development dates by', SELECTION_METRIC + ':', results['chosen_on_dev'])
    print(table(results, dates))


if __name__ == '__main__':
    main()
