"""L90 within each test subgroup for the model chosen on L90 and the history-rate rule.

Each list ranks only the subgroup's restaurants; L90 = share of that list needed to find 90% of the subgroup's
serious violators. Saved = rule minus model in points, with a 95% interval from 2,000 bootstrap resamples of the
subgroup's restaurants. Writes results/subgroup_l90.json.
"""
import json

import numpy as np
import pandas as pd

import history
import ml_features
import splits
import train_models as tm
from config import RESULTS, SERIOUS, TEST_SCORING_DATE
from recall_targets import list_share, rank

N_BOOT = 2000


def main():
    chosen = json.loads((RESULTS / 'recall_targets.json').read_text())['chosen']
    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    tr = tm.train_rows(rows, TEST_SCORING_DATE)
    m = tm.GRID[chosen['family']][1](chosen['params']).fit(tr[tm.X], tr.y)
    lab = splits.test_set(ins)
    ft = ml_features.build(ins, lab.id.tolist(), TEST_SCORING_DATE)
    y = (lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    ft['q'] = m.predict_proba(ft[tm.X])[:, 1]

    month = lab.set_index('id').date.reindex(ft.index)
    nr, pit = ft.n_routine, ft.municipal.str.contains('Pittsburgh', case=False)
    groups = [('Time', 'Aug-Dec 2025', month.dt.year.eq(2025)), ('Time', 'Jan-Jul 2026', month.dt.year.eq(2026)),
              ('Location', 'Pittsburgh', pit), ('Location', 'Elsewhere', ~pit),
              ('Type', 'Chain', ft.is_chain.eq(1)), ('Type', 'Independent', ft.is_chain.eq(0)),
              ('Past routine inspections', 'None', nr.eq(0)), ('Past routine inspections', '1-3', nr.between(1, 3)),
              ('Past routine inspections', '4+', nr >= 4)]
    rng = np.random.default_rng(0)
    out = []
    for block, name, mask in groups:
        mask = mask.to_numpy()
        f, yy = ft[mask], y[mask]
        o_m, o_b = rank(f.q.to_numpy(), f), rank(f.history_rate.to_numpy(), f)
        l_m, l_b = list_share(o_m, yy, 0.9), list_share(o_b, yy, 0.9)
        pos_m, pos_b = np.argsort(o_m), np.argsort(o_b)
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(yy), len(yy))
            boot.append(100 * (list_share(np.argsort(pos_b[i], kind='stable'), yy[i], 0.9)
                               - list_share(np.argsort(pos_m[i], kind='stable'), yy[i], 0.9)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(dict(block=block, subgroup=name, n=int(len(yy)), serious=int(yy.sum()),
                        model_l90=round(100 * l_m, 1), rule_l90=round(100 * l_b, 1),
                        saved=round(100 * (l_b - l_m), 1), ci95=[round(float(lo), 1), round(float(hi), 1)]))
        print(out[-1])
    (RESULTS / 'subgroup_l90.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
