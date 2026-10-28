"""Line 1: inspections the chosen violation model saves at the history-rate rule's recall, spent on a random audit.

For each evaluation set (development years 2023 and 2024, held-out test), with K = 50% of inspected restaurants:
the rule's recall at K; K_model = the smallest number of top-ranked restaurants at which the chosen model (from
results/ml_results.json, fit the same way as in train_models.py) finds at least as many serious violators as the
rule finds in its top K; saved = K - K_model. Bootstrap SE by restaurant (resample restaurants, keep both
rankings). The saved inspections fund a uniform random audit of the scoring date's candidates (splits.candidates)
outside the model's top K_model. Expected serious violators found by the audit = slots x serious base rate among
inspected restaurants; this assumes uninspected candidates violate at the same rate, which is not observed.
Writes results/line1_audit.json.
"""
import json

import numpy as np
import pandas as pd

import history
import ml_features
import splits
import train_models as tm
from config import DEV_YEARS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE

N_BOOT = 2000


def k_model_at(y_by_model, target):
    """Smallest k whose top-k (in model order) holds at least `target` positives."""
    return int(np.searchsorted(np.cumsum(y_by_model), target) + 1)


def saved_inspections(rank_m, rank_b, y):
    n = len(y); k = int(np.ceil(n / 2))
    target = y[rank_b <= k].sum()
    km = k_model_at(y[np.argsort(rank_m)], target)
    return k, int(target), km


def audit(f, y, score, ins, scoring_date, rng):
    o_m, ev_m = tm.rank_eval(score, f, y)
    o_b, ev_b = tm.rank_eval(f.history_rate.to_numpy(), f, y)
    rank_m, rank_b = np.argsort(o_m) + 1, np.argsort(o_b) + 1
    k, found, km = saved_inspections(rank_m, rank_b, y)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        kb, _, kmb = saved_inspections(np.argsort(np.argsort(rank_m[i], kind='stable')) + 1,
                                       np.argsort(np.argsort(rank_b[i], kind='stable')) + 1, y[i])
        boot.append(kb - kmb)
    top_model = set(f.index[rank_m <= km])
    pool = [c for c in splits.candidates(ins, scoring_date) if c not in top_model]
    base = float(y.mean())
    slots = k - km
    return dict(n_inspected=int(len(y)), serious=int(y.sum()), K=k, rule_found_at_K=found,
                rule_recall_at_K=round(found / y.sum(), 4), model_recall_at_K=round(ev_m['recall_at_50'], 4),
                K_model=km, saved=slots, saved_share_of_K=round(slots / k, 4),
                saved_bootstrap_se=round(float(np.std(boot)), 1),
                saved_ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                audit_pool=len(pool), audit_slots=slots, serious_base_rate=round(base, 4),
                expected_serious_found_by_audit=round(slots * base, 1))


def main():
    rng = np.random.default_rng(SEED)
    chosen = json.loads((RESULTS / 'ml_results.json').read_text())['chosen']
    make = tm.GRID[chosen['family']][1]
    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    out = dict(chosen=chosen, assumption='audited (uninspected) candidates violate at the serious base rate of '
                                         'inspected restaurants; not observed')
    for v in DEV_YEARS:
        tr, dv = tm.train_rows(rows, f'{v}-01-01'), rows[rows.year == v]
        m = make(chosen['params']).fit(tr[tm.X], tr.y)
        out[f'dev{v}'] = audit(dv, dv.y.to_numpy(), m.predict_proba(dv[tm.X])[:, 1], ins, f'{v}-01-01', rng)
    lab = splits.test_set(ins)
    ft = ml_features.build(ins, lab.id.tolist(), TEST_SCORING_DATE)
    yt = (lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    tr = tm.train_rows(rows, TEST_SCORING_DATE)
    m = make(chosen['params']).fit(tr[tm.X], tr.y)
    out['test'] = audit(ft, yt, m.predict_proba(ft[tm.X])[:, 1], ins, TEST_SCORING_DATE, rng)
    (RESULTS / 'line1_audit.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
