"""Subgroup check and appendix lines 1 and 2 for the model chosen on L90 (results/recall_targets.json).

L90 = share of the ranked list needed to reach 90% of the serious violators. K90 = the chosen model's number of
top-ranked test restaurants that reach it (1,049 of 1,616).
- Subgroups: recall within each subgroup, counting the same global top K90 for the model and the history-rate rule.
- Line 1: inspections saved at 90% recall = rule's K90 - model's K90, per evaluation set, with a bootstrap by
  restaurant; the saved inspections fund a random audit of candidates outside the model's top K90. Expected
  serious found assumes audited restaurants violate at the inspected rate (not observed).
- Line 2: q = the chosen model's score for every Jan 1, 2025 candidate; cutoff = the score of the K90-th inspected
  restaurant. e and predicted-inspected come from results/line2_scores.csv (line2_propensity.py).
Writes results/l90_followups.json.
"""
import json

import numpy as np
import pandas as pd

import history
import ml_features
import splits
import train_models as tm
from config import DEV_YEARS, RESULTS, SEED, SERIOUS, TEST_SCORING_DATE
from recall_targets import rank

N_BOOT = 2000
TARGET = 0.9


def k_at(o, y, target=TARGET):
    return int(np.searchsorted(np.cumsum(y[o]), np.ceil(target * y.sum())) + 1)


def line1(f, y, score, ins, scoring_date, rng):
    o_m, o_b = rank(score, f), rank(f.history_rate.to_numpy(), f)
    km, kb = k_at(o_m, y), k_at(o_b, y)
    pos_m, pos_b = np.argsort(o_m), np.argsort(o_b)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boot.append(k_at(np.argsort(pos_b[i], kind='stable'), y[i]) - k_at(np.argsort(pos_m[i], kind='stable'), y[i]))
    top_model = set(f.index[o_m[:km]])
    pool = [c for c in splits.candidates(ins, scoring_date) if c not in top_model]
    base = float(y.mean())
    return dict(n_inspected=int(len(y)), serious=int(y.sum()), target_found=int(np.ceil(TARGET * y.sum())),
                K90_rule=kb, K90_model=km, saved=kb - km, saved_bootstrap_se=round(float(np.std(boot)), 1),
                saved_ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])], audit_pool=len(pool),
                serious_base_rate=round(base, 4), expected_serious_found_by_audit=round((kb - km) * base, 1))


def main():
    rng = np.random.default_rng(SEED)
    chosen = json.loads((RESULTS / 'recall_targets.json').read_text())['chosen']
    make = tm.GRID[chosen['family']][1]
    ins = history.load_inspections()
    rows = pd.concat([tm.labeled(ins, y) for y in tm.TRAIN_YEARS])
    out = dict(chosen=chosen, target_recall=TARGET)

    l1 = {}
    for v in DEV_YEARS:
        tr, dv = tm.train_rows(rows, f'{v}-01-01'), rows[rows.year == v]
        m = make(chosen['params']).fit(tr[tm.X], tr.y)
        l1[f'dev{v}'] = line1(dv, dv.y.to_numpy(), m.predict_proba(dv[tm.X])[:, 1], ins, f'{v}-01-01', rng)
    lab = splits.test_set(ins)
    ft = ml_features.build(ins, lab.id.tolist(), TEST_SCORING_DATE)
    yt = (lab.set_index('id').n_high.reindex(ft.index) >= SERIOUS).astype(int).to_numpy()
    m = make(chosen['params']).fit(*(lambda t: (t[tm.X], t.y))(tm.train_rows(rows, TEST_SCORING_DATE)))
    score = m.predict_proba(ft[tm.X])[:, 1]
    l1['test'] = line1(ft, yt, score, ins, TEST_SCORING_DATE, rng)
    out['line1'] = l1

    # subgroups at the model's K90 on test
    o_m, o_b = rank(score, ft), rank(ft.history_rate.to_numpy(), ft)
    k = l1['test']['K90_model']
    top_m = np.zeros(len(yt), bool); top_m[o_m[:k]] = True
    top_b = np.zeros(len(yt), bool); top_b[o_b[:k]] = True
    month = lab.set_index('id').date.reindex(ft.index)
    nr, muni = ft.n_routine, ft.municipal.str.contains('Pittsburgh', case=False)
    groups = {'Aug-Dec 2025': month.dt.year.eq(2025).to_numpy(), 'Jan-Jul 2026': month.dt.year.eq(2026).to_numpy(),
              'Pittsburgh': muni.to_numpy(), 'Elsewhere': (~muni).to_numpy(),
              'Chain': ft.is_chain.eq(1).to_numpy(), 'Independent': ft.is_chain.eq(0).to_numpy(),
              'No history': nr.eq(0).to_numpy(), '1-3 past': nr.between(1, 3).to_numpy(), '4+ past': (nr >= 4).to_numpy()}
    out['subgroups'] = dict(K=k, all=dict(model_found=int(yt[top_m].sum()), rule_found=int(yt[top_b].sum())))
    for name, mask in groups.items():
        pos = int(yt[mask].sum())
        out['subgroups'][name] = dict(n=int(mask.sum()), serious=pos, model_found=int(yt[mask & top_m].sum()),
                                      rule_found=int(yt[mask & top_b].sum()), model_in_top=int((mask & top_m).sum()),
                                      rule_in_top=int((mask & top_b).sum()))

    # line 2: q from the chosen model on every candidate, e from line2_propensity.py
    sc = pd.read_csv(RESULTS / 'line2_scores.csv', dtype={'id': str}).set_index('id')
    fc = ml_features.build(ins, sc.index.tolist(), TEST_SCORING_DATE)
    sc['q'] = m.predict_proba(fc[tm.X])[:, 1]
    cutoff = float(np.sort(sc.q[sc.inspected.eq(1)].to_numpy())[::-1][k - 1])
    pni = sc[sc.pred_inspected.eq(0)]
    insp = sc[sc.inspected.eq(1)]
    nohist = sc[sc.n_routine.eq(0)]
    out['line2'] = dict(q_cutoff=round(cutoff, 5), serious_above_cutoff=int(((insp.q >= cutoff) & insp.y.eq(1)).sum()),
                        serious=int(insp.y.sum()), predicted_not_inspected=int(len(pni)),
                        uninspected_above_cutoff=int(((sc.inspected.eq(0)) & (sc.q >= cutoff)).sum()),
                        predicted_not_inspected_above_cutoff=int((pni.q >= cutoff).sum()),
                        no_history=int(len(nohist)), no_history_above_cutoff=int((nohist.q >= cutoff).sum()),
                        no_history_q_median=round(float(nohist.q.median()), 3))
    (RESULTS / 'l90_followups.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
