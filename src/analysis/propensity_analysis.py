"""Analysis tables for the inspection-decision models: forward vs in-period AUC by inputs and model, training-window
sensitivity, univariate AUCs, inspection rate by schedule bucket, calibration, test-date AUC. Prints to stdout;
the saved copy is results/diagnostics/propensity_analysis.txt."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from src import panel
from src.features import risk, selection
from src.models import train_propensity as tp
from src.config import SEED
p = panel.load(); rows = tp.all_rows(p)
X_RISK, X_SEL = risk.FEATURES, selection.SELECTION_FEATURES
X_UNION = X_RISK + [c for c in X_SEL if c not in X_RISK]
sets = {'risk (old inputs)': X_RISK, 'selection (new inputs)': X_SEL, 'union': X_UNION}
mk = {fam: m for fam, (_, m) in tp.GRID.items()}
cfg = {'logistic_regression': {'C': 0.01}, 'gaussian_nb': {'var_smoothing': 1e-9}, 'gradient_boosting': {'learning_rate': 0.03, 'max_iter': 300}}

print('== 1. Forward AUC (train on usable earlier Jan-1 dates, score year V) and in-period 5-fold CV AUC, by feature set and model')
out = []
for V in (2022, 2023, 2024):
    d = f'{V}-01-01'; tr = tp.usable(rows, d, True); va = rows[rows.scoring_date == pd.Timestamp(d)]
    for sname, X in sets.items():
        for fam, params in cfg.items():
            m = mk[fam](params).fit(tr[X], tr.y); e = m.predict_proba(va[X])[:, 1]
            cv = cross_val_predict(mk[fam](params), va[X], va.y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method='predict_proba')[:, 1]
            out.append(dict(year=V, inputs=sname, model=fam, forward_auc=roc_auc_score(va.y, e), inperiod_auc=roc_auc_score(va.y, cv),
                            mean_pred=e.mean(), observed=va.y.mean(), brier_fwd=brier_score_loss(va.y, e), brier_in=brier_score_loss(va.y, cv)))
df = pd.DataFrame(out)
pd.set_option('display.width', 250)
print(df.pivot_table(index=['inputs', 'model'], columns='year', values=['forward_auc', 'inperiod_auc']).round(3).to_string())
print(df[df.model == 'gradient_boosting'].pivot_table(index='inputs', columns='year', values=['mean_pred', 'observed']).round(3).to_string())

print('\n== 2. Does more history help? GBM on selection inputs, forward AUC at V by training window')
for V in (2022, 2023, 2024):
    d = f'{V}-01-01'; va = rows[rows.scoring_date == pd.Timestamp(d)]; tr_all = tp.usable(rows, d, True)
    res = {}
    for label, yrs in [('all usable', None), ('last 1 usable year', 1), ('last 3 usable years', 3), ('pre-2020 only', 'pre')]:
        if yrs is None: tr = tr_all
        elif yrs == 'pre': tr = tr_all[tr_all.scoring_date.dt.year < 2020]
        else:
            last = sorted(tr_all.scoring_date.unique())[-yrs:]; tr = tr_all[tr_all.scoring_date.isin(last)]
        m = mk['gradient_boosting'](cfg['gradient_boosting']).fit(tr[X_SEL], tr.y)
        res[label] = (round(roc_auc_score(va.y, m.predict_proba(va[X_SEL])[:, 1]), 3), len(tr))
    print(V, res)

print('\n== 3. Univariate AUC of each selection feature, in period (sign-adjusted), by year')
uni = {}
for V in (2019, 2022, 2024):
    va = rows[rows.scoring_date == pd.Timestamp(f'{V}-01-01')]
    uni[V] = {}
    for f in X_SEL:
        x = va[f].fillna(va[f].median()) if va[f].notna().any() else None
        if x is None or x.nunique() < 2: continue
        a = roc_auc_score(va.y, x); uni[V][f] = round(max(a, 1 - a), 3)
print(pd.DataFrame(uni).sort_values(2024, ascending=False).to_string())

print('\n== 4. Inspection rate by days_to_due bucket and by H tier, per year')
for V in (2019, 2022, 2024):
    va = rows[rows.scoring_date == pd.Timestamp(f'{V}-01-01')]
    b = pd.cut(va.days_to_due, [-np.inf, -365, -180, -90, 0, 90, 180, np.inf]).astype(str)
    print(V, 'overall', round(va.y.mean(), 3))
    print('   by days_to_due', va.groupby(b).y.agg(['mean', 'size']).round(3).T.to_dict())
    print('   by h_tier', va.groupby(va.h_tier.fillna(-1)).y.agg(['mean', 'size']).round(3).T.to_dict())
    print('   by share_years_covered_3y', va.groupby(va.share_years_covered_3y.round(2)).y.agg(['mean', 'size']).round(3).T.to_dict())

print('\n== 5. Calibration of chosen GBM forward model (selection inputs) at 2024, raw deciles')
d = '2024-01-01'; tr = tp.usable(rows, d, True); va = rows[rows.scoring_date == pd.Timestamp(d)]
e = mk['gradient_boosting'](cfg['gradient_boosting']).fit(tr[X_SEL], tr.y).predict_proba(va[X_SEL])[:, 1]
dec = pd.qcut(pd.Series(e).rank(method='first'), 10, labels=False).to_numpy()
print(pd.DataFrame({'mean_pred': [e[dec == k].mean() for k in range(10)], 'observed': [va.y.to_numpy()[dec == k].mean() for k in range(10)]}).round(3).T.to_string())

print('\n== 6. Aug 11 test-date model (trained on Aug 11 2019-2023 rows): forward AUC on 2025-08-11 (inspected = linked Accela report), by inputs and model')
d = '2025-08-11'; tr = tp.usable(rows, d, True); va = rows[rows.scoring_date == pd.Timestamp(d)]
for sname, X in sets.items():
    print(sname, {fam: round(roc_auc_score(va.y, mk[fam](params).fit(tr[X], tr.y).predict_proba(va[X])[:, 1]), 3) for fam, params in cfg.items()})
