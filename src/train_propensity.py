"""The P(inspected) experiment: a propensity model e(x) = P(I = 1 | x) and the Bayes-rule score s = q * e.

Training rows: every candidate at the Jan 1 scoring dates, target inspected within 12 months. The inputs are
the inspection-decision features in selection_features.SELECTION_FEATURES (schedule tier, calendar-year status,
own and area coverage, permit anniversary, size, complaints, placard), not the outcome model's risk features.
A scoring date t can train a model applied at T only when t + 12 months + 30 days <= T, because a negative
label is unknown before then.

Validation: forward in time for V in 2022..2024, model chosen by mean AUC (tiebreak: Brier after an intercept
shift that sets mean e to C(V)/N(V), because t is not in x and the inspection rate moves between years).

Bayes score at the development dates and the test date: q from results/q_scores.parquet, forecast e from the
chosen model refit on usable dates (Aug 11 dates for the Aug 11 test date), s = q * e raw and with the shifted
e. q conditions on the risk features only, so s equals the posterior up to a constant only if the selection
features carry no extra information about the outcome given inspection. As a check, q_union refits the chosen
q configuration on the union of both feature sets, and s_union = q_union * e. Every ranking is evaluated with evaluate.py; the IPW yield uses an in-period cross-fitted propensity and is
computed at the development dates only. The output records which assumption each number rests on.
Writes results/propensity_runs.json and results/propensity_results.json.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import evaluate
import ml_features
import panel
import selection_features
import train_models
from baselines import ranks
from config import DAYS_PER_MONTH, DEV_YEARS, HORIZON_MONTHS, REPORT_LAG_DAYS, RESULTS, SEED, TEST_SCORING_DATE

X = selection_features.SELECTION_FEATURES     # inputs of e
X_Q = ml_features.FEATURES                     # inputs of q
X_UNION = X_Q + [c for c in X if c not in X_Q]
VALIDATION_YEARS = (2022, 2023, 2024)
IPW_FLOORS = (0.02, 0.05)
LABEL_DELAY = pd.Timedelta(days=HORIZON_MONTHS * DAYS_PER_MONTH + REPORT_LAG_DAYS)

GRID = {
    'logistic_regression': (
        dict(C=[0.01, 0.1, 1]),
        lambda p: make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                LogisticRegression(C=p['C'], max_iter=5000))),
    'gaussian_nb': (
        dict(var_smoothing=[1e-9]),   # sklearn default, not tuned
        lambda p: make_pipeline(SimpleImputer(strategy='median', add_indicator=True), GaussianNB(**p))),
    'gradient_boosting': (
        dict(learning_rate=[0.03, 0.1], max_iter=[100, 300]),
        lambda p: HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **p)),
}

ASSUMPTIONS = {
    'lower_bound_yield': 'no assumption; counts uninspected restaurants as misses, so it rewards any score that favors inspected restaurants',
    'conditional_yield': 'no assumption; conditions on ACHD selection, biased for P(Y=1|x)',
    'conditional_recall': 'no assumption; conditions on ACHD selection, biased for P(Y=1|x)',
    'ipw_yield': 'ignorability: P(I=1 | Y=1, x) = P(I=1 | x), with overlap; under it q is already P(Y=1|x) and s is the wrong score',
    's_ranking': 'constant c: P(I=1 | Y=1, x) does not depend on x; untestable here',
}


def all_rows(p: pd.DataFrame) -> pd.DataFrame:
    """Every candidate row with its inspected label and the date the label is known."""
    parts = [panel.at(p, t) for t in sorted(p.scoring_date.unique())]
    rows = pd.concat(parts, ignore_index=True)
    rows['y'] = rows.inspected.astype(int)
    rows['label_known'] = rows.scoring_date + LABEL_DELAY
    return rows


def usable(rows: pd.DataFrame, scoring_date, same_day: bool) -> pd.DataFrame:
    """Rows whose label is known at the scoring date; optionally only rows on the same month and day."""
    t = pd.Timestamp(scoring_date)
    r = rows[rows.label_known <= t]
    if same_day:
        r = r[(r.scoring_date.dt.month == t.month) & (r.scoring_date.dt.day == t.day)]
    return r


def shift_to_mean(e: np.ndarray, target: float) -> np.ndarray:
    """Shift e on the logit scale so its mean equals target."""
    z = logit(np.clip(e, 1e-6, 1 - 1e-6))
    d = brentq(lambda d: expit(z + d).mean() - target, -20, 20)
    return expit(z + d)


def describe(e: np.ndarray, y: np.ndarray) -> dict:
    """Discrimination, calibration and weight diagnostics of a propensity vector."""
    target = float(y.mean())
    es = shift_to_mean(e, target)
    dec = pd.qcut(pd.Series(es).rank(method='first'), 10, labels=False).to_numpy()
    cal = [dict(decile=int(d), mean_e_shifted=float(es[dec == d].mean()), observed=float(y[dec == d].mean()),
                n=int((dec == d).sum())) for d in range(10)]
    w = 1 / es[y == 1]
    return dict(auc=float(roc_auc_score(y, e)), brier_raw=float(brier_score_loss(y, e)),
                brier_shifted=float(brier_score_loss(y, es)), mean_e_raw=float(e.mean()), observed_rate=target,
                e_shifted_quantiles={k: float(np.quantile(es, q)) for k, q in
                                     (('min', 0), ('p1', .01), ('p5', .05), ('p50', .5), ('p95', .95))},
                ess_inspected=float(w.sum() ** 2 / (w ** 2).sum()), n_inspected=int(y.sum()), calibration=cal)


def rank_metrics(cand: pd.DataFrame, score: np.ndarray, c: int, boot: bool = True) -> tuple[np.ndarray, dict]:
    f = cand.set_index('id')[['last_result', 'overdue']].copy(); f['score'] = score
    r = ranks(f, 'score')
    ins, ser = cand.inspected.to_numpy(), cand.serious.to_numpy()
    m = evaluate.to_dict(evaluate.metrics(r, ins, ser, c))
    if boot:
        m['bootstrap'] = {k: evaluate.bootstrap_metric(k, r, ins, ser, c, seed=SEED)
                          for k in ('lower_bound_yield', 'conditional_recall')}
    return r, m


def main() -> None:
    t0 = time.time()
    p = panel.load()
    rows = all_rows(p)
    q_scores = pd.read_parquet(train_models.Q_SCORES)
    make = {fam: mk for fam, (_, mk) in GRID.items()}
    q_cfg = json.loads((RESULTS / 'ml_results.json').read_text())['chosen']
    make_q = {fam: mk for fam, (_, mk) in train_models.GRID.items()}[q_cfg['family']]
    q_rows = train_models.training_rows(p)

    # forward validation of the propensity models
    runs = []
    for family, params, mk in train_models.configs(GRID):
        r = dict(family=family, params=params)
        for v in VALIDATION_YEARS:
            d = f'{v}-01-01'
            tr, va = usable(rows, d, same_day=True), rows[rows.scoring_date == pd.Timestamp(d)]
            m = mk(params).fit(tr[X], tr.y)
            r[f'val{v}'] = describe(m.predict_proba(va[X])[:, 1], va.y.to_numpy())
            r[f'val{v}']['n_train'] = int(len(tr))
        r['mean_auc'] = float(np.mean([r[f'val{v}']['auc'] for v in VALIDATION_YEARS]))
        r['mean_brier_shifted'] = float(np.mean([r[f'val{v}']['brier_shifted'] for v in VALIDATION_YEARS]))
        runs.append(r)
        print(f"{family:20s} {params} AUC {r['mean_auc']:.3f} shifted Brier {r['mean_brier_shifted']:.4f}")
    chosen = max(runs, key=lambda r: (r['mean_auc'], -r['mean_brier_shifted']))
    print('chosen:', chosen['family'], chosen['params'])

    # importances of the chosen model on the 2024 validation date
    tr, va = usable(rows, '2024-01-01', True), rows[rows.scoring_date == pd.Timestamp('2024-01-01')]
    m = make[chosen['family']](chosen['params']).fit(tr[X], tr.y)
    pi = permutation_importance(m, va[X], va.y, scoring='roc_auc', n_repeats=5, random_state=SEED)
    importance = dict(permutation_auc_drop_2024=dict(sorted(zip(X, map(float, pi.importances_mean)), key=lambda kv: -kv[1])))
    lr = make['logistic_regression']({'C': 0.1}).fit(tr[X], tr.y)
    importance['logistic_standardized_coef_2024'] = dict(zip(lr[0].get_feature_names_out(X), map(float, lr[-1].coef_[0])))

    # Bayes score at the development dates and the test date
    dates = [f'{y}-01-01' for y in DEV_YEARS] + [TEST_SCORING_DATE]
    results = {}
    for d in dates:
        cand = panel.at(p, d)
        c = int(cand.inspected.sum())
        q = q_scores[q_scores.scoring_date == pd.Timestamp(d)].set_index('id').q.reindex(cand.id).to_numpy()
        tr = usable(rows, d, same_day=True)
        e = make[chosen['family']](chosen['params']).fit(tr[X], tr.y).predict_proba(cand[X])[:, 1]
        e_shift = shift_to_mean(e, c / len(cand))
        trq = train_models.usable(q_rows, d)
        q_union = make_q(q_cfg['params']).fit(trq[X_UNION], trq.y).predict_proba(cand[X_UNION])[:, 1]
        scores = {'q': q, 'q_union': q_union, 'e_forecast': e, 's_raw': q * e, 's_shifted': q * e_shift,
                  's_union': q_union * e, 'history_rate': cand.history_rate.to_numpy()}
        rec = dict(C=c, n_candidates=len(cand), n_train_e=int(len(tr)),
                   train_dates_e=sorted(str(x.date()) for x in tr.scoring_date.unique()),
                   forecast_e=dict(mean_raw=float(e.mean()), target=c / len(cand), auc_in_period=float(roc_auc_score(cand.inspected, e))),
                   max_q_times_e=dict(raw=float(np.nanmax(q * e)), shifted=float(np.nanmax(q * e_shift))),
                   rankings={}, top_c_overlap={})
        rk = {}
        for name, s in scores.items():
            rk[name], rec['rankings'][name] = rank_metrics(cand, s, c)
        names = list(scores)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rec['top_c_overlap'][f'{a} & {b}'] = int((evaluate.top_mask(rk[a], c) & evaluate.top_mask(rk[b], c)).sum())
        if d != TEST_SCORING_DATE:
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            e_in = cross_val_predict(make[chosen['family']](chosen['params']), cand[X], cand.inspected.astype(int), cv=cv,
                                     method='predict_proba')[:, 1]
            rec['in_period_e'] = describe(e_in, cand.inspected.astype(int).to_numpy())
            rec['in_period_e'].pop('calibration')
            ins, ser = cand.inspected.to_numpy(), cand.serious.to_numpy()
            rec['ipw'] = {}
            for name in ('q', 'q_union', 's_raw', 's_union', 'history_rate'):
                rec['ipw'][name] = {str(fl): evaluate.ipw_yield(rk[name], ins, ser, e_in, c, fl) for fl in IPW_FLOORS}
                r_ = rk[name]
                rec['ipw'][name]['bootstrap_at_' + str(IPW_FLOORS[0])] = evaluate.bootstrap(
                    lambda idx: evaluate.ipw_yield(r_[idx], ins[idx], ser[idx], e_in[idx], c, IPW_FLOORS[0])['ipw_yield'],
                    len(cand), seed=SEED)
        else:
            rec['ipw'] = 'not computed: at the test date inspected means "linked Accela report", which no model of ACHD selection describes'
        results[d] = rec
        print(d, {k: round(v['lower_bound_yield'], 4) for k, v in rec['rankings'].items()}, 'ACHD', round(rec['rankings']['q']['achd_yield'], 4))

    # covariate shift between the last Jan 1 date and the Aug 11 test date
    qs = [.05, .25, .5, .75, .95]
    shift = {f: {d: {str(k): float(v) for k, v in panel.at(p, d)[f].quantile(qs).items()} for d in ('2024-01-01', TEST_SCORING_DATE)}
             for f in ('days_since_last_routine', 'days_to_due', 'routine_this_year', 'zip_coverage_prev_year', 'history_rate', 'last_result')}

    out = dict(features_e=X, features_q=X_Q, q_config=q_cfg, validation_years=VALIDATION_YEARS, selection='mean AUC, tiebreak shifted Brier',
               chosen=dict(family=chosen['family'], params=chosen['params']),
               family_best={fam: max((r for r in runs if r['family'] == fam), key=lambda r: r['mean_auc']) for fam in GRID},
               importance=importance, bayes=results, covariate_shift=shift, assumptions=ASSUMPTIONS,
               notes=['e uses the selection features, q the risk features; s is the posterior up to a constant only if the selection features add nothing about the outcome given inspection (compare s_union)',
                      'q is a Jan 1 model scored on Aug 11 features at the test date; e is refit on Aug 11 dates there',
                      'a constant c must be at least max_q_times_e, because q e / c is a probability',
                      'the lower-bound yield rewards scores that favor inspected restaurants, so s beating q on it is partly mechanical'],
               runtime_s=round(time.time() - t0))
    (RESULTS / 'propensity_runs.json').write_text(json.dumps(runs, indent=1, default=str))
    (RESULTS / 'propensity_results.json').write_text(json.dumps(out, indent=1, default=str))
    print('runtime', out['runtime_s'], 's')


if __name__ == '__main__':
    main()
