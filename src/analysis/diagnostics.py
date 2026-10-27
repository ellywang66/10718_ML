"""Must-have diagnostics on the panel. Each writes one JSON file under results/diagnostics/.

D1 coverage against horizon; D2 inspection rate against the rule features; D3 base rates and the test-date
label counts; D4 unlabeled share of the top C for three rankings; D5 bootstrap noise of two metrics;
D6 candidates per permit status code. Old-system scoring dates unless noted.
"""
import json

import numpy as np
import pandas as pd

from src import evaluate
from src import panel
from src.models.baselines import ranks
from src.config import DAYS_PER_MONTH, DIAGNOSTICS, PROCESSED, SEED, SERIOUS, TEST_SCORING_DATE

HORIZONS = (6, 12, 18, 24)
NR_BUCKETS = {'0': lambda n: n.eq(0), '1-3': lambda n: n.between(1, 3), '4+': lambda n: n >= 4}


def jan1(p: pd.DataFrame) -> pd.DataFrame:
    """Old-system rows at the Jan 1 scoring dates."""
    return p[p.scoring_date.dt.month.eq(1) & p.scoring_date.dt.day.eq(1)]


def date_key(t) -> str:
    return str(pd.Timestamp(t).date())


def share(ins: pd.Series) -> float | None:
    """Mean of a nullable boolean series; None when the horizon is not covered."""
    return None if ins.isna().all() else float(ins.dropna().astype(float).mean())


def d1_coverage(p: pd.DataFrame) -> dict:
    out = {}
    for t, rows in p.groupby('scoring_date'):
        rec = {'n': len(rows), 'months_covered': float(rows.months_covered.iloc[0])}
        for h in HORIZONS:
            ins = panel.inspected(rows, h)
            rec[f'H{h}'] = dict(all=share(ins),
                                by_n_routine={k: share(ins[f(rows.n_routine)]) for k, f in NR_BUCKETS.items()},
                                by_category={k: share(ins[rows.description.eq(k)]) for k in sorted(rows.description.unique())})
        out[date_key(t)] = rec
    return out


def d2_selection(p: pd.DataFrame) -> dict:
    out = {}
    for t, rows in jan1(p).groupby('scoring_date'):
        rows = panel.at(p, t)
        ins = rows.inspected.astype(float)
        dec = pd.qcut(rows.history_rate.rank(method='first'), 10, labels=False)
        lr = rows.last_result.fillna(-1).clip(upper=3).astype(int).map({-1: 'missing', 0: '0', 1: '1', 2: '2', 3: '3+'})
        ov = pd.cut(rows.overdue, [-np.inf, 0.5, 0.8, 1.0, 1.2, 1.5, 2, np.inf]).astype(str)
        out[date_key(t)] = dict(
            overall=float(ins.mean()),
            by_history_rate_decile=ins.groupby(dec).mean().round(4).to_dict(),
            history_rate_decile_bounds=rows.history_rate.groupby(dec).agg(['min', 'max']).round(4).to_dict('index'),
            by_last_result=ins.groupby(lr).agg(['mean', 'size']).round(4).to_dict('index'),
            by_overdue=ins.groupby(ov).agg(['mean', 'size']).round(4).to_dict('index'),
            by_n_routine={k: float(ins[f(rows.n_routine)].mean()) for k, f in NR_BUCKETS.items()})
    return out


def d3_base_rates(p: pd.DataFrame) -> dict:
    out = {'by_scoring_date': {}}
    for t in sorted(p.scoring_date.unique()):
        rows = panel.at(p, t)
        hi = rows.loc[rows.inspected, 'n_high_first']
        out['by_scoring_date'][date_key(t)] = dict(
            inspected=int(len(hi)), share_1plus=float((hi >= 1).mean()), share_2plus=float((hi >= 2).mean()),
            share_3plus=float((hi >= 3).mean()), mean_high=float(hi.mean()))
    test = panel.at(p, TEST_SCORING_DATE)
    ins = test[test.inspected]
    rep = pd.read_csv(PROCESSED / 'accela_reports.csv', dtype={'old_id': str})
    rep['date'] = pd.to_datetime(rep.inspection_date)
    comp = rep[rep.purpose.eq('Comprehensive') & rep.old_id.notna()]
    out['test_date'] = dict(
        inspected_linked=int(len(ins)),
        share_2plus_n_high=float((ins.n_high_first >= SERIOUS).mean()),
        share_2plus_mk_high=float((ins.mk_high >= SERIOUS).mean()),
        agree_at_threshold=float(((ins.n_high_first >= SERIOUS) == (ins.mk_high >= SERIOUS)).mean()),
        n_disagree_at_threshold=int(((ins.n_high_first >= SERIOUS) != (ins.mk_high >= SERIOUS)).sum()),
        linked_comprehensive_reports=int(len(comp)),
        linked_comprehensive_before_test_date=int((comp.date < pd.Timestamp(TEST_SCORING_DATE)).sum()),
        by_linked_via=ins.linked_via.value_counts().to_dict(),
        accela_reports_total=int(len(rep)), accela_unlinked=int(rep.old_id.isna().sum()),
        accela_last_report_date=str(rep.date.max().date()))
    return out


def three_rankings(rows: pd.DataFrame, rng: np.random.Generator) -> dict:
    f = rows.set_index('id')
    rnd = f[['last_result', 'overdue']].copy(); rnd['score'] = rng.random(len(f))
    return {'history_rate': ranks(f, 'history_rate'), 'last_result': ranks(f, 'last_result'), 'random': ranks(rnd, 'score')}


def d4_bound_width(p: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for t, rows in jan1(p).groupby('scoring_date'):
        rows = rows.reset_index(drop=True)
        rec = {}
        for h in (12, 24):
            ins = panel.inspected(rows, h)
            if ins.isna().any():
                rec[f'H{h}'] = None; continue
            ins = ins.astype(bool).to_numpy()
            c = int(ins.sum())
            rec[f'H{h}'] = {'C': c, **{name: float(1 - (ins & evaluate.top_mask(r, c)).sum() / c)
                                       for name, r in three_rankings(rows, rng).items()}}
        out[date_key(t)] = rec
    return out


def d5_noise(p: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for t in ('2024-01-01', TEST_SCORING_DATE):
        rows = panel.at(p, t)
        ins, ser = rows.inspected.to_numpy(), rows.serious.to_numpy()
        c = int(ins.sum())
        rec = {'C': c}
        for name, r in three_rankings(rows, rng).items():
            m = evaluate.metrics(r, ins, ser, c)
            rec[name] = {k: dict(point=getattr(m, k), **evaluate.bootstrap_metric(k, r, ins, ser, c, seed=SEED))
                         for k in ('lower_bound_yield', 'conditional_recall')}
        out[t] = rec
    return out


def d6_status(p: pd.DataFrame) -> dict:
    tab = p.groupby([p.scoring_date.dt.date.astype(str), 'status']).size().unstack(fill_value=0)
    return {'note': 'status is constant within id in the WPRDC dump (a snapshot at dump time); code meanings undocumented',
            'candidates_by_status': {d: row.to_dict() for d, row in tab.iterrows()}}


def main() -> None:
    p = panel.load()
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    for name, fn in [('d1_coverage', d1_coverage), ('d2_selection', d2_selection), ('d3_base_rates', d3_base_rates),
                     ('d4_bound_width', d4_bound_width), ('d5_noise', d5_noise), ('d6_status', d6_status)]:
        out = fn(p)
        (DIAGNOSTICS / f'{name}.json').write_text(json.dumps(out, indent=1, default=str))
        print('wrote', name)


if __name__ == '__main__':
    main()
