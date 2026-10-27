"""Point-in-time panel: one row per (candidate restaurant, scoring date).

Every column is computed from records known at the scoring date t (inspection date + 30 days < t), except the
label columns, which describe the first routine inspection on or after t. Horizon and threshold are applied
by the reader through `inspected()` and `serious()`, so the panel never bakes them in.

Label source by scoring date: the old-system table before ACCELA_START, the linked Accela reports from
ACCELA_START on. `months_covered` says how far past t the source extends; a horizon longer than that is
unknown, not negative.
"""
import sys

import numpy as np
import pandas as pd

from src.data import history
from src.features import risk
from src.features import selection
from src.config import (ACCELA_DATA_END, ACCELA_START, DAYS_PER_MONTH, HORIZON_MONTHS, LOOKBACK_YEARS, OLD_DATA_END, PROCESSED,
                    REPORT_LAG_DAYS, RESTAURANT, SCORING_DATES, SERIOUS)

LAG = pd.Timedelta(days=REPORT_LAG_DAYS)
PANEL = PROCESSED / 'panel.parquet'
LABEL_EXTRA = ['mk_high', 'linked_via']   # Accela-only columns carried when present


def known(ins: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Records whose report is known at t."""
    return ins[ins.date + LAG < t]


def candidates(ins: pd.DataFrame, t: pd.Timestamp, lookback_years: int = LOOKBACK_YEARS,
               require_status: set | None = None) -> pd.DataFrame:
    """Candidate restaurants at t, indexed by id, with the snapshot columns of their most recent known record.

    A candidate's most recent known record is filed under a restaurant category and became known within
    `lookback_years` before t. `require_status` optionally restricts the permit status code.
    """
    last = known(ins, t).sort_values(['date', 'encounter']).drop_duplicates('id', keep='last')
    keep = (last.date + LAG > t - pd.DateOffset(years=lookback_years)) & last.description.isin(RESTAURANT)
    if require_status is not None:
        keep &= last.status.isin(require_status)
    return last.loc[keep, ['id', 'status', 'description', 'municipal', 'zip', 'bus_st_date']].set_index('id')


def labels(events: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, data_end: pd.Timestamp) -> pd.DataFrame:
    """First routine inspection on or after t for each id.

    events: frame with id, date, n_high, routine (and optionally LABEL_EXTRA columns).
    Returns first_routine_date, n_high_first, gap_days, months_covered, plus any LABEL_EXTRA present.
    """
    r = events[events.routine & (events.date >= t) & events.id.isin(set(ids))]
    first = r.sort_values(['date']).drop_duplicates('id').set_index('id')
    out = pd.DataFrame(index=pd.Index(ids, name='id'))
    out['first_routine_date'] = first.date.reindex(out.index)
    out['n_high_first'] = first.n_high.reindex(out.index).astype(float)
    out['gap_days'] = (out.first_routine_date - t).dt.days.astype(float)
    out['months_covered'] = (data_end - t).days / DAYS_PER_MONTH
    for c in LABEL_EXTRA:
        if c in first:
            out[c] = first[c].reindex(out.index)
    return out


def inspected(panel: pd.DataFrame, horizon_months: int) -> pd.Series:
    """I = 1 when the first routine inspection falls within the horizon; NA where the source does not cover it."""
    hit = panel.gap_days <= DAYS_PER_MONTH * horizon_months
    return hit.where(panel.months_covered >= horizon_months).astype('boolean')


def serious(panel: pd.DataFrame, horizon_months: int, threshold: int = SERIOUS, count: str = 'n_high_first') -> pd.Series:
    """S = 1 when inspected within the horizon and the first routine inspection has >= threshold high-risk violations."""
    return (inspected(panel, horizon_months) & (panel[count] >= threshold)).astype('boolean')


def accela_events() -> pd.DataFrame:
    """Linked Accela reports as label events: id = old-system id, routine = Comprehensive."""
    rep = pd.read_csv(PROCESSED / 'accela_reports.csv', dtype={'old_id': str, 'client_id': str})
    rep = rep[rep.old_id.notna()].rename(columns={'old_id': 'id'})
    rep['date'] = pd.to_datetime(rep.inspection_date)
    rep['routine'] = rep.purpose.eq('Comprehensive')
    return rep[['id', 'date', 'routine', 'n_high', 'mk_high', 'linked_via']]


def label_source(ins: pd.DataFrame, t: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Events and data end for the label source that covers scoring date t."""
    if t >= pd.Timestamp(ACCELA_START):
        return accela_events(), pd.Timestamp(ACCELA_DATA_END)
    return ins, pd.Timestamp(OLD_DATA_END)


def rows_at(ins: pd.DataFrame, t: pd.Timestamp, roster: pd.DataFrame) -> pd.DataFrame:
    """One panel row per candidate at t: snapshot columns, outcome features, selection features, labels."""
    cand = candidates(ins, t)
    feats = risk.build(ins, cand.index.tolist(), t)
    sel = selection.build(ins, roster, cand, t)
    events, data_end = label_source(ins, t)
    labs = labels(events, cand.index, t, data_end)
    out = pd.concat([cand[['status', 'description', 'zip']], feats, sel, labs], axis=1)
    out.insert(0, 'scoring_date', t)
    return out.reset_index()


def build(ins: pd.DataFrame, scoring_dates=SCORING_DATES) -> pd.DataFrame:
    """Panel over all scoring dates."""
    parts = []
    roster = selection.load_roster()
    for d in scoring_dates:
        t = pd.Timestamp(d)
        p = rows_at(ins, t, roster)
        print(f'{d}: {len(p)} candidates, {int(inspected(p, 12).fillna(False).sum())} inspected within 12 months',
              file=sys.stderr)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def load() -> pd.DataFrame:
    """Read the saved panel."""
    return pd.read_parquet(PANEL)


def main() -> None:
    ins = history.load_inspections()
    panel = build(ins)
    panel.to_parquet(PANEL, index=False)
    print('wrote', PANEL, panel.shape)


if __name__ == '__main__':
    main()


def at(panel: pd.DataFrame, t, horizon_months: int = HORIZON_MONTHS, count: str = 'n_high_first') -> pd.DataFrame:
    """Rows of one scoring date with boolean `inspected` and `serious` columns at the given horizon.

    Raises if the label source does not cover the horizon, because an uncovered row is unknown, not negative.
    """
    rows = panel[panel.scoring_date == pd.Timestamp(t)].reset_index(drop=True)
    ins = inspected(rows, horizon_months)
    assert not ins.isna().any(), f'horizon {horizon_months} months not covered at {t}'
    rows['inspected'] = ins.astype(bool)
    rows['serious'] = serious(rows, horizon_months, count=count).astype(bool)
    return rows
