"""Point-in-time features for the inspection decision: who gets a routine inspection in the next year.

These describe ACHD's schedule rather than restaurant risk. Sources: the old-system inspection table (dates,
purpose types, reinspection flag `reispt_cd`, critical-category counts, placards, zip, business start date) and
the WPRDC facility roster (seat count, floor area). Everything except the roster attributes is computed from
records known at the scoring date t (inspection date + 30 days < t).

Feature groups, from the data survey of 2026-10-26:
1. H tier and due date. ACHD's Priority Code H: a reinspection was required for 2 or more of the nine critical
   categories at the last routine or new-facility inspection; such facilities are revisited in about six
   months, others once per calendar year.
2. Calendar-year status, since Code 1 is "at least once per calendar year".
3. The facility's own coverage history (persistence of being inspected).
4. Coverage of the facility's zip and municipality in the prior year, and the zip's share of recent visits.
5. Months to the permit anniversary (permits expire at the end of the business start month).
6. Facility size and type.
7. Complaint and service-request visits, own and nearby.
8. Last placard status.
"""
import numpy as np
import pandas as pd

from config import RAW, REPORT_LAG_DAYS

LAG = pd.Timedelta(days=REPORT_LAG_DAYS)
H_TIER_DUE_DAYS = 180       # ACHD revisits high-frequency facilities in five to seven months
ROUTINE_DUE_DAYS = 365
PLACEHOLDER_START_DATES = {'1984-06-17', '1970-01-01', '1957-01-01', '1960-01-01', '1985-01-01'}
NOT_PERMITTED_PLACARDS = {'2', '3', '5', '6', '7'}   # consumer alert and the closure placards

NEW = ['h_tier', 'n_critical_last', 'reinsp_required_last', 'days_to_due',
       'routine_this_year', 'cal_years_since_routine',
       'share_years_covered_3y', 'routine_per_year',
       'zip_coverage_prev_year', 'muni_coverage_prev_year', 'zip_share_visits_90d',
       'months_to_anniversary',
       'seat_count', 'sq_feet',
       'n_service_1y', 'zip_complaints_90d',
       'last_placard_not_permitted']
SHARED = ['days_since_last_routine', 'days_since_any', 'n_routine', 'has_liquor', 'is_chain', 'n_complaint_1y']
SELECTION_FEATURES = NEW + SHARED


def load_roster() -> pd.DataFrame:
    """Facility roster indexed by id with numeric seat_count and sq_feet and the business start date."""
    r = pd.read_csv(RAW / 'geocoded_facilities.csv', dtype=str).drop_duplicates('id').set_index('id')
    out = pd.DataFrame(index=r.index)
    out['seat_count'] = pd.to_numeric(r.seat_count, errors='coerce')
    out['sq_feet'] = pd.to_numeric(r.sq_feet, errors='coerce')
    out['fdo'] = r.fdo
    return out


def anniversary_months(start: pd.Series, t: pd.Timestamp) -> pd.Series:
    """Months from t until the next permit anniversary month (0 = this month); NaN for placeholder dates."""
    s = start.where(~start.isin(PLACEHOLDER_START_DATES))
    d = pd.to_datetime(s, errors='coerce')
    return ((d.dt.month - t.month) % 12).astype(float)


def build(ins: pd.DataFrame, roster: pd.DataFrame, cand: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Selection features for the candidates (frame indexed by id with zip, municipal, bus_st_date) at t."""
    known = ins[ins.date + LAG < t].sort_values(['date', 'encounter'])
    ids = cand.index
    f = pd.DataFrame(index=ids)
    own = known[known.id.isin(set(ids))]

    # 1. H tier and due date, from the last routine or new-facility visit
    base = own[own.type.isin(['routine', 'new_facility'])].drop_duplicates('id', keep='last').set_index('id')
    last_routine = own[own.routine].drop_duplicates('id', keep='last').set_index('id')
    f['n_critical_last'] = base.n_critical.reindex(ids).astype(float)
    f['reinsp_required_last'] = base.reispt_cd.isin(['1', '3']).reindex(ids).astype(float)
    f['h_tier'] = ((f.reinsp_required_last == 1) & (f.n_critical_last >= 2)).astype(float)
    f.loc[f.n_critical_last.isna(), ['h_tier', 'reinsp_required_last']] = np.nan
    lr_date = last_routine.date.reindex(ids)
    due = lr_date + pd.to_timedelta(np.where(f.h_tier == 1, H_TIER_DUE_DAYS, ROUTINE_DUE_DAYS), unit='D')
    f['days_to_due'] = (due - t).dt.days.astype(float)

    # 2. calendar-year status
    routine_years = own[own.routine].groupby('id').date.agg(lambda d: set(d.dt.year))
    years = routine_years.reindex(ids)
    f['routine_this_year'] = years.map(lambda s: float(t.year in s) if isinstance(s, set) else np.nan)
    f['cal_years_since_routine'] = (t.year - lr_date.dt.year).astype(float)

    # 3. own coverage history
    window = {t.year - 3, t.year - 2, t.year - 1}
    f['share_years_covered_3y'] = years.map(lambda s: len(s & window) / 3 if isinstance(s, set) else 0.0)
    first_any = own.groupby('id').date.min().reindex(ids)
    span_years = ((t - first_any).dt.days / 365.25).clip(lower=1.0)
    f['routine_per_year'] = (own[own.routine].groupby('id').size().reindex(ids).fillna(0) / span_years).astype(float)

    # 4. area coverage in the prior calendar year (among candidates) and recent visit share by zip
    prev = own[own.routine & (own.date.dt.year == t.year - 1)].id.unique()
    area = cand[['zip', 'municipal']].copy()
    area['covered'] = area.index.isin(set(prev)).astype(float)
    f['zip_coverage_prev_year'] = area.groupby('zip').covered.transform('mean').reindex(ids)
    f['muni_coverage_prev_year'] = area.groupby('municipal').covered.transform('mean').reindex(ids)
    recent = known[(known.date >= t - pd.Timedelta(days=90 + REPORT_LAG_DAYS))]
    zip_share = recent[recent.routine].zip.value_counts(normalize=True)
    f['zip_share_visits_90d'] = cand.zip.map(zip_share).fillna(0.0).astype(float)

    # 5. permit anniversary
    start = cand.bus_st_date.fillna(roster.fdo.reindex(ids))
    f['months_to_anniversary'] = anniversary_months(start, t)

    # 6. size
    f['seat_count'] = roster.seat_count.reindex(ids)
    f['sq_feet'] = roster.sq_feet.reindex(ids)

    # 7. service requests and nearby complaints
    last_year = own[own.date >= t - pd.Timedelta(days=365)]
    f['n_service_1y'] = last_year[last_year.purpose.fillna('').str.contains('Service Request')].groupby('id').size().reindex(ids).fillna(0).astype(float)
    zip_compl = recent[recent.type.eq('complaint')].zip.value_counts()
    f['zip_complaints_90d'] = cand.zip.map(zip_compl).fillna(0).astype(float)

    # 8. last placard
    last_any = own.drop_duplicates('id', keep='last').set_index('id')
    f['last_placard_not_permitted'] = last_any.placard_st.isin(NOT_PERMITTED_PLACARDS).reindex(ids).astype(float)
    f.loc[last_any.placard_st.reindex(ids).isna(), 'last_placard_not_permitted'] = np.nan
    return f[NEW]
