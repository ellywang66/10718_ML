"""Candidate lists and labels for the development years and the held-out test year.

Candidates on a scoring date: restaurants whose most recent known record (known 30 days after the inspection,
on or before the scoring date) became known within the three years before that date, and whose most recent
record is filed under a restaurant category.

Development (scoring date Jan 1, 2023 and Jan 1, 2024): label = the first routine inspection (purpose "Initial")
of that calendar year, from the old-system violations table.
Test (scoring date Jan 1, 2025): label = the first routine ("Comprehensive") inspection between Aug 11, 2025
and Jul 31, 2026, from the new-system inspection reports (data/processed/accela_reports.csv).
Restaurants without such an inspection have no label and are not scored; they are never counted as negatives.
"""
import pandas as pd
from config import RESTAURANT, REPORT_LAG_DAYS, CANDIDATE_YEARS, PROCESSED, TEST_WINDOW

LAG = pd.Timedelta(days=REPORT_LAG_DAYS)


def candidates(ins, scoring_date):
    t = pd.Timestamp(scoring_date)
    known = ins[ins.date + LAG <= t]
    last = known.sort_values(['date', 'encounter']).drop_duplicates('id', keep='last')
    keep = (last.date + LAG > t - pd.DateOffset(years=CANDIDATE_YEARS)) & last.description.isin(RESTAURANT)
    return last.loc[keep, 'id'].tolist()


def dev_set(ins, year):
    cand = set(candidates(ins, f'{year}-01-01'))
    first = (ins[ins.purpose.eq('Initial') & (ins.date.dt.year == year) & ins.id.isin(cand)]
             .sort_values(['id', 'date']).drop_duplicates('id'))
    return first[['id', 'date', 'n_high']].reset_index(drop=True)


def test_set(ins):
    cand = set(candidates(ins, '2025-01-01'))
    rep = pd.read_csv(PROCESSED / 'accela_reports.csv', dtype={'old_id': str})
    rep['date'] = pd.to_datetime(rep.inspection_date)
    rep = rep[rep.purpose.eq('Comprehensive') & rep.date.between(*TEST_WINDOW) & rep.n_high.notna() & rep.old_id.notna()]
    first = rep.sort_values('date').drop_duplicates('old_id').rename(columns={'old_id': 'id'})
    first = first[first.id.isin(cand)]
    return first[['id', 'date', 'n_high']].reset_index(drop=True)
