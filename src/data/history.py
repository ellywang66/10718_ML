"""One row per old-system inspection (Jan 2014 to Jul 2025) with its violation counts, critical-category count,
reinspection flag, and inspection type.

Inspection types come from src/data/purpose_types.csv, which maps every observed `purpose` value to one of
routine, complaint, reinspection, new_facility, other. `routine` is the only type used for labels.
Columns that never vary within a restaurant in this dump (status, description, municipal, bus_st_date,
category_cd) are facility snapshots at dump time, not point-in-time values.
"""
from pathlib import Path

import pandas as pd
from src.config import RAW

PURPOSE_TYPES = Path(__file__).with_name('purpose_types.csv')
TYPES = ('routine', 'complaint', 'reinspection', 'new_facility', 'other')

# ACHD's nine critical violation categories (Priority Code H definition), as they appear in `description_new`.
# Cold and hot holding are one category in ACHD's list.
CRITICAL = {'Cooking Temperatures': 'cooking', 'Cooling Food': 'cooling', 'Cold Holding Temperatures': 'holding',
            'Hot Holding Temperatures': 'holding', 'Reheating Temperatures': 'reheating', 'Employee Health': 'employee_health',
            'Employee Personal Hygiene': 'hygiene', 'Cleaning and Sanitization': 'sanitization',
            'Cross-Contamination Prevention': 'cross_contamination', 'Pest Management': 'pests'}


def purpose_types() -> pd.Series:
    """Mapping from `purpose` value to inspection type."""
    m = pd.read_csv(PURPOSE_TYPES).set_index('purpose')['type']
    assert m.isin(TYPES).all(), m[~m.isin(TYPES)]
    return m


def load_inspections() -> pd.DataFrame:
    """All old-system inspections with n_high/n_med/n_low, `type`, and `routine`, sorted by id and date."""
    ins = pd.read_csv(RAW / 'inspections_2014_2025.csv', dtype=str).drop_duplicates('encounter')
    vio = pd.read_csv(RAW / 'violations_2014_2025.csv', dtype=str)
    counts = (vio.assign(h=vio.high.eq('T'), m=vio.medium.eq('T'), l=vio.low.eq('T'))
                 .groupby('encounter').agg(n_high=('h', 'sum'), n_med=('m', 'sum'), n_low=('l', 'sum')))
    critical = vio.assign(cat=vio.description_new.map(CRITICAL)).dropna(subset=['cat']).groupby('encounter').cat.nunique()
    ins = ins.join(counts, on='encounter').join(critical.rename('n_critical'), on='encounter')
    # An inspection with no row in the violations table had no violations (checked against inspection reports).
    for c in ('n_high', 'n_med', 'n_low', 'n_critical'):
        ins[c] = ins[c].fillna(0).astype(int)
    ins['date'] = pd.to_datetime(ins.inspect_dt)
    types = purpose_types()
    unknown = set(ins.purpose.dropna()) - set(types.index)
    assert not unknown, f'purpose values missing from purpose_types.csv: {unknown}'
    ins['type'] = ins.purpose.map(types).fillna('other')
    ins['routine'] = ins.type.eq('routine')
    ins['reinspection_date'] = pd.to_datetime(ins.reispt_dt, errors='coerce')
    cols = ['encounter', 'id', 'description', 'municipal', 'zip', 'status', 'placard_st', 'bus_st_date',
            'category_cd', 'date', 'purpose', 'type', 'routine', 'reispt_cd', 'reinspection_date',
            'n_high', 'n_med', 'n_low', 'n_critical']
    return ins[cols].sort_values(['id', 'date', 'encounter']).reset_index(drop=True)
