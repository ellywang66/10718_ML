"""One row per old-system inspection (Jan 2014 to Jul 2025) with its high/medium/low violation counts."""
import pandas as pd
from config import RAW

ROUTINE_EXCLUDE = 'Complaint|New Facility|Reinspection'


def load_inspections():
    ins = pd.read_csv(RAW / 'inspections_2014_2025.csv', dtype=str).drop_duplicates('encounter')
    vio = pd.read_csv(RAW / 'violations_2014_2025.csv', dtype=str)
    counts = (vio.assign(h=vio.high.eq('T'), m=vio.medium.eq('T'), l=vio.low.eq('T'))
                 .groupby('encounter').agg(n_high=('h', 'sum'), n_med=('m', 'sum'), n_low=('l', 'sum')))
    ins = ins.join(counts, on='encounter')
    # An inspection with no row in the violations table had no violations (checked against inspection reports).
    for c in ('n_high', 'n_med', 'n_low'):
        ins[c] = ins[c].fillna(0).astype(int)
    ins['date'] = pd.to_datetime(ins.inspect_dt)
    purpose = ins.purpose.fillna('')
    ins['routine'] = purpose.str.contains('Initial') & ~purpose.str.contains(ROUTINE_EXCLUDE)
    cols = ['encounter', 'id', 'description', 'municipal', 'date', 'purpose', 'routine', 'n_high', 'n_med', 'n_low']
    return ins[cols].sort_values(['id', 'date', 'encounter']).reset_index(drop=True)
