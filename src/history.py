"""One row per old-system inspection (Jan 2014 to Jul 2025) with its high/medium/low violation counts."""
from pathlib import Path

import pandas as pd
from config import RAW

# Every observed `purpose` value mapped to routine / complaint / reinspection / new_facility / other. Features and
# development labels use the same routine set: "Initial", "COVID-19, Initial Inspection", "Initial, Service Request".
PURPOSE_TYPES = pd.read_csv(Path(__file__).with_name('purpose_types.csv'), dtype=str).set_index('purpose').type
ROUTINE = set(PURPOSE_TYPES.index[PURPOSE_TYPES.eq('routine')])


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
    unknown = set(purpose[purpose.ne('')]) - set(PURPOSE_TYPES.index)
    assert not unknown, f'purpose values missing from purpose_types.csv: {unknown}'
    ins['routine'] = purpose.isin(ROUTINE)
    cols = ['encounter', 'id', 'description', 'municipal', 'date', 'purpose', 'routine', 'n_high', 'n_med', 'n_low']
    return ins[cols].sort_values(['id', 'date', 'encounter']).reset_index(drop=True)
