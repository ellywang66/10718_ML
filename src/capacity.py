"""Inspection capacity: how many restaurants ACHD gives a routine inspection each year.

C(y) = number of candidate restaurants (candidate list on Jan 1 of year y) with a routine inspection in year y.
For the new records system's first year (Aug 11, 2025 to Jul 31, 2026) we count restaurants with a
Comprehensive inspection in the WPRDC new-system table. Writes results/capacity.json.
"""
import json
import pandas as pd

import history
import splits
from config import RAW, RESULTS, TEST_WINDOW


def main():
    ins = history.load_inspections()
    out = {'old_system': {}}
    for year in range(2018, 2025):
        n_cand = len(splits.candidates(ins, f'{year}-01-01'))
        n_insp = len(splits.dev_set(ins, year))
        out['old_system'][year] = dict(candidates=n_cand, inspected=n_insp, share=round(n_insp / n_cand, 3))
    cur = pd.read_csv(RAW / 'inspections_current.csv', dtype=str).drop_duplicates('inspection_id')
    cur['date'] = pd.to_datetime(cur.inspect_dt)
    new = cur[cur.category.fillna('').str.contains('Restaurant') & cur.inspection_purpose.eq('Comprehensive')
              & cur.date.between(*TEST_WINDOW)]
    out['new_system_first_year'] = dict(window=list(TEST_WINDOW), routine_inspections=len(new),
                                        restaurants=int(new.groupby(['facility_name', 'num', 'zip_code']).ngroups))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'capacity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
