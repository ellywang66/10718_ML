"""Inspection capacity C(t): candidates with a routine inspection within the horizon of each scoring date.

Computed from the panel, so C uses the same candidate and routine definitions as every metric. For the test
date two counts are written: `linked_candidates` (candidates with a linked Accela Comprehensive report, the
count every test-date ranking metric uses) and `wprdc_restaurants` (all restaurants with a Comprehensive
inspection in the WPRDC new-system table, which has no old-system id; context only). Writes results/capacity.json.
"""
import json

import pandas as pd

from src import panel
from src.config import HORIZON_MONTHS, RAW, RESULTS, TEST_SCORING_DATE, TEST_WINDOW


def main() -> None:
    p = panel.load()
    out = {'horizon_months': HORIZON_MONTHS, 'by_scoring_date': {}}
    for t in sorted(p.scoring_date.unique()):
        rows = panel.at(p, t)
        n, c = len(rows), int(rows.inspected.sum())
        out['by_scoring_date'][str(pd.Timestamp(t).date())] = dict(
            candidates=n, inspected=c, share=round(c / n, 3), serious=int(rows.serious.sum()),
            label_source='accela_linked' if pd.Timestamp(t) >= pd.Timestamp(TEST_SCORING_DATE) else 'old_system')
    cur = pd.read_csv(RAW / 'inspections_current.csv', dtype=str).drop_duplicates('inspection_id')
    cur['date'] = pd.to_datetime(cur.inspect_dt)
    new = cur[cur.category.fillna('').str.contains('Restaurant') & cur.inspection_purpose.eq('Comprehensive')
              & cur.date.between(*TEST_WINDOW)]
    out['test_date'] = dict(
        scoring_date=TEST_SCORING_DATE,
        linked_candidates=out['by_scoring_date'][TEST_SCORING_DATE]['inspected'],
        wprdc_restaurants=int(new.groupby(['facility_name', 'num', 'zip_code']).ngroups),
        wprdc_routine_inspections=int(len(new)), wprdc_window=list(TEST_WINDOW))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'capacity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
