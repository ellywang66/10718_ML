"""Parse downloaded new-system inspection reports into data/processed/accela_reports.csv.

Page 1 of each report has two 9-number section summaries [exceptional, S, NA, NO, V, High, Med, Low, Imminent]
(items 1-25 and 26-33); their sums give the violation counts. "*HIGH RISK*" markers in the violation comments
are counted as a cross-check. Each report is linked to an old-system facility ID: migrated records carry it as
the Client ID; a new permit is linked when its search returned exactly one old-system record at that address.
Inspector names are not written out.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, ROOT  # noqa: E402

OUT = ROOT / 'data' / 'accela'


def field(lines, label):
    for k, line in enumerate(lines):
        if line.strip().startswith(label):
            rest = line.strip()[len(label):].strip()
            if rest:
                return rest
            for c in lines[k + 1:k + 3]:
                if c.strip():
                    return c.strip()
    return None


def parse(path):
    r = pypdf.PdfReader(path)
    lines = r.pages[0].extract_text().splitlines()
    full = '\n'.join(pg.extract_text() for pg in r.pages)
    runs, cur = [], []
    for x in (line.strip() for line in lines):
        if re.fullmatch(r'\d+', x):
            cur.append(int(x))
        else:
            if len(cur) >= 9:
                runs.append(cur[:9])
            cur = []
    if len(cur) >= 9:
        runs.append(cur[:9])
    out = dict(client_id=field(lines, 'Client ID:'), inspection_date=field(lines, 'Inspection Date:'),
               purpose=field(lines, 'Purpose:'), category=field(lines, 'Category Code:'))
    if len(runs) >= 2:
        s1, s2 = runs[0], runs[1]
        out.update(n_viol=s1[4] + s2[4], n_high=s1[5] + s2[5], n_med=s1[6] + s2[6], n_low=s1[7] + s2[7],
                   n_imminent=s1[8] + s2[8])
    out['mk_high'] = len(re.findall(r'\*\s*HIGH RISK', full.upper()))
    return out


def main():
    rows = []
    for f in sorted(OUT.glob('collection.*.jsonl')):
        for line in open(f):
            rec = json.loads(line)
            old_records = [r.get('record_no') for r in rec.get('records', [])
                           if r.get('record_no') and re.fullmatch(r'\d{5,12}', r['record_no'])]
            for r in rec.get('records', []):
                for rel in r.get('downloaded', []):
                    try:
                        row = parse(OUT / rel)
                    except Exception as e:
                        row = dict(error=str(e)[:100])
                    row['old_candidates'] = ';'.join(old_records)
                    rows.append(row)
    d = pd.DataFrame(rows)
    d['date'] = pd.to_datetime(d.inspection_date, errors='coerce')
    d = d.dropna(subset=['date']).drop_duplicates(['client_id', 'date', 'purpose'])
    is_old = d.client_id.fillna('').str.fullmatch(r'\d{5,12}')
    one = d.old_candidates.fillna('').str.split(';').map(lambda x: x[0] if len(x) == 1 and x[0] else None)
    d['old_id'] = np.where(is_old, d.client_id, None)
    d['linked_via'] = np.where(is_old, 'client_id', np.where(one.notna(), 'same_address_record', 'unlinked'))
    d['old_id'] = d.old_id.fillna(one)
    cols = ['client_id', 'old_id', 'linked_via', 'inspection_date', 'purpose', 'category',
            'n_viol', 'n_high', 'n_med', 'n_low', 'n_imminent', 'mk_high']
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d[cols].to_csv(PROCESSED / 'accela_reports.csv', index=False)
    print(len(d), 'reports;', d.linked_via.value_counts().to_dict())


if __name__ == '__main__':
    main()
