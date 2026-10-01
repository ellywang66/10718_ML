"""Download new-system inspection reports (Aug 11, 2025 onward) from ACHD's Accela portal.

For each restaurant with a Comprehensive inspection in the WPRDC new-system table, search Accela by facility
name, keep records whose address starts with the same street number, and download every new-format report
attached to them. Migrated records keep the old facility ID as their record number, which links the two systems.
Resumable; writes one JSON line per restaurant to data/accela/collection.<shard>.jsonl and PDFs under
data/accela/pdfs/. Takes about a day at one request per second; run several shards in parallel.

Usage: python src/accela/collect_reports.py <shard> <n_shards> [limit]
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import session as C  # noqa: E402
from config import RAW, ROOT  # noqa: E402

OUT = ROOT / 'data' / 'accela'


def result_rows(t):
    rows = []
    for tr in re.findall(r'<tr class="ACA_TabRow[^"]*".*?</tr>', t, re.S):
        c = C.caps(tr)
        if not c:
            continue
        cells = [re.sub(r'\s+', ' ', C.html.unescape(re.sub(r'<[^>]+>', ' ', x))).strip()
                 for x in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
        rows.append(dict(cap=c[0], cells=[x for x in cells if x]))
    return rows


def search_name(s, name):
    f = C.form_of(C.get(s, C.HOME).text)
    f.update({C.P + 'txtGSProjectName': name, C.P + 'txtGSStartDate': '01/01/1950', C.P + 'txtGSEndDate': '12/31/2030',
              '__EVENTTARGET': 'ctl00$PlaceHolderMain$btnNewSearch', '__EVENTARGUMENT': ''})
    p = C.post(s, C.HOME, f, C.HOME)
    if 'CapDetail' in p.url:
        return [dict(cap=C.caps(p.url + p.text)[0], cells=['single'], single=True)]
    return result_rows(p.text)


def record_reports(s, cap, tag):
    det = (f'{C.B}/Cap/CapDetail.aspx?Module=EnvHealth&TabName=EnvHealth&capID1={cap[0]}&capID2={cap[1]}'
           f'&capID3={cap[2]}&agencyCode=ALLEGHENYCO')
    d = C.get(s, det).text
    addr = re.search(r'Work Location\s*(?:<[^>]*>\s*)*([^<]+)', d)
    a = C.get(s, C.ATT, headers={'Referer': det}).text
    all_files, got, pages = [], [], 0
    while True:  # the attachment grid shows 5 rows per page; follow "Next >" until the last page
        pages += 1
        files = re.findall(r'id="attachmentList_gdvAttachmentList_(ctl\d+)_lblName">([^<]+)<', a)
        all_files += [n for _, n in files]
        for ctl, name in files:
            m = C.NEW_REPORT.search(name.strip())
            if not m or m.group(1) < '20250811':
                continue
            f = C.form_of(a); f['__EVENTTARGET'] = f'attachmentList$gdvAttachmentList${ctl}$lnkFileName'; f['__EVENTARGUMENT'] = ''
            p = C.post(s, C.ATT, f, C.ATT)
            if p.content[:4] == b'%PDF':
                path = OUT / 'pdfs' / tag / f'{cap[0]}-{cap[2]}_{name.strip().lstrip("/")}'
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(p.content)
                got.append(str(path.relative_to(OUT)))
        nxt = re.search(r"__doPostBack\(&#39;(attachmentList\$gdvAttachmentList\$ctl\d+\$ctl\d+)&#39;,&#39;&#39;\)\">Next &gt;</a>", a)
        if not nxt or pages >= 20:
            break
        f = C.form_of(a); f['__EVENTTARGET'] = nxt.group(1); f['__EVENTARGUMENT'] = ''
        a = C.post(s, C.ATT, f, C.ATT).text
    return dict(cap=cap, address=addr.group(1).strip() if addr else None, files=all_files, pages=pages, downloaded=got)


def main():
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    cur = pd.read_csv(RAW / 'inspections_current.csv', dtype=str).drop_duplicates('inspection_id')
    rest = cur[cur.category.fillna('').str.contains('Restaurant') & cur.inspection_purpose.eq('Comprehensive')]
    fac = rest.groupby(['facility_name', 'num', 'street', 'zip_code']).size().reset_index().sort_values(['facility_name', 'num'])
    fac = fac.iloc[shard::n_shards]
    if limit:
        fac = fac.head(limit)
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f'collection.{shard}.jsonl'
    done = {json.loads(line)['key'] for line in open(log)} if log.exists() else set()
    s = C.session(); n = 0; t0 = time.time()
    for _, x in fac.iterrows():
        key = f'{x.facility_name}|{x.num}|{x.zip_code}'
        if key in done:
            continue
        rec = dict(key=key, records=[])
        try:
            rows = search_name(s, x.facility_name)
            num = re.match(r'\d+', str(x.num)); num = num.group(0) if num else None
            for row in rows:
                txt = ' '.join(row['cells'])
                rec_no = re.search(r'\b(\d{12}|\d{4,6}|FSP-[A-Z]+-\d+)\b', txt)
                if not row.get('single') and num and not re.search(rf'(^|\s){num}(\s|-)', txt):
                    continue
                info = record_reports(s, row['cap'], re.sub(r'\W+', '_', key)[:80])
                info['record_no'] = rec_no.group(1) if rec_no else None
                rec['records'].append(info)
        except Exception as e:  # keep going; a failed restaurant is retried on the next run
            rec['error'] = str(e)[:200]; time.sleep(20); s = C.session()
        if 'error' not in rec:
            with open(log, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        n += 1
        if n % 10 == 0:
            print(f'shard {shard}: {n} done, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
