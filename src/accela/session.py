"""Minimal client for ACHD's public Accela Citizen Access portal (ASP.NET postbacks, ~1 request/s)."""
import html
import re
import time
import requests

B = 'https://aca-prod.accela.com/ALLEGHENYCO'
H = 'https://aca-prod.accela.com'
HOME = B + '/Cap/CapHome.aspx?module=EnvHealth&TabName=Home'
ATT = B + ('/FileUpload/AttachmentsList.aspx?iframeid=ctl00_PlaceHolderMain_attachmentEdit&module=EnvHealth&isInConfirm=False'
           '&isdetail=True&isaccountmanager=False&isAdmin=False&isPeopleDocument=&agencyCode=ALLEGHENYCO&isForConditionDocument=N')
P = 'ctl00$PlaceHolderMain$generalSearchForm$'
NEW_REPORT = re.compile(r'FoodSafety_Inspection_Report_\w*?_(\d{8})_\d{6}\.pdf$')
PAUSE = 1.0


def session():
    s = requests.Session()
    s.headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36'
    return s


def get(s, url, **kw):
    time.sleep(PAUSE)
    return s.get(url, timeout=90, **kw)


def post(s, url, data, referer):
    time.sleep(PAUSE)  # the portal rejects postbacks without Referer and Origin headers
    return s.post(url, data=data, headers={'Referer': referer, 'Origin': H}, timeout=180)


def form_of(t):
    f = {}
    for tag in re.findall(r'<input[^>]*>', t):
        n = re.search(r'name="([^"]+)"', tag)
        if not n:
            continue
        ty = (re.search(r'type="([^"]+)"', tag) or [None, 'text'])[1]
        if ty in ('submit', 'button', 'image', 'checkbox', 'radio'):
            continue
        v = re.search(r'value="([^"]*)"', tag)
        f[n.group(1)] = html.unescape(v.group(1)) if v else ''
    for sel in re.finditer(r'<select[^>]*name="([^"]+)"[^>]*>(.*?)</select>', t, re.S):
        o = (re.search(r'<option[^>]*selected="selected"[^>]*value="([^"]*)"', sel.group(2))
             or re.search(r'<option[^>]*value="([^"]*)"', sel.group(2)))
        f[sel.group(1)] = html.unescape(o.group(1)) if o else ''
    return f


def caps(t):
    return sorted(set(re.findall(r'capID1=(\w+)&(?:amp;)?capID2=(\w+)&(?:amp;)?capID3=(\w+)', t)))
