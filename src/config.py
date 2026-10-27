"""Paths, data sources and the fixed choices of the evaluation setup."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
DIAGNOSTICS = RESULTS / 'diagnostics'

# Western Pennsylvania Regional Data Center, "Allegheny County Restaurant/Food Facility Inspections and Locations"
WPRDC = {
    'inspections_2014_2025.csv': 'https://data.wprdc.org/datastore/dump/4ea730d8-2bf9-4783-b0a5-b41ed687097e',
    'violations_2014_2025.csv': 'https://data.wprdc.org/datastore/dump/1a1329e2-418c-4bd3-af2c-cc334e7559af',
    'inspections_current.csv': 'https://data.wprdc.org/datastore/dump/1c9cf8d8-f3db-4364-b925-d7078c2fb02f',
    # "Geocoded Food Facilities (as of 2025)": one row per facility with seat_count, sq_feet, p_code (current snapshot)
    'geocoded_facilities.csv': 'https://data.wprdc.org/datastore/dump/112a3821-334d-4f3f-ab40-4de1220b1a0a',
}

RESTAURANT = {'Restaurant with Liquor', 'Restaurant without Liquor',
              'Chain Restaurant with Liquor', 'Chain Restaurant without Liquor'}

REPORT_LAG_DAYS = 30            # a report is known once inspection date + 30 days < scoring date
LOOKBACK_YEARS = 3              # candidates: any record known in the three years before the scoring date
SERIOUS = 2                     # serious violator: two or more high-risk violations at one inspection
HORIZON_MONTHS = 12             # label window: [scoring date, scoring date + 12 months)
DAYS_PER_MONTH = 30.44

# Scoring dates. Jan 1 dates carry old-system labels. Aug 11 dates 2019-2023 exist for the propensity refit at
# the test date (same season). The test date is the first day of the new (Accela) records system.
DEV_YEARS = (2023, 2024)
TEST_SCORING_DATE = '2025-08-11'
SCORING_DATES = tuple(f'{y}-01-01' for y in range(2016, 2025)) + tuple(f'{y}-08-11' for y in range(2019, 2024)) \
    + (TEST_SCORING_DATE,)

# Last date each label source covers. Old system: the WPRDC table ends in July 2025. Accela: the scrape's
# last complete month (September 2026 is partial).
OLD_DATA_END = '2025-07-31'
ACCELA_DATA_END = '2026-08-31'
ACCELA_START = TEST_SCORING_DATE     # scoring dates from here on take labels from the Accela reports
TEST_WINDOW = ('2025-08-11', '2026-07-31')  # used only to count the new system's first-year inspections

BUDGETS = (0.3, 0.5)             # legacy: K = top 30% and top 50% of inspected restaurants
RANDOM_TIE_DRAWS = 1000
SEED = 20261005
