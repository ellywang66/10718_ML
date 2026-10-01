"""Paths, data sources and the fixed choices of the evaluation setup."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'

# Western Pennsylvania Regional Data Center, "Allegheny County Restaurant/Food Facility Inspections and Locations"
WPRDC = {
    'inspections_2014_2025.csv': 'https://data.wprdc.org/datastore/dump/4ea730d8-2bf9-4783-b0a5-b41ed687097e',
    'violations_2014_2025.csv': 'https://data.wprdc.org/datastore/dump/1a1329e2-418c-4bd3-af2c-cc334e7559af',
    'inspections_current.csv': 'https://data.wprdc.org/datastore/dump/1c9cf8d8-f3db-4364-b925-d7078c2fb02f',
}

RESTAURANT = {'Restaurant with Liquor', 'Restaurant without Liquor',
              'Chain Restaurant with Liquor', 'Chain Restaurant without Liquor'}

REPORT_LAG_DAYS = 30            # a report counts as known 30 days after the inspection
CANDIDATE_YEARS = 3              # candidates: any record known in the three years before the scoring date
SERIOUS = 2                      # serious violator: two or more high-risk violations at one inspection

DEV_YEARS = (2023, 2024)         # scoring date Jan 1; label = first routine inspection that calendar year
TEST_SCORING_DATE = '2025-01-01'
TEST_WINDOW = ('2025-08-11', '2026-07-31')  # first year of the new (Accela) records system
BUDGETS = (0.3, 0.5)             # K = top 30% and top 50% of inspected restaurants
RANDOM_TIE_DRAWS = 1000
SEED = 20261005
