# Restaurant Inspection Risk: Baselines

Team TELOS (Benhao Huang, Joseph Liu, Shushi Hong, Yitong Wang), course 10-718.

The Allegheny County Health Department (ACHD) cannot inspect every restaurant each year, so it has to choose
whom to visit first. This repository implements the non-ML baselines for that decision and evaluates them on a
time-based split: each baseline ranks restaurants from their inspection history before the scoring date, and we
measure how many serious violators (two or more high-risk violations at one routine inspection) fall in the
top-ranked half of the restaurants inspected afterwards.

## Results

Test set: the 1,617 restaurants on the Jan 1, 2025 candidate list that had a routine inspection in the new
records system's first year (Aug 11, 2025 to Jul 31, 2026); 332 were serious violators. K = 809 is the top half.

| Baseline | Dev 2023 recall@50% | Dev 2024 recall@50% | Test recall@K=809 | Test precision@K=809 | Test recall@K=486 (30%) | Test AUC | Test recall@809, random ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| **History-rate rule (chosen on dev)** | 78.7% | 77.3% | **72.3%** | **29.7%** | 50.3% | 0.694 | 71.1-73.2% |
| Recency-weighted rate | 79.2% | 71.5% | 70.8% | 29.0% | 44.9% | 0.673 | 70.8% |
| Mean high-risk count | 75.4% | 74.2% | 70.5% | 28.9% | 51.5% | 0.667 | 69.3-71.1% |
| History rate, past three years | 76.0% | 71.8% | 68.7% | 28.2% | 49.1% | 0.673 | 66.3-69.3% |
| Last-result rule | 69.4% | 66.4% | 68.1% | 27.9% | 49.1% | 0.645 | 61.7-67.2% |
| Rotation | 57.9% | 51.5% | 57.8% | 23.7% | 38.0% | 0.565 | 57.8% |
| Most overdue first (current practice) | 53.6% | 53.0% | 55.4% | 22.7% | 36.1% | 0.547 | 55.4% |
| Random order (sanity check) | 50% | 50% | 50% | 20.5% | 30% | 0.500 | |

The last column is the 2.5th to 97.5th percentile of recall when ties in each rule's score are broken at random
(1,000 draws); a single value means no tie crosses the cutoff. `results/` holds the full numbers.

## Baselines

All use only inspection reports known 30 days before the scoring date. Higher score = visit first.

| Baseline | Score |
|---|---|
| Most overdue first | days since the last routine inspection / mean of its last four gaps between routine inspections |
| Rotation | days since the last routine inspection |
| Last-result rule | number of high-risk violations at the last routine inspection |
| History-rate rule | (s + 0.6) / (n + 2) for s routine inspections with a high-risk violation out of n; 0.3 with no history |
| Recency-weighted rate | the same rate with each inspection weighted by 0.5^(years ago) |
| History rate, past three years | the same rate over the past three years |
| Mean high-risk count | mean number of high-risk violations per routine inspection |

Ties: every rule orders restaurants by its score, then by the high-risk count at the last routine inspection,
then by how overdue they are, then by facility ID; missing values sort last. The rule with the best mean
recall on the development years (2023, 2024) is the chosen baseline.

## Split

| | Development | Test |
|---|---|---|
| Scoring date | Jan 1, 2023 and Jan 1, 2024 | Jan 1, 2025 |
| Candidates | restaurants whose most recent record became known in the previous three years | same rule |
| Label | first routine inspection that calendar year (old-system violations table) | first routine inspection Aug 11, 2025 to Jul 31, 2026 (new-system reports) |
| Size | 1,651 and 2,151 inspected restaurants | 1,617 inspected restaurants, 332 serious violators |

Restaurants without a routine inspection in the label window have no label and are left out; they are never
counted as negatives.

## Reproduce

```bash
pip install -r requirements.txt
python src/download_wprdc.py     # public WPRDC tables into data/raw/ (about 150 MB)
python src/run_baselines.py      # writes results/ in under a minute
```

`data/processed/accela_reports.csv` holds the violation counts parsed from 3,556 new-system inspection reports
(no inspector names); it supplies the test labels. To rebuild it from the Accela portal (slow, about one request
per second):

```bash
python src/accela/collect_reports.py 0 1   # or split across shards: <shard> <n_shards>
python src/accela/parse_reports.py
```

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | data sources, dates and constants of the evaluation setup |
| `src/download_wprdc.py` | downloads the public inspection and violation tables |
| `src/history.py` | one row per old-system inspection with its high/medium/low violation counts |
| `src/features.py` | point-in-time history features for a restaurant on a scoring date |
| `src/splits.py` | candidate lists and labels for the development and test sets |
| `src/baselines.py` | the rules and the tie-breaking order |
| `src/run_baselines.py` | evaluates every rule; writes `results/` |
| `src/accela/` | collects and parses new-system inspection reports |
| `results/` | `baseline_results.json`, `results_table.md`, `test_rankings.csv` (facility IDs and ranks only) |

## Data sources

- [Allegheny County Restaurant/Food Facility Inspections and Locations (WPRDC)](https://data.wprdc.org/dataset/allegheny-county-restaurant-food-facility-inspection-violations): inspections and violations 2014 to Jul 2025, and the new system's inspection table from Aug 2025.
- [ACHD Accela Citizen Access portal](https://aca-prod.accela.com/ALLEGHENYCO/Cap/CapHome.aspx?module=EnvHealth&TabName=Home): new-system inspection reports.
