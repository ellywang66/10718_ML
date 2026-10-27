# Restaurant Inspection Risk Ranking

Team TELOS (Benhao Huang, Joseph Liu, Shushi Hong, Yitong Wang), course 10-718.

The Allegheny County Health Department (ACHD) cannot inspect every restaurant each year, so it has to choose
whom to visit. This repository ranks candidate restaurants from their inspection history and evaluates the
rankings on a time-based split. Outcomes exist only for restaurants ACHD chose to inspect (the selective labels
problem), so every metric is reported with the assumption it rests on; see `proposed_changes.md` for the
long-term plan and `src/evaluate.py` for the metric definitions.

## Setup

One unit is a pair (candidate restaurant, scoring date). A restaurant is a candidate when its most recent known
record is filed under a restaurant category and became known within the previous three years. A record is
known 30 days after its inspection date. Features use known records only. The label describes the first
routine inspection within 12 months of the scoring date: `inspected` (there was one) and `serious` (it found
two or more high-risk violations). Every rule and model ranks all candidates; metrics are taken at
C = the number of candidates ACHD inspected within the window (`results/capacity.json`).

| | Development | Test |
|---|---|---|
| Scoring dates | Jan 1, 2023 and Jan 1, 2024 | Aug 11, 2025 (first day of the new records system) |
| Label source | old-system WPRDC tables | new-system inspection reports (`data/processed/accela_reports.csv`), linked to old ids |
| Training rows for models | Jan 1, 2016 onward, labels known before the scoring date | same |

At the test date, `inspected` means "has a linked Comprehensive report", which undercounts the 2,513
restaurants in the WPRDC new-system table (that table carries no old-system id).

## Results

`results/results_table.md` and `results/baseline_results.json` hold the non-ML rules, `results/ml_results.json`
the outcome models, `results/propensity_results.json` the P(inspected) experiment, and `results/diagnostics/`
the coverage, selection, base-rate, bound-width, noise, and permit-status diagnostics.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.data.download_wprdc      # public WPRDC tables into data/raw/ (about 150 MB), records data/raw/MANIFEST.json
python -m src.panel                    # data/processed/panel.parquet, a few minutes
python -m src.analysis.capacity        # results/capacity.json
python -m src.models.run_baselines     # non-ML rules
python -m src.models.train_outcome     # ML outcome models q(x); writes results/q_scores.parquet
python -m src.models.train_lr5         # five-feature logistic regression
python -m src.analysis.diagnostics     # results/diagnostics/
python -m src.models.train_propensity  # P(inspected) models and the Bayes-rule score
pytest
```

Run every command from the repository root. `data/processed/accela_reports.csv` holds the violation counts parsed
from 3,556 new-system inspection reports (no inspector names). To rebuild it from the Accela portal (slow, about
one request per second):

```bash
python -m src.data.accela.collect_reports 0 1   # or split across shards: <shard> <n_shards>
python -m src.data.accela.parse_reports
```

## Layout

`src/` is one package, organized by pipeline stage.

| Path | Contents |
|---|---|
| `src/config.py` | data sources, scoring dates, horizon, lag, and the other fixed choices |
| `src/panel.py` | candidates, labels, and features for every scoring date; `inspected()` and `serious()` apply the horizon |
| `src/evaluate.py` | ranking metrics over the candidate set at a given C |
| `src/data/download_wprdc.py` | downloads the public tables and writes `data/raw/MANIFEST.json` |
| `src/data/history.py` | one row per old-system inspection with violation counts and inspection type |
| `src/data/purpose_types.csv` | maps every `purpose` value to routine, complaint, reinspection, new_facility, other |
| `src/data/accela/` | collects and parses new-system inspection reports |
| `src/features/rates.py`, `src/features/risk.py` | point-in-time risk features (inputs of the outcome models) |
| `src/features/selection.py` | point-in-time inspection-decision features (inputs of the propensity model): schedule tier, calendar-year status, coverage, anniversary, size, complaints, placard |
| `src/models/baselines.py` | the rules and the tie-breaking order |
| `src/models/run_baselines.py` | evaluates every rule |
| `src/models/train_outcome.py`, `src/models/train_lr5.py` | ML outcome models q(x) = P(serious given inspected, x) |
| `src/models/train_propensity.py` | propensity models P(inspected given selection features), the Bayes-rule score q * e, IPW yield |
| `src/analysis/capacity.py` | C per scoring date |
| `src/analysis/diagnostics.py` | diagnostics D1 to D6 |
| `src/analysis/propensity_analysis.py` | tables on the propensity models; saved copy in `results/diagnostics/propensity_analysis.txt` |
| `tests/` | cutoff and label tests on synthetic rows |
| `data/processed/` | `panel.parquet`, `accela_reports.csv`, `purpose_values.csv`, `status_values.csv` |

## Data sources

- [Allegheny County Restaurant/Food Facility Inspections and Locations (WPRDC)](https://data.wprdc.org/dataset/allegheny-county-restaurant-food-facility-inspection-violations): inspections and violations 2014 to Jul 2025, the new system's inspection table from Aug 2025, and the geocoded facility roster (seat count, floor area; a current snapshot).
- [ACHD Accela Citizen Access portal](https://aca-prod.accela.com/ALLEGHENYCO/Cap/CapHome.aspx?module=EnvHealth&TabName=Home): new-system inspection reports.
