# Predicting which restaurants ACHD inspects

## Purpose
The Allegheny County Health Department (ACHD) gives a routine inspection to only some restaurants each year. Violation outcomes exist only for inspected restaurants, and ACHD does not choose them at random. This problem is called selective labels. One proposed fix needs a model of the inspection decision itself. That model is the propensity e(x), the probability that a restaurant gets a routine inspection in the next 12 months given its features x. This experiment tests whether public data can predict e(x). It does not evaluate violation outcomes or any ranking for the project submission.

## Data
- **Sources:** the WPRDC (Western Pennsylvania Regional Data Center) inspection and violation tables for 2014 to July 2025, the WPRDC geocoded facilities table (seat count, square feet), and reports from ACHD's newer Accela system from August 2025 onward. The Accela reports are linked to the old restaurant ids for the test date.
- **Unit:** one row per candidate restaurant and scoring date t. A restaurant is a candidate at t when its latest record has a restaurant category and was filed within the 3 years before t.
- **Known records:** a record is known at t only when its inspection date plus a 30-day reporting lag is before t. Every feature uses known records only.
- **Scoring dates:** January 1 of 2016 to 2024 (about 4,500 to 5,000 candidates each), August 11 of 2019 to 2023, and the test date 2025-08-11 (4,664 candidates).
- **Label:** 1 when the restaurant has a routine inspection in the 12 months after t. `src/purpose_types.csv` defines which inspection purposes are routine. At the test date the label is "has a linked Accela Comprehensive report", which undercounts true inspections.
- **Share inspected:** 0.78 (2016), 0.63 (2018 and 2019), 0.37 (2020), 0.58 (2021), 0.31 (2022), 0.36 (2023), 0.45 (2024), 0.37 (test date).

## Features
Each of two feature sets has 23 columns. A third set is their union. Missing values get median imputation plus a missing indicator per column.
- **Risk inputs** (the set the violation model uses): history rates of serious results, last result, counts of high violations, days since the last routine visit, complaint and reinspection counts, and liquor and chain flags.
- **Selection inputs** (built for this experiment after a review of ACHD's published scheduling rules): days to the due date under ACHD priority codes (H tier 6 months, code 1 12 months, code 2 24 months), H-tier flag, critical categories at the last visit, reinspection-required flag, calendar years since the last routine visit, share of the last 3 years with a visit, routine visits per year, coverage of the zip and municipality in the previous year, share of the zip's visits in the last 90 days, months to the permit anniversary, seat count, square feet, service requests and zip complaints in the last year, last placard status, and the shared schedule, chain, and liquor columns.

## Models and validation
- **Models** (scikit-learn): logistic regression (LR) with an L2 penalty and C in {0.01, 0.1, 1}, Gaussian Naive Bayes (NB) with the default var_smoothing, and HistGradientBoostingClassifier (GB) with learning_rate in {0.03, 0.1} and max_iter in {100, 300}. The 8 configurations run on each of the 3 feature sets.
- **Metrics:** AUC is the area under the ROC curve. The Brier score is the mean squared error of the predicted probability. The "shifted" Brier score is computed after a logit intercept shift that sets the mean prediction to the observed share.
- **Forward validation** (the deployment setting): for each year V from 2022 to 2024, the model trains on January 1 dates t whose label window plus lag ends before V-01-01 (so t <= V-2). It then scores the rows of year V.
- **Within-year check:** 5-fold cross-validation (CV) on the rows of year V alone. This check is a diagnostic only, because it trains on the same year's decisions.
- **Model choice:** highest mean forward AUC over the three years, with ties broken by shifted Brier score. The chosen model is GB with learning_rate 0.03, max_iter 300, and selection inputs.
- **Test date:** each model is retrained on the August 11 rows of 2019 to 2023 and scored on 2025-08-11.

## Results
| Inputs | GB forward AUC 2022 / 2023 / 2024 | GB within-year AUC 2022 / 2023 / 2024 | Test-date AUC LR / NB / GB |
|---|---|---|---|
| Risk | 0.695 / 0.615 / 0.672 | 0.723 / 0.657 / 0.698 | 0.612 / 0.625 / 0.645 |
| Selection | 0.711 / 0.625 / 0.672 | 0.801 / 0.754 / 0.756 | 0.650 / 0.634 / 0.656 |
| Union | 0.711 / 0.635 / 0.677 | 0.806 / 0.750 / 0.751 | 0.648 / 0.628 / 0.651 |

- **LR and NB:** forward AUC 0.58 to 0.68 on every input set. NB on selection inputs reached 0.681 in 2024 against 0.672 for GB. GB was higher in every other cell. Mean shifted Brier was 0.33 for NB and 0.22 for GB.
- **Calibration** (mean prediction against observed share): 0.45 vs 0.31 (2022), 0.48 vs 0.36 (2023), 0.37 vs 0.45 (2024). The features omit the year's inspection volume, so the level is wrong every year. After the shift, 2024 deciles run from 0.14 observed at 0.03 predicted to 0.66 observed at 0.74 predicted.
- **Training window:** training on only the last usable year, the last three, or only pre-2020 dates changed forward AUC by at most 0.02.
- **Single-feature within-year AUC, 2024:** days to due 0.665, days since last routine 0.661, calendar years since routine 0.658, share of 3 years covered 0.647, routine visits per year 0.624, municipality coverage 0.602, zip coverage 0.593, number of routine visits 0.573, chain flag 0.564. Every other selection feature scored 0.50 to 0.53.
- **Permutation importance** (AUC drop, chosen forward model, 2024): days since any visit 0.100, days to due 0.028, routine visits per year 0.019, chain flag 0.015, all others below 0.01.
- **Inspection rate, 2024:** 0.28 when more than a year overdue, 0.40 to 0.50 when overdue up to a year, 0.65 when due within 90 days. Restaurants visited in each of the last 3 years: 0.72, against 0.28 for those visited in none. H tier: 0.39, against 0.47 for the rest.

## Why forward AUC is lower than within-year AUC
Three follow-up checks used the chosen GB configuration on selection inputs.
1. **Leave-one-year-out** (train on every other January date, later ones included): AUC 0.713 / 0.650 / 0.703. This is close to forward AUC, so the gap is a year-specific pattern rather than drift over time.
2. **Area features:** within-year CV with no zip shared between folds gave 0.688 / 0.647 / 0.720. Ordinary within-year CV without the four area features (zip and municipality coverage, zip visit share, zip complaints) gave 0.720 / 0.650 / 0.703. Within-year permutation importance of municipality coverage is 0.029, against -0.006 forward. So the within-year advantage comes from learning which areas ACHD works that year from same-year neighbors.
3. **County-level "seeding" features** (routine visits in the 60, 90, and 180 days before t, year-over-year ratios, backlog share): forward AUC changed by +0.000, +0.024, and +0.022 (2022 to 2024). The mean prediction moved further from the observed share (2024: 0.25 predicted, 0.45 observed). Within schedule groups (not yet due, overdue, more than a year overdue), forward AUC is 0.64 to 0.69 against 0.70 to 0.80 within-year.

**Likely mechanism** (my judgment from an online search, not measured in the data): ACHD assigns each of up to 30 inspectors a geographic territory (Board of Health minutes, January 2025). Staffing fell to 14 of 30 in March 2022 and returned to full in January 2025 (PublicSource; Board minutes). Other inspectors absorb vacant territories, so the covered areas change each year. No public table records this. Inspector names appear only in the old-system report PDFs.

## What was not done
- NB was not tuned.
- No features use inspector identity or territory.
- No one evaluated the violation ranking with this model.

## Files
- Code: `src/models/train_propensity.py`, `src/features/selection.py`, `src/analysis/propensity_analysis.py`
- Results: `results/propensity_results.json`, `results/propensity_runs.json`, `results/diagnostics/propensity_analysis.txt`
