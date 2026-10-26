# Plan: data reprocessing and experimental redesign

This plan replaces the proof-of-concept evaluation in @src/run_baselines.py and the design in @doc.txt.
It has three parts. Part 1 lists the data reprocessing steps. Part 2 lists the diagnostic experiments
that must run before any design is fixed. Part 3 lists the factors that the final design must weigh.
Part 3 does not prescribe the design, because most decisions depend on results from Part 2.

## Decisions already made

- The decision we support is which candidate restaurants receive one of ACHD's routine inspection slots.
  The quantity of interest is inspector efficiency, which is serious violators found per routine inspection.
- Timing of detection within a year is out of scope. We do not know when or how ACHD would use a list,
  so we cannot evaluate timing correctly. Drop the "first six months" framing and the 50% cutoff.
- The current metric conditions on ACHD's own selection of restaurants. It measures P(violation | inspected),
  and the design must address the gap between that and P(violation).
- The ranking decides which restaurants are inspected, so the comparator for current practice is the set
  ACHD actually inspected and its observed yield. Rule proxies such as "most overdue first" are not exactly current practice.

## Part 1. Data reprocessing

The output is one table with one row per (candidate restaurant, scoring date). Every column is point-in-time
correct with respect to the scoring date. Write tests for the cutoff logic and the label construction.

Step 1. Rebuild the old-system inspection table from the WPRDC dumps (@src/history.py).
- Keep every column, including `status` (permit status), `placard_st`, `reispt_cd`, `bus_st_date`,
  `category_cd`, and `municipal`. The current loader drops `status`.
- Get the data dictionary for the `status` codes (observed values 0, 1, 2, 6, 7, 9) and record it in the repo.
- Pin the download. Record the dump date and row counts in `data/raw/MANIFEST.json`, because WPRDC serves live dumps.
- Keep outcomes that ACHD records without a routine inspection: consumer alert and closed placards, complaint and
  foodborne-illness inspections, and enforcement actions. They exist for every candidate, inspected or not, and
  Part 2 may use them as auxiliary outcomes.

Step 2. Define inspection types once, in one module, and use that module everywhere.
- The purpose field has 37 distinct values, many compound ("Initial, Foodborne Illness", "COVID-19, Initial Inspection").
  Classify each value into routine, complaint, reinspection, new facility, and other. Store the mapping as a table in the repo.
- @src/history.py (contains "Initial" minus three exclusions) and @src/splits.py (equals "Initial") currently disagree.
  Replace both with the mapping.

Step 3. Define candidates (@src/splits.py).
- Restrict to restaurant categories as now, and add permit status. Record how many restaurants each filter removes per year.
- Keep the look-back window as a parameter. Three years is the current value and has no justification yet.
- Also build the candidate list from the WPRDC facility roster where it is available, and compare the two lists.

Step 4. Build labels as events, not as a single column.
- For every (restaurant, scoring date), store the date and high-risk count of each routine inspection after the
  scoring date, in order. Any horizon H and any threshold can then be derived without rebuilding.
- Store the gap between the scoring date and the first routine inspection, and the gap from the previous
  routine inspection. Part 2 needs both.
- Do not encode a threshold (currently 2) or a horizon (currently one year) in the table.

Step 5. Build features with the known-date rule.
- Each inspection becomes known at inspection date plus a lag. Keep the lag as a parameter (currently 30 days,
  unmeasured). Store the known date on each row so the cutoff is one comparison.
- Add complaint history (count and recency of complaint inspections), reinspection history, category,
  municipality, business start date, and the last placard status. Complaints matter twice: they predict outcomes
  and they drive selection, so the selection diagnostic in Part 2 needs them.
- Keep the smoothing constants out of the feature table. Store raw counts (n routine, n with high-risk, sums of
  weighted counts) and apply any prior at scoring time.

Step 6. Rebuild the capacity table (@src/capacity.py).
- Count routine inspections and inspected candidates per calendar year on the new candidate definition.
- Apply the same candidate restriction to the new-system year. The current 2,513 is unrestricted and is not comparable.

Step 7. Treat the new-system data as a separate, lower-trust dataset.
- Fix @src/accela/parse_reports.py. The category reader runs onto the next field (53 rows read "Priority Code:").
  Parse the Priority Code field itself, since it is ACHD's frequency tier.
- Consider collecting the priority code per facility from ACHD's restaurant finder, which can search by priority
  code (see Sources). A news report describes Priority Code H as a follow-up within five to seven months after two
  or more high-risk violations. If that holds, ACHD's selection depends on the same past result that every risk
  rule ranks on, and the code is both ACHD's schedule and ACHD's own severity label.
- Reconcile the two high-risk counts. The section-summary count and the `*HIGH RISK*` marker count disagree on
  580 of 3,556 reports. Sample disagreements, read the PDFs, and record which count is right and why.
- Record link quality per report (`linked_via`), the 406 unlinked reports, and the restaurants in the WPRDC
  new-system table that the scrape never found. Report the match rate against the 2,513 restaurants.
- Keep the Accela collection (`data/accela/`) in an archive outside git so results can be rebuilt without a day of scraping.

Step 8. Keep old-system inspection reports in reserve.
- @doc.txt cites 37,560 archived old-system reports used to check labels. If they carry inspector identity, store a
  hashed inspector id per inspection. Part 2 needs it for the label-stability experiment. Do not store names.
- Also record the inspector's district or assignment area if the reports show it. Experiment M needs assignment,
  not only identity.

## Part 2. Experiments to run before fixing the design

Each experiment produces a table or figure under `results/diagnostics/` and a short note. Run them on the
old-system data only, with scoring dates from 2016 through 2023. Use the new-system year only in experiment D.

A. Coverage against horizon. For each scoring date, the share of candidates with a routine inspection within
   H months, for H from 6 to 48. Break down by category, municipality, history length, and prior-score decile.

B. Selection mechanism. For each scoring date, model P(routine inspection within 12 months | features).
   Report discrimination, the features that drive it, and the relationship between selection and each
   candidate risk score. Compare 2020 to 2022 with the other years. Repeat with "within 24 months".

C. Outcome stability against gap. Among pairs of consecutive routine inspections of the same restaurant,
   measure P(serious at the second | result at the first) as a function of the gap in months, for several
   thresholds. Also measure how often a restaurant's label changes between consecutive inspections.

D. Label construct. Base rates of one, two, and three or more high-risk violations by year and by records
   system. The reconciliation from Part 1 Step 7. If hashed inspector ids exist, the spread of high-risk rates
   across inspectors with similar restaurants.

E. Candidate population. How many candidates per year have an expired permit, no record in two years, or a
   business start date after the scoring date. The effect of each filter on the capacity shares.

F. Bound width. For several rankings (history rate, last result, random, business age), the share of the top C
   that is unlabeled at each horizon H, where C is ACHD's routine count that year. This decides whether a
   bounded yield comparison can be conclusive.

G. Score granularity and smoothing. For the rate-based rules, the number of distinct scores, the size of the tie
   group at the cutoff, and the yield across a full grid of prior means and weights. @results/prior_check.json
   shows the prior mean moves recall by about three points while the weight does nothing. Measure this with yield.

H. Complaints and outcomes. P(serious at the next routine inspection | complaint in the past year), and the
   share of selected restaurants that had a recent complaint.

I. Noise. Bootstrap standard errors of yield at C for a few rankings, by restaurant and by scoring date.
   Measure how correlated the errors are across scoring dates when the same inspection labels several of them.

J. Propensity and weighted yield. P(inspected | x) and P(inspected and serious | x) are observed for every
   candidate, so both can be fit on the full candidate set. Their ratio is still P(serious | inspected, x), but the
   propensity gives an off-policy estimate of a ranking's yield: sum A_i / P(B_i | x_i) over the inspected
   restaurants in its top C, with a doubly robust variant that adds an outcome model. Report the propensity
   distribution per year, its minimum, the effective sample size after weighting, and the weighted yield with its
   standard error for a few rankings. Compare with the bounds from F. Agreement is evidence because the two rest on
   different assumptions.

K. Auxiliary outcomes on the full candidate set. For each ranking tier, the rate of complaints, foodborne-illness
   reports, consumer alert and closed placards, and enforcement actions in the following year, over all candidates.
   Also the agreement between the placard outcome and each count threshold among inspected restaurants.

L. Near-unselected subpopulations. P(serious | x) estimated inside groups with near-complete coverage, compared with
   the estimate from all inspected restaurants. Candidate groups: 2018 and 2019 (63% coverage), new restaurants at
   their first routine inspection, Priority H follow-ups, and municipalities or districts with high coverage in a
   year. The new-restaurant group also gives an unselected estimate of the no-history risk.

M. Inspector or district heterogeneity. If assignment is recorded, how much coverage varies across inspectors or
   districts with similar restaurants, and whether assignment looks unrelated to restaurant risk. This decides
   whether the contraction method of Lakkaraju et al. (see Sources) applies, with inspectors in the role of judges.

N. External calibration. NYC requires at least one unannounced inspection of every establishment per cycle of about
   a year (see Sources), so its public data has little selection. Compare the relationship between history features
   and current violations in NYC with the relationship in Allegheny's inspected subset.

## Part 3. Factors the final design must weigh

The final design is not prescribed here. Whatever it is, the write-up must state how it handled each factor
below, with reference to the Part 2 results that informed the choice.

Metric and the unlabeled restaurants
- The metric lives on the candidate set and is a yield per inspection at a fixed number of inspections.
  The number C per year comes from the capacity table, not from a fraction of the labeled set.
- Unlabeled restaurants in a ranking's top C need a stated treatment. Options include counting them as misses
  (a lower bound), reporting a lower and upper bound, and reweighting under an ignorability assumption.
  The choice depends on the bound width (F) and on the selection strength (B).
- Weighted estimates (J) need overlap and no unobserved driver of both selection and outcome. ACHD's own schedule
  appears to depend on past high-risk results (Priority Code H), which confounds any unweighted conditional metric
  but is observable, so the weighting assumption is more plausible if the priority code and inspection history are
  in x. State which assumption each reported number rests on.
- Auxiliary outcomes (K) are the only checks that cover the whole candidate set. Their own reporting bias
  (complaints come from customers) must be stated next to the result.
- Any conditional metric (recall among inspected restaurants) is at most a secondary number, labeled as conditional.

Horizon
- A longer horizon raises coverage (A) and narrows the bound (F), but a later inspection is a weaker stand-in for
  the restaurant's state at the scoring date (C). The design must state the trade-off it made.
- Overlapping horizons let one inspection label several scoring dates. Uncertainty estimates must account for it (I).

Label
- The threshold depends on stability (C), base rate (D), and what ACHD treats as serious. The construct must be
  the same across the two records systems or the new-system year cannot serve as a test.
- ACHD's own severity signals, the consumer alert and closed placards and the priority code, are alternatives to a
  count threshold. They carry ACHD's definition of serious, at the cost of depending on ACHD's follow-up process (K).
- The label uses routine inspections only. Complaint-driven inspections are features and part of the selection mechanism.

Comparator
- Current practice is ACHD's realized inspection set and its observed yield. A rule proxy for current practice
  needs evidence that ACHD follows it. The Priority Code, if parsed, is the closest thing to ACHD's actual schedule.
- Every rule and model must be tuned on the same development scoring dates with the same procedure. The smoothing
  constants of the primary rule were searched and the variants' constants were not.

Candidates
- Permit status, the look-back window, and closures (E) change both the denominator of capacity and the pool a
  ranking draws from. Restaurants with no routine history need a stated placement, since the prior mean alone decides it (G).
  The first routine inspection of new restaurants (L) is a measured alternative to a prior set by hand.

Split and targets
- The number of development scoring dates, the gap before the test dates, and whether the new-system year is
  usable as a test (D, Part 1 Step 7) are design choices. State them before reading test outcomes.
- A success target is a value judgment about yield, stated before the test is read. The minimum detectable
  difference (I) is a property of the test set and is reported next to the target, not used as the target.

What this evaluation cannot show
- No retrospective design measures P(violation) for restaurants nobody inspected. The random audit in @doc.txt
  Q3 is the only unbiased field check and stays in the proposal.
- We do not know how ACHD would use the list. The design must not encode assumptions about ordering, timing, or
  partial adoption. Report the yield at C and leave the use of the list to ACHD.

Reporting
- Report yield by municipality and category where the counts allow it.
- Pin data snapshots, record seeds, and keep tests for the cutoff and label logic so a result can be rerun.

Related work to read before the design is fixed
- Lakkaraju et al. 2017 on selective labels and the contraction method (see Sources).
- Off-policy evaluation with inverse propensity and doubly robust estimators, which is the formal version of J.
- From memory and unverified: Chicago's food inspection forecasting project and the Boston restaurant inspection
  tournament with Yelp data (Glaeser and coauthors). Both faced non-random inspection and evaluated in the field.

## Sources

- Lakkaraju, Kleinberg, Leskovec, Ludwig, Mullainathan. The Selective Labels Problem. KDD 2017.
  https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf
- ACHD, About Food Safety Inspections (placards, inspection frequency).
  https://www.alleghenycounty.us/Services/Health-Department/Food-Safety/Alerts-Closures-and-Enforcements/About-Food-Safety-Inspections
- ACHD restaurant finder with priority code search. https://eapps.alleghenycounty.us/Restaurant/RestaurantSearch.aspx
- Pitt News report describing Priority Code H (undated, verify).
  https://pittnews.com/article/121690/top-stories/thai-hana-temporarily-closed-violations-resolved/
- NYC OATH information sheet, every establishment inspected at least once a year.
  https://www.nyc.gov/assets/oath/downloads/pdf/small-business-dohmh-en-ada.pdf
- NYC Comptroller audit of DOHMH restaurant inspections (2015).
  https://comptroller.nyc.gov/reports/audit-report-on-the-new-york-city-department-of-health-and-mental-hygienes-follow-up-on-health-code-violations-at-restaurants

# TA Feedback
- For ML baseline in the future, consider Naive Bayes or GMM. Or Grad. Boosting, Random Forest, Regression, etc.