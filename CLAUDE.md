# Restaurant inspection risk ranking (10-718, Team TELOS)

The Allegheny County Health Department cannot give every restaurant a routine inspection each year. This project
ranks candidate restaurants so that inspector time finds more serious violators. The central difficulty is
selective labels: an outcome exists only for restaurants ACHD chose to inspect, and that choice is not random, so
every evaluation must say what it assumes about the restaurants nobody inspected.

Read first: `doc.txt` (the assignment and the original design), `proposed_changes.md` (the long-term plan and the
known problems), `README.md` (layout, data sources, how to reproduce), `src/config.py` (every fixed choice).

Organization: `src/` is one package run from the repository root (`python -m src.<module>`). `src/data/` downloads
the raw public tables; `src/features/` computes point-in-time features; `src/panel.py` joins them into a panel with a
row per (candidate, scoring date); `src/models/` and `src/analysis/` read that panel and write JSON under
`results/`; `tests/` checks the cutoff and label logic on synthetic rows. Scripts are durable code: docstrings, type hints, explicit parameters.
