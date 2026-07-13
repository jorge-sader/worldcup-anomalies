# World Cup Statistical Irregularities

[![View the notebook](https://img.shields.io/badge/notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/jorge-sader/worldcup-anomalies/blob/main/notebooks/world_cup_irregularities.ipynb)

Auto-fetches all **men's** FIFA World Cup data (1930–2022) — matches, scores, phases, referees,
bookings — plus a FIFA-leadership table, and runs statistical models that surface **irregularities
worth looking into**. The deliverable is the notebook `notebooks/world_cup_irregularities.ipynb`;
the logic lives in the importable, testable package `worldcup_anomalies/`.

## What it flags

1. **Referee anomalies** — cards / penalties / added-time per referee vs. a modeled baseline (1970+).
2. **Result / score anomalies** — scorelines vs. Poisson/Dixon–Coles expectation; results that
   conveniently satisfy qualification math (e.g. the 1982 "Disgrace of Gijón").
3. **Host & bracket effects** — host overperformance vs. strength expectation.
4. **Per-team "easy path"** — any team reaching the QF/SF/Final with an unusually weak
   strength-of-schedule (few established powers beaten).
5. **FIFA-leadership correlation** — anomaly rates by FIFA-president era (exploratory, caveated).

## Data sources

- **Primary:** [`jfjelstul/worldcup`](https://github.com/jfjelstul/worldcup) normalized CSVs
  (matches, referees, bookings, goals, standings). Filtered to men's editions. Cards exist 1970+.
- **FIFA leadership:** Wikipedia "List of presidents of FIFA" (with a hardcoded fallback table).
- **Team strength:** an Elo engine computed in-repo over the match data (no external Elo dependency).

## Usage

```bash
uv run python -m worldcup_anomalies.fetch          # download + cache raw data to data/raw/
uv run pytest                                       # unit tests (Elo + anomaly scoring)
uv run jupyter nbconvert --to notebook --execute \
    notebooks/world_cup_irregularities.ipynb        # run the full analysis end-to-end
```

## Caveats

The FIFA-leadership "corruption lens" is **exploratory and correlational only** — the number of
tournaments is small, so era-level differences are suggestive, never conclusive. All detectors
report a standardized anomaly score with an explicit multiple-testing caveat.
