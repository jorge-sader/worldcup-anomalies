# World Cup Statistical Irregularities

[![View on GitHub](https://img.shields.io/badge/notebook-view%20on%20GitHub-24292e?logo=github&logoColor=white)](https://github.com/jorge-sader/worldcup-anomalies/blob/main/notebooks/world_cup_irregularities.ipynb)
[![Open in Colab](https://img.shields.io/badge/notebook-open%20in%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/jorge-sader/worldcup-anomalies/blob/main/notebooks/world_cup_irregularities.ipynb)
[![View on nbviewer](https://img.shields.io/badge/notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/jorge-sader/worldcup-anomalies/blob/main/notebooks/world_cup_irregularities.ipynb)

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
5. **Draw luck vs manipulation** — group-draw softness (seed-controlled), plus a fair-draw
   Monte-Carlo that separates a *lucky* soft group from an *engineered* one (soft group **and**
   rivals clustered elsewhere). Only Brazil 2002 matches the full signature 1954–2022.
6. **FIFA-leadership correlation** — anomaly rates by FIFA-president era (exploratory, caveated).

## Tournaments at a glance

Every men's World Cup — hosts, field size, games, penalty shootouts, cards, and the top-four
finishers (`worldcup_anomalies.reports.tournament_overview`). Card columns are **0 before 1970**:
booking data does not exist that far back. "Shootouts" counts games decided on penalties.

| Year | Host | Teams | Games | Shootouts | Yellow | Red | 4th | 3rd | 2nd | 1st |
|---|---|---|---|---|---|---|---|---|---|---|
| 1930 | Uruguay | 13 | 18 | 0 | 0 | 0 | Yugoslavia | United States | Argentina | Uruguay |
| 1934 | Italy | 16 | 17 | 0 | 0 | 0 | Austria | Germany | Czechoslovakia | Italy |
| 1938 | France | 15 | 18 | 0 | 0 | 0 | Sweden | Brazil | Hungary | Italy |
| 1950 | Brazil | 13 | 22 | 0 | 0 | 0 | Spain | Sweden | Brazil | Uruguay |
| 1954 | Switzerland | 16 | 26 | 0 | 0 | 0 | Uruguay | Austria | Hungary | West Germany |
| 1958 | Sweden | 16 | 35 | 0 | 0 | 0 | West Germany | France | Sweden | Brazil |
| 1962 | Chile | 16 | 32 | 0 | 0 | 0 | Yugoslavia | Chile | Czechoslovakia | Brazil |
| 1966 | England | 16 | 32 | 0 | 0 | 0 | Soviet Union | Portugal | West Germany | England |
| 1970 | Mexico | 16 | 32 | 0 | 52 | 0 | Uruguay | West Germany | Italy | Brazil |
| 1974 | West Germany | 16 | 38 | 0 | 87 | 5 | Brazil | Poland | Netherlands | West Germany |
| 1978 | Argentina | 16 | 38 | 0 | 46 | 3 | Italy | Brazil | Netherlands | Argentina |
| 1982 | Spain | 24 | 52 | 1 | 99 | 5 | France | Poland | West Germany | Italy |
| 1986 | Mexico | 24 | 52 | 3 | 138 | 8 | Belgium | France | West Germany | Argentina |
| 1990 | Italy | 24 | 52 | 4 | 169 | 16 | England | Italy | Argentina | West Germany |
| 1994 | United States | 24 | 52 | 3 | 228 | 15 | Bulgaria | Sweden | Italy | Brazil |
| 1998 | France | 32 | 64 | 3 | 254 | 22 | Netherlands | Croatia | Brazil | France |
| 2002 | Korea, Japan | 32 | 64 | 2 | 266 | 17 | South Korea | Turkey | Germany | Brazil |
| 2006 | Germany | 32 | 64 | 4 | 326 | 28 | Portugal | Germany | France | Italy |
| 2010 | South Africa | 32 | 64 | 2 | 254 | 17 | Uruguay | Germany | Netherlands | Spain |
| 2014 | Brazil | 32 | 64 | 4 | 184 | 10 | Brazil | Netherlands | Argentina | Germany |
| 2018 | Russia | 32 | 64 | 4 | 221 | 4 | England | Belgium | Croatia | France |
| 2022 | Qatar | 32 | 64 | 5 | 224 | 4 | Morocco | Croatia | France | Argentina |

The **2026** edition (48 teams, in progress) is tracked separately in the notebook (§2f) from the
live `martj42/international_results` feed — results-only, kept out of the detectors while the
tournament is unfinished and lacks referee/card data. `worldcup_anomalies.worldcup_2026`.

## Data sources

- **Primary:** [`jfjelstul/worldcup`](https://github.com/jfjelstul/worldcup) normalized CSVs
  (matches, referees, bookings, goals, standings). Filtered to men's editions. Cards exist 1970+.
- **Team strength:** an **Elo** rating. Two flavours share one engine:
  - `intl_elo.py` (default) builds Elo over **every international match since 1872** —
    friendlies, qualifiers, continental cups, World Cups — from
    [`martj42/international_results`](https://github.com/martj42/international_results), so no team
    is a blank 1500 on debut (Qatar 2022 enters ~1790, grounded in real qualifiers).
  - `elo.py` builds a self-contained World-Cup-only Elo as a cross-check.
- **FIFA leadership:** Wikipedia "List of presidents of FIFA" (with a hardcoded fallback table).

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
