"""Data layer: download, cache, and tidy men's World Cup data.

Primary source is the ``jfjelstul/worldcup`` normalized CSV dataset. Everything is cached
under ``data/raw/`` so reruns are offline and reproducible. The dataset also contains the
women's tournaments, so the very first thing we do after loading is filter to the 22 men's
editions (1930-2022).

FIFA leadership (presidents + term ranges) is a small curated table committed in-code (the
``FIFA_PRESIDENTS`` fallback); we optionally try to enrich/verify it from Wikipedia, but the
pipeline never depends on the network for it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/"

# Tables we pull from jfjelstul/worldcup.
TABLES = (
    "tournaments",
    "tournament_stages",
    "matches",
    "goals",
    "penalty_kicks",
    "bookings",
    "referees",
    "referee_appearances",
    "team_appearances",
    "group_standings",
    "host_countries",
    "teams",
)

# Tables that carry a tournament_name column and therefore get the men's filter applied.
_EVENT_TABLES = {
    "tournaments",
    "tournament_stages",
    "matches",
    "goals",
    "penalty_kicks",
    "bookings",
    "referee_appearances",
    "team_appearances",
    "group_standings",
    "host_countries",
}

_HTTP_TIMEOUT = 30


# --------------------------------------------------------------------------------------
# Download + cache
# --------------------------------------------------------------------------------------

def download_table(name: str, *, refresh: bool = False) -> Path:
    """Download one CSV table to ``data/raw/`` (skips if cached unless ``refresh``)."""
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}; known: {', '.join(TABLES)}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{name}.csv"
    if dest.exists() and not refresh:
        return dest
    resp = requests.get(f"{BASE_URL}{name}.csv", timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_raw(name: str, *, refresh: bool = False) -> pd.DataFrame:
    """Load a single raw table (downloading + caching on demand), unfiltered."""
    return pd.read_csv(download_table(name, refresh=refresh))


def _is_mens(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only men's rows. The dataset names men's editions '... Men's World Cup'."""
    if "tournament_name" not in df.columns:
        return df
    return df[df["tournament_name"].str.contains("Men's", na=False)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# FIFA leadership (curated, with optional Wikipedia enrichment)
# --------------------------------------------------------------------------------------

# Presidents whose terms overlap the men's World Cup era (1930-2022). Dates are term
# start/end; the incumbent has an open end (NaT). Source: Wikipedia "List of presidents
# of FIFA" (kept in-code so the pipeline is reproducible offline).
FIFA_PRESIDENTS = [
    ("Jules Rimet", "France", "1921-03-01", "1954-06-21"),
    ("Rodolphe Seeldrayers", "Belgium", "1954-06-21", "1955-10-07"),
    ("Arthur Drewry", "England", "1955-10-07", "1961-03-25"),
    ("Stanley Rous", "England", "1961-09-28", "1974-06-11"),
    ("João Havelange", "Brazil", "1974-06-11", "1998-06-08"),
    ("Sepp Blatter", "Switzerland", "1998-06-08", "2016-02-26"),
    ("Gianni Infantino", "Switzerland/Italy", "2016-02-26", None),
]

WIKI_PRESIDENTS_URL = (
    "https://en.wikipedia.org/w/index.php?title=List_of_presidents_of_FIFA&action=raw"
)


def load_leadership(*, try_web: bool = False) -> pd.DataFrame:
    """Return FIFA presidents with parsed term ranges.

    The curated ``FIFA_PRESIDENTS`` table is authoritative. ``try_web=True`` attempts a
    best-effort fetch of the Wikipedia page purely so the caller can confirm nothing has
    drifted; parse failures fall back silently to the curated table.
    """
    df = pd.DataFrame(
        FIFA_PRESIDENTS, columns=["president", "country", "took_office", "left_office"]
    )
    df["took_office"] = pd.to_datetime(df["took_office"])
    df["left_office"] = pd.to_datetime(df["left_office"])  # NaT for the incumbent
    df["incumbent"] = df["left_office"].isna()

    if try_web:
        try:
            resp = requests.get(WIKI_PRESIDENTS_URL, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            df.attrs["wikipedia_bytes"] = len(resp.content)
        except Exception as exc:  # network/parse issues never break the pipeline
            df.attrs["wikipedia_error"] = repr(exc)
    return df


def president_at(date: pd.Timestamp, leadership: pd.DataFrame | None = None) -> str:
    """Name of the FIFA president in office on ``date`` (empty string if none)."""
    if leadership is None:
        leadership = load_leadership()
    date = pd.Timestamp(date)
    for row in leadership.itertuples():
        end = row.left_office if pd.notna(row.left_office) else pd.Timestamp.max
        if row.took_office <= date <= end:
            return row.president
    return ""


# --------------------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------------------

@dataclass
class WorldCupData:
    """Tidy, men's-only World Cup tables plus FIFA leadership."""

    tournaments: pd.DataFrame
    tournament_stages: pd.DataFrame
    matches: pd.DataFrame
    goals: pd.DataFrame
    penalty_kicks: pd.DataFrame
    bookings: pd.DataFrame
    referees: pd.DataFrame
    referee_appearances: pd.DataFrame
    team_appearances: pd.DataFrame
    group_standings: pd.DataFrame
    host_countries: pd.DataFrame
    teams: pd.DataFrame
    leadership: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Row counts per table (for a quick sanity check)."""
        rows = [
            (f.name, len(getattr(self, f.name)))
            for f in fields(self)
        ]
        return pd.DataFrame(rows, columns=["table", "rows"])


def load_data(*, refresh: bool = False, try_web_leadership: bool = False) -> WorldCupData:
    """Download (or read cached), filter to men's, and return the full bundle."""
    raw = {name: load_raw(name, refresh=refresh) for name in TABLES}

    # Men's filter on every event table.
    for name in _EVENT_TABLES:
        raw[name] = _is_mens(raw[name])

    # Master tables (referees, teams) keep every row; we restrict referees to those who
    # actually officiated a men's match so downstream joins stay men's-only.
    mens_ref_ids = set(raw["referee_appearances"]["referee_id"].unique())
    raw["referees"] = raw["referees"][
        raw["referees"]["referee_id"].isin(mens_ref_ids)
    ].reset_index(drop=True)

    # Attach match_date to matches as a real datetime for time-ordering / era mapping.
    raw["matches"]["match_date"] = pd.to_datetime(raw["matches"]["match_date"])

    return WorldCupData(
        tournaments=raw["tournaments"],
        tournament_stages=raw["tournament_stages"],
        matches=raw["matches"],
        goals=raw["goals"],
        penalty_kicks=raw["penalty_kicks"],
        bookings=raw["bookings"],
        referees=raw["referees"],
        referee_appearances=raw["referee_appearances"],
        team_appearances=raw["team_appearances"],
        group_standings=raw["group_standings"],
        host_countries=raw["host_countries"],
        teams=raw["teams"],
        leadership=load_leadership(try_web=try_web_leadership),
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch + cache men's World Cup data.")
    parser.add_argument("--refresh", action="store_true", help="force re-download")
    args = parser.parse_args()

    data = load_data(refresh=args.refresh)
    summary = data.summary()
    print(f"Cached raw data in {RAW_DIR}")
    print(summary.to_string(index=False))

    n_tournaments = data.tournaments["tournament_id"].nunique()
    yr_min, yr_max = data.tournaments["year"].min(), data.tournaments["year"].max()
    print(f"\nMen's tournaments: {n_tournaments} ({yr_min}-{yr_max})")
    print(f"Matches: {len(data.matches)}  |  Referees: {len(data.referees)}  "
          f"|  Bookings: {len(data.bookings)}")
    booked_years = pd.to_datetime(data.bookings["match_date"], errors="coerce").dt.year
    if len(booked_years):
        print(f"Card data spans {booked_years.min()}-{booked_years.max()} "
              "(pre-1970 cards do not exist in the source).")
    print(f"FIFA presidents in era: {len(data.leadership)}")


if __name__ == "__main__":
    _main()
