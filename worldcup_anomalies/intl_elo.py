"""Full-history international Elo — a grounded alternative to the World-Cup-only ratings.

The engine in :mod:`worldcup_anomalies.elo` only ever sees World Cup matches, so a team enters
its first tournament at a neutral 1500 no matter how strong it really is, and even established
teams update off a handful of matches every four years. That is a real weakness when the whole
question is "did this team avoid strong opponents?".

This module fixes it by building the same Elo over **every international match since 1872**
(friendlies, all qualifiers, continental cups, World Cups) from the ``martj42/international_results``
dataset, then reading off each World Cup team's rating *as it stood entering the match*. Because
qualifiers and friendlies are included, essentially no World Cup team is a blank 1500 — Qatar in
2022, Japan on debut in 1998, etc. all carry ratings earned over their real international record.

The output is deliberately schema-compatible with :func:`worldcup_anomalies.elo.compute_elo`, so
every downstream detector (upsets, easy-path, host performance) works unchanged — you just pass
these ``elo_matches`` instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .elo import BASE_RATING, HOME_ADVANTAGE, K_FACTOR, _expected, _gd_multiplier
from .fetch import RAW_DIR

INTL_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# jfjelstul team names -> martj42 team names, for the handful that differ. martj42 keeps a
# continuous lineage (West Germany's record lives under "Germany", the USSR's under "Russia",
# Serbia & Montenegro's under "Serbia"), which is exactly the continuity we want for strength.
JF_TO_MARTJ42 = {
    "West Germany": "Germany",
    "East Germany": "German DR",
    "Soviet Union": "Russia",
    "Serbia and Montenegro": "Serbia",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
}


def jf_to_canonical(name: str) -> str:
    """Map a jfjelstul team name to its martj42/canonical equivalent."""
    return JF_TO_MARTJ42.get(name, name)


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------

def load_intl_results(*, refresh: bool = False) -> pd.DataFrame:
    """Download + cache the full international-results dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "intl_results.csv"
    if not dest.exists() or refresh:
        resp = requests.get(INTL_URL, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    df = pd.read_csv(dest, parse_dates=["date"])
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Global Elo timeline
# --------------------------------------------------------------------------------------

def build_intl_elo(
    intl_results: pd.DataFrame,
    *,
    base_rating: float = BASE_RATING,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Chronological Elo over every international match.

    Returns a per-team timeline ``{team: (dates, ratings_after)}`` with the dates sorted, so a
    later lookup can ask "what was this team's rating just before date D?". Home advantage is
    applied to the home side except when ``neutral`` is True.
    """
    df = intl_results.sort_values(["date"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    hist_dates: dict[str, list] = {}
    hist_ratings: dict[str, list] = {}

    dates = df["date"].to_numpy()
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy()

    for i in range(len(df)):
        h, a = home[i], away[i]
        rh = ratings.get(h, base_rating)
        ra = ratings.get(a, base_rating)
        hfa = 0.0 if neutral[i] else home_advantage
        exp_h = _expected(rh + hfa, ra)

        gh, ga = hs[i], as_[i]
        if gh > ga:
            score_h = 1.0
        elif gh < ga:
            score_h = 0.0
        else:
            score_h = 0.5
        delta = k_factor * _gd_multiplier(gh - ga) * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

        for team in (h, a):
            hist_dates.setdefault(team, []).append(dates[i])
            hist_ratings.setdefault(team, []).append(ratings[team])

    return {
        team: (np.array(hist_dates[team]), np.array(hist_ratings[team]))
        for team in hist_dates
    }


def _rating_before(timeline: dict, team: str, date: np.datetime64, base: float) -> float:
    """Team's most recent post-match rating strictly before ``date`` (base if none)."""
    tl = timeline.get(team)
    if tl is None:
        return base
    dates, ratings = tl
    idx = np.searchsorted(dates, np.datetime64(date), side="left")
    return float(ratings[idx - 1]) if idx > 0 else base


def annotate_world_cup(
    wc_matches: pd.DataFrame,
    timeline: dict | None = None,
    *,
    base_rating: float = BASE_RATING,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Annotate World Cup matches with full-history pre-match Elo (compute_elo-compatible).

    Adds ``elo_home_pre``, ``elo_away_pre``, ``elo_prob_home``, ``elo_home_post``,
    ``elo_away_post`` — read from the global international timeline as of each match date — so
    the result is a drop-in replacement for :func:`worldcup_anomalies.elo.compute_elo`.
    """
    if timeline is None:
        timeline = build_intl_elo(
            load_intl_results(refresh=refresh),
            base_rating=base_rating, k_factor=k_factor, home_advantage=home_advantage,
        )

    df = wc_matches.sort_values(["match_date", "match_id"]).reset_index(drop=True).copy()
    home_pre, away_pre, prob_home, home_post, away_post = [], [], [], [], []

    for r in df.itertuples():
        h = jf_to_canonical(r.home_team_name)
        a = jf_to_canonical(r.away_team_name)
        d = np.datetime64(pd.Timestamp(r.match_date))
        rh = _rating_before(timeline, h, d, base_rating)
        ra = _rating_before(timeline, a, d, base_rating)

        host_h = home_advantage if r.country_name == r.home_team_name else 0.0
        host_a = home_advantage if r.country_name == r.away_team_name else 0.0
        exp_h = _expected(rh + host_h, ra + host_a)

        # Self-consistent post ratings (not used downstream, but keeps schema parity).
        if pd.notna(r.home_team_score) and pd.notna(r.away_team_score):
            hg, ag = r.home_team_score, r.away_team_score
            score_h = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
            delta = k_factor * _gd_multiplier(hg - ag) * (score_h - exp_h)
        else:
            delta = 0.0

        home_pre.append(rh); away_pre.append(ra); prob_home.append(exp_h)
        home_post.append(rh + delta); away_post.append(ra - delta)

    df["elo_home_pre"] = home_pre
    df["elo_away_pre"] = away_pre
    df["elo_prob_home"] = prob_home
    df["elo_home_post"] = home_post
    df["elo_away_post"] = away_post
    return df
