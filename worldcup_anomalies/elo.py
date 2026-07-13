"""Chronological Elo strength engine over men's World Cup matches.

There is no reliable open historical Elo dataset, so we compute one in-repo. Ratings carry
forward across tournaments (a team keeps the rating it earned), start at ``BASE_RATING``, and
update after every match with the World-Football-Elo goal-difference multiplier. The host
nation gets a home-field bonus only for matches played in its own country.

Outputs:

- :func:`compute_elo` returns the matches frame augmented with pre-match ratings and the
  Elo win probability for each side (``elo_home_pre``, ``elo_away_pre``, ``elo_prob_home``).
- :func:`pre_tournament_ratings` gives each team's rating at the start of each tournament,
  which the path / strength-of-schedule analysis relies on.
"""

from __future__ import annotations

import pandas as pd

BASE_RATING = 1500.0
K_FACTOR = 40.0
HOME_ADVANTAGE = 65.0  # Elo points, applied to the host nation at home only.


def _gd_multiplier(goal_diff: int) -> float:
    """World Football Elo goal-difference weight."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def _expected(rating_a: float, rating_b: float) -> float:
    """Expected score (win prob + half draw prob) for A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    *,
    base_rating: float = BASE_RATING,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
) -> pd.DataFrame:
    """Return ``matches`` (sorted by date) with pre-match Elo columns added.

    Added columns:
        ``elo_home_pre``, ``elo_away_pre`` : ratings before the match
        ``elo_prob_home``                  : host-adjusted expected score for the home side
        ``elo_home_post``, ``elo_away_post``: ratings after the match
    """
    df = matches.sort_values(["match_date", "match_id"]).reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    home_pre, away_pre, prob_home, home_post, away_post = [], [], [], [], []

    for row in df.itertuples():
        h, a = row.home_team_id, row.away_team_id
        rh = ratings.get(h, base_rating)
        ra = ratings.get(a, base_rating)

        # Home-field advantage only when the "home" team is the host playing in its country.
        host_h = home_advantage if row.country_name == row.home_team_name else 0.0
        host_a = home_advantage if row.country_name == row.away_team_name else 0.0

        exp_h = _expected(rh + host_h, ra + host_a)

        hg, ag = row.home_team_score, row.away_team_score
        if pd.isna(hg) or pd.isna(ag):
            # No score (e.g. walkover): leave ratings unchanged.
            home_pre.append(rh); away_pre.append(ra); prob_home.append(exp_h)
            home_post.append(rh); away_post.append(ra)
            continue

        if hg > ag:
            score_h = 1.0
        elif hg < ag:
            score_h = 0.0
        else:
            score_h = 0.5

        mult = _gd_multiplier(hg - ag)
        delta = k_factor * mult * (score_h - exp_h)
        rh_new, ra_new = rh + delta, ra - delta
        ratings[h], ratings[a] = rh_new, ra_new

        home_pre.append(rh); away_pre.append(ra); prob_home.append(exp_h)
        home_post.append(rh_new); away_post.append(ra_new)

    df["elo_home_pre"] = home_pre
    df["elo_away_pre"] = away_pre
    df["elo_prob_home"] = prob_home
    df["elo_home_post"] = home_post
    df["elo_away_post"] = away_post
    return df


def pre_tournament_ratings(elo_matches: pd.DataFrame) -> pd.DataFrame:
    """Each team's Elo rating at the start of each tournament it appeared in.

    Uses the pre-match rating of the team's first match in that tournament. Returns columns
    ``tournament_id, year, team_id, team_name, elo_pre``.
    """
    long = _to_team_long(elo_matches)
    long = long.sort_values(["tournament_id", "team_id", "match_date"])
    first = long.groupby(["tournament_id", "team_id"], as_index=False).first()
    year = (
        elo_matches.groupby("tournament_id")["match_date"].min().dt.year.rename("year")
    )
    out = first.merge(year, on="tournament_id")
    return out[["tournament_id", "year", "team_id", "team_name", "elo_pre"]]


def _to_team_long(elo_matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape match rows to one row per (team, match) with that team's pre-match Elo."""
    home = elo_matches.assign(
        team_id=elo_matches["home_team_id"],
        team_name=elo_matches["home_team_name"],
        opponent_id=elo_matches["away_team_id"],
        opponent_name=elo_matches["away_team_name"],
        elo_pre=elo_matches["elo_home_pre"],
        opp_elo_pre=elo_matches["elo_away_pre"],
    )
    away = elo_matches.assign(
        team_id=elo_matches["away_team_id"],
        team_name=elo_matches["away_team_name"],
        opponent_id=elo_matches["home_team_id"],
        opponent_name=elo_matches["home_team_name"],
        elo_pre=elo_matches["elo_away_pre"],
        opp_elo_pre=elo_matches["elo_home_pre"],
    )
    cols = [
        "tournament_id", "match_id", "match_date", "stage_name",
        "team_id", "team_name", "opponent_id", "opponent_name",
        "elo_pre", "opp_elo_pre",
    ]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def team_match_long(elo_matches: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper: one row per (team, match) with pre-match Elo for team and opponent."""
    return _to_team_long(elo_matches)
