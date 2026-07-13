"""Unit tests for the Elo strength engine (pure / synthetic — no network)."""

import numpy as np
import pandas as pd

from worldcup_anomalies.elo import (
    _expected,
    _gd_multiplier,
    compute_elo,
    pre_tournament_ratings,
)


def test_expected_is_symmetric_and_bounded():
    assert _expected(1500, 1500) == 0.5
    assert _expected(1900, 1500) > 0.5
    assert _expected(1100, 1500) < 0.5
    # Expectations for the two sides sum to 1.
    assert np.isclose(_expected(1700, 1500) + _expected(1500, 1700), 1.0)


def test_gd_multiplier_matches_world_football_elo():
    assert _gd_multiplier(0) == 1.0
    assert _gd_multiplier(1) == 1.0
    assert _gd_multiplier(2) == 1.5
    assert _gd_multiplier(3) == (11 + 3) / 8
    assert _gd_multiplier(-4) == (11 + 4) / 8  # magnitude only


def _toy_matches():
    return pd.DataFrame({
        "tournament_id": ["T1", "T1"],
        "match_id": ["m1", "m2"],
        "match_date": pd.to_datetime(["2000-01-01", "2000-01-05"]),
        "stage_name": ["group stage", "group stage"],
        "country_name": ["Neutral", "Neutral"],
        "home_team_id": ["A", "A"],
        "home_team_name": ["A", "A"],
        "away_team_id": ["B", "C"],
        "away_team_name": ["B", "C"],
        "home_team_score": [3, 0],
        "away_team_score": [0, 0],
    })


def test_compute_elo_is_zero_sum_per_match():
    em = compute_elo(_toy_matches())
    for r in em.itertuples():
        # Points gained by home equal points lost by away.
        home_delta = r.elo_home_post - r.elo_home_pre
        away_delta = r.elo_away_post - r.elo_away_pre
        assert np.isclose(home_delta, -away_delta)


def test_winner_gains_rating():
    em = compute_elo(_toy_matches())
    first = em.iloc[0]
    assert first["elo_home_post"] > first["elo_home_pre"]  # A won 3-0
    assert first["elo_away_post"] < first["elo_away_pre"]


def test_ratings_carry_forward():
    em = compute_elo(_toy_matches())
    # A's pre-rating in match 2 equals its post-rating from match 1.
    a_m1_post = em.iloc[0]["elo_home_post"]
    a_m2_pre = em.iloc[1]["elo_home_pre"]
    assert np.isclose(a_m1_post, a_m2_pre)


def test_pre_tournament_ratings_one_row_per_team():
    em = compute_elo(_toy_matches())
    pt = pre_tournament_ratings(em)
    # Teams A, B, C each appear once for tournament T1.
    assert set(pt["team_id"]) == {"A", "B", "C"}
    assert (pt["elo_pre"] == 1500.0).all()  # everyone starts at base entering T1
