"""Tests for the full-history international Elo (integration — uses cached data)."""

import numpy as np
import pandas as pd

from worldcup_anomalies.fetch import load_data
from worldcup_anomalies.intl_elo import (
    annotate_world_cup,
    build_intl_elo,
    jf_to_canonical,
    load_intl_results,
)
from worldcup_anomalies.elo import pre_tournament_ratings


def test_name_mapping():
    assert jf_to_canonical("West Germany") == "Germany"
    assert jf_to_canonical("Soviet Union") == "Russia"
    assert jf_to_canonical("Serbia and Montenegro") == "Serbia"
    assert jf_to_canonical("Brazil") == "Brazil"  # identity for the common case


def test_annotation_schema_matches_compute_elo():
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    for col in ["elo_home_pre", "elo_away_pre", "elo_prob_home",
                "elo_home_post", "elo_away_post"]:
        assert col in em.columns
    assert len(em) == len(d.matches)
    assert em["elo_home_pre"].notna().all()
    assert em["elo_away_pre"].notna().all()


def test_no_world_cup_team_starts_at_neutral_1500():
    # The whole point of the upgrade: qualifiers/friendlies mean nobody is a blank 1500.
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    pre = pre_tournament_ratings(em)
    at_base = pre[pre["elo_pre"].round(1) == 1500.0]
    assert len(at_base) == 0


def test_probabilities_are_valid():
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    assert ((em["elo_prob_home"] > 0) & (em["elo_prob_home"] < 1)).all()
