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


def test_phase_split_columns_and_bounds():
    from worldcup_anomalies.paths import phase_strength_split
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    sp = phase_strength_split(em, d.team_appearances)
    for col in ["group_softness_pct", "ko_toughness_pct", "split_index"]:
        assert sp[col].between(0, 100).all()
    # Argentina's 2022 group was among the softest of its edition.
    arg22 = sp[(sp.year == 2022) & (sp.team_name == "Argentina")]
    assert len(arg22) == 1
    assert arg22["group_softness_pct"].iloc[0] >= 75


def test_group_draw_difficulty_flags_argentina_soft_groups():
    from worldcup_anomalies.paths import group_draw_difficulty
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    gd = group_draw_difficulty(em, d.team_appearances)
    arg = gd[gd["team_name"] == "Argentina"].set_index("year")
    # 2014 was the softest seeded group of its edition; 2022 the 2nd-softest.
    assert arg.loc[2014, "seed_soft_rank"] == 1
    assert arg.loc[2022, "seed_soft_rank"] == 2
    # Every edition has ~8 seeds and a well-defined softest group.
    assert (gd["n_seeds_total"] > 0).all()
