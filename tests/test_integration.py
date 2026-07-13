"""Integration tests over the real cached dataset.

These read from ``data/raw/`` (populated by ``python -m worldcup_anomalies.fetch``). If the
cache is absent they will fetch once; run the fetch step first for a fully offline test run.
"""

import pandas as pd

from worldcup_anomalies.fetch import load_data, president_at
from worldcup_anomalies.elo import compute_elo
from worldcup_anomalies.models import detect_convenient_results


def test_data_is_mens_only_and_complete():
    d = load_data()
    # Exactly the 22 men's editions, no women's leakage.
    assert d.tournaments["tournament_id"].nunique() == 22
    assert not d.tournaments["tournament_name"].str.contains("Women").any()
    assert d.tournaments["year"].min() == 1930
    assert d.tournaments["year"].max() == 2022
    assert len(d.matches) > 900


def test_president_mapping():
    # 2014 World Cup fell under Sepp Blatter; 1970 under Stanley Rous.
    assert president_at(pd.Timestamp("2014-06-15")) == "Sepp Blatter"
    assert president_at(pd.Timestamp("1970-06-01")) == "Stanley Rous"
    assert president_at(pd.Timestamp("2022-11-20")) == "Gianni Infantino"


def test_gijon_1982_is_flagged_convenient():
    d = load_data()
    cr = detect_convenient_results(d.matches, d.group_standings)
    # The West Germany 1-0 Austria "Disgrace of Gijón" must be flagged.
    assert cr["match_name"].str.contains("West Germany vs Austria").any()
    gijon = cr[cr["match_name"].str.contains("West Germany vs Austria")].iloc[0]
    assert gijon["non_simultaneous"]
    assert "Algeria" in gijon["eliminated_finished_earlier"]


def test_elo_covers_all_matches():
    d = load_data()
    em = compute_elo(d.matches)
    assert len(em) == len(d.matches)
    assert em["elo_home_pre"].notna().all()
