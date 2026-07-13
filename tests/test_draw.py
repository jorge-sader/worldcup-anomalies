"""Tests for the fair-draw Monte-Carlo (integration — uses cached data)."""

from worldcup_anomalies.fetch import load_data
from worldcup_anomalies.intl_elo import annotate_world_cup, build_intl_elo, load_intl_results
from worldcup_anomalies.draw import draw_monte_carlo, engineered_draw_flags


def _draws():
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    return d, draw_monte_carlo(em, d.group_standings, d.team_appearances, n_sims=4000, seed=0)


def test_percentiles_in_range():
    _, draws = _draws()
    assert draws["draw_luck_pct"].between(0, 100).all()
    assert draws["rival_clustering_pct"].between(0, 100).all()
    # One anchor per group -> at least a few hundred rows across editions.
    assert len(draws) > 100


def test_reproducible_with_seed():
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    a = draw_monte_carlo(em, d.group_standings, d.team_appearances, n_sims=3000, seed=7)
    b = draw_monte_carlo(em, d.group_standings, d.team_appearances, n_sims=3000, seed=7)
    assert a["draw_luck_pct"].equals(b["draw_luck_pct"])


def test_2002_argentina_group_of_death_is_hard_not_soft():
    # Argentina's 2002 group (with England, Sweden, Nigeria) was famously brutal: its
    # draw_luck should be HIGH (hard), not low — a sanity check on the metric's sign.
    _, draws = _draws()
    arg02 = draws[(draws.year == 2002) & (draws.team_name == "Argentina")]
    assert len(arg02) == 1
    assert arg02["draw_luck_pct"].iloc[0] > 80


def test_engineered_signature_flags_brazil_2002():
    # Brazil 2002 is the one edition matching soft-group + clustered-rivals + deep run.
    _, draws = _draws()
    flags = engineered_draw_flags(draws)
    assert ((flags.year == 2002) & (flags.team_name == "Brazil")).any()
