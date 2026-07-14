"""Tests for the in-progress 2026 tracker (uses cached martj42 results)."""

from worldcup_anomalies.intl_elo import build_intl_elo, load_intl_results
from worldcup_anomalies.worldcup_2026 import (
    draw_luck_2026,
    group_membership_2026,
    load_2026,
    pre_tournament_strength_2026,
    quarterfinalists_2026,
    summary_2026,
    team_path_2026,
)


def test_load_2026_stages_inferred():
    m = load_2026()
    counts = m["stage"].value_counts()
    # Fixed 48-team bracket: 72 group matches, then 16 R32, 8 R16, 4 QF (as of the cache).
    assert counts.get("group stage", 0) == 72
    assert counts.get("round of 32", 0) == 16
    assert m["home_score"].notna().all()  # results-only
    assert m["game_index"].is_monotonic_increasing


def test_summary_and_quarterfinalists():
    m = load_2026()
    s = summary_2026(m)
    assert s["teams"] == 48
    assert s["games_total"] == 104
    qf = quarterfinalists_2026(m)
    assert len(qf) == 8
    # Argentina reached at least the quarter-finals in 2026.
    assert "Argentina" in qf


def test_team_path_schema_matches_plotter():
    m = load_2026()
    pct = pre_tournament_strength_2026(build_intl_elo(load_intl_results()), m)
    path = team_path_2026(m, "Argentina", pct)
    for col in ["game_no", "opponent", "result", "stage_short", "opponent_strength_pct"]:
        assert col in path.columns
    assert path["opponent_strength_pct"].between(0, 100).all()
    assert list(path["game_no"]) == list(range(1, len(path) + 1))


def test_draw_luck_2026_on_48_team_bracket():
    m = load_2026()
    gm = group_membership_2026(m)
    # 12 groups of exactly 4 reconstructed from the group-stage matches.
    assert gm["group"].nunique() == 12
    assert (gm.groupby("group").size() == 4).all()

    from worldcup_anomalies.intl_elo import build_intl_elo, load_intl_results
    dl = draw_luck_2026(m, build_intl_elo(load_intl_results()), n_sims=3000, seed=0)
    assert len(dl) == 12                                  # one anchor per group
    assert dl["draw_luck_pct"].between(0, 100).all()
    assert dl["rival_clustering_pct"].between(0, 100).all()
    assert (dl["rival_clustering_pct"].nunique() == 1)    # edition-level, constant
