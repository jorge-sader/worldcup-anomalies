"""Tests for the general-context reports (integration — uses cached data)."""

from worldcup_anomalies.fetch import load_data
from worldcup_anomalies.intl_elo import annotate_world_cup, build_intl_elo, load_intl_results
from worldcup_anomalies.reports import (
    all_teams,
    champions_paths,
    team_history,
    team_path,
    tournament_overview,
)


def test_tournament_overview_shape_and_champions():
    d = load_data()
    ov = tournament_overview(d)
    assert len(ov) == 22
    assert list(ov.columns) == [
        "year", "host", "teams", "games", "penalty_shootouts", "yellow", "red",
        "fourth", "third", "runner_up", "champion",
    ]
    # 2022: Argentina champion, France runner-up, 64 games, 32 teams.
    r2022 = ov[ov.year == 2022].iloc[0]
    assert r2022["champion"] == "Argentina"
    assert r2022["runner_up"] == "France"
    assert r2022["games"] == 64 and r2022["teams"] == 32
    # No card data before 1970.
    assert ov[ov.year < 1970]["yellow"].sum() == 0


def test_team_path_order_and_opponents():
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    path = team_path(d, "WC-2022", "Argentina", em)
    assert list(path["game_no"]) == [1, 2, 3, 4, 5, 6, 7]
    assert path["opponent"].iloc[0] == "Saudi Arabia"   # famous opener
    assert path["opponent"].iloc[-1] == "France"        # the final
    assert path["stage_short"].iloc[-1] == "FIN"
    assert path["opponent_elo"].notna().all()


def test_all_teams_sorted_and_complete():
    d = load_data()
    teams = all_teams(d)
    assert teams == sorted(teams)          # alphabetical for the dropdown
    assert "Argentina" in teams and "Brazil" in teams
    assert len(teams) > 70


def test_team_history_covers_all_appearances():
    d = load_data()
    em = annotate_world_cup(d.matches, build_intl_elo(load_intl_results()))
    hist = team_history(d, "Brazil", em)
    # Brazil has played every men's World Cup (22).
    assert len(hist) == 22
    years = [int(y) for y, _ in hist]
    assert years == sorted(years)          # oldest first
    # Each path carries the era-fair opponent-strength percentile used for colour.
    for _, path in hist:
        assert path["opponent_strength_pct"].between(0, 100).all()


def test_champions_paths_matrix():
    d = load_data()
    cp = champions_paths(d, [2018, 2022])
    assert cp.shape[0] == 2
    # Argentina's last game in 2022 was the final vs France.
    arg = cp.loc[(2022, "Argentina")].dropna()
    assert "France" in arg.iloc[-1]
