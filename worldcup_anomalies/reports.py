"""General-context reports: a per-tournament overview and per-team knockout paths.

These are presentation helpers — they summarise the raw data for humans, and feed the context
visualisations in the notebook (a tournament reference table, path charts, and the Elo primer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fetch import WorldCupData

# Stage ordering for laying a team's games out left-to-right.
_STAGE_ORDER = {
    "group stage": 0,
    "first group stage": 0,
    "second group stage": 1,
    "round of 16": 2,
    "quarter-finals": 3,
    "final round": 4,
    "semi-finals": 5,
    "third-place match": 6,
    "final": 7,
}

# Short stage labels for display.
STAGE_SHORT = {
    "group stage": "GRP",
    "first group stage": "GRP",
    "second group stage": "GRP2",
    "round of 16": "R16",
    "quarter-finals": "QF",
    "final round": "FR",
    "semi-finals": "SF",
    "third-place match": "3rd",
    "final": "FIN",
}


def tournament_overview(data: WorldCupData) -> pd.DataFrame:
    """One row per World Cup: hosts, size, discipline totals, and the top-four finishers.

    Columns: ``year, host, teams, games, penalty_shootouts, yellow, red, fourth, third,
    runner_up, champion``. Card columns are 0 before 1970 (no card data exists that far back).
    """
    m = data.matches
    games = m.groupby("tournament_id").size().rename("games")
    shootouts = m.groupby("tournament_id")["penalty_shootout"].sum().rename("penalty_shootouts")
    teams = (
        data.team_appearances.groupby("tournament_id")["team_id"].nunique().rename("teams")
    )

    b = data.bookings
    yellow = b.groupby("tournament_id")["yellow_card"].sum().rename("yellow")
    # A red = a straight red plus a second-yellow dismissal.
    reds = (b["red_card"] + b["second_yellow_card"]).groupby(b["tournament_id"]).sum().rename("red")

    st = data.tournament_standings
    place = {
        1: "champion", 2: "runner_up", 3: "third", 4: "fourth",
    }
    top4 = (
        st[st["position"].isin(place)]
        .assign(slot=lambda x: x["position"].map(place))
        .pivot_table(index="tournament_id", columns="slot", values="team_name", aggfunc="first")
    )

    t = data.tournaments.set_index("tournament_id")
    out = t[["year", "host_country"]].rename(columns={"host_country": "host"})
    out = (
        out.join(teams).join(games).join(shootouts).join(yellow).join(reds).join(top4)
    )
    for col in ["yellow", "red"]:
        out[col] = out[col].fillna(0).astype(int)
    out["penalty_shootouts"] = out["penalty_shootouts"].fillna(0).astype(int)
    ordered = [
        "year", "host", "teams", "games", "penalty_shootouts", "yellow", "red",
        "fourth", "third", "runner_up", "champion",
    ]
    return out[ordered].sort_values("year").reset_index(drop=True)


def team_path(
    data: WorldCupData,
    tournament_id: str,
    team_name: str,
    elo_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ordered game-by-game path of one team in one tournament.

    Returns one row per game: ``game_no, stage, stage_short, opponent, goals_for,
    goals_against, result``, and (if ``elo_matches`` is given) the opponent's pre-match Elo.
    """
    ta = data.team_appearances
    rows = ta[(ta["tournament_id"] == tournament_id) & (ta["team_name"] == team_name)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["stage_ord"] = rows["stage_name"].map(_STAGE_ORDER).fillna(0)
    rows = rows.sort_values(["stage_ord", "match_date"]).reset_index(drop=True)
    rows["game_no"] = np.arange(1, len(rows) + 1)
    rows["stage_short"] = rows["stage_name"].map(STAGE_SHORT).fillna(rows["stage_name"])

    if elo_matches is not None:
        opp_elo = []
        elo_lookup = elo_matches.set_index("match_id")
        for r in rows.itertuples():
            mm = elo_lookup.loc[r.match_id] if r.match_id in elo_lookup.index else None
            if mm is None:
                opp_elo.append(np.nan)
            elif mm["home_team_id"] == r.team_id:
                opp_elo.append(mm["elo_away_pre"])
            else:
                opp_elo.append(mm["elo_home_pre"])
        rows["opponent_elo"] = opp_elo

    cols = [
        "game_no", "stage_name", "stage_short", "opponent_name",
        "goals_for", "goals_against", "result",
    ]
    if "opponent_elo" in rows:
        cols.append("opponent_elo")
    return rows[cols].rename(columns={"opponent_name": "opponent", "stage_name": "stage"})


def champions_paths(
    data: WorldCupData,
    years: list[int],
    elo_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Wide 'road to the title' matrix: one row per champion, one column per game number.

    Cell value is the opponent name (with the stage short-code). Useful for comparing how hard
    each winner's route was at a glance. Missing games (shorter formats) are blank.
    """
    t = data.tournaments
    rows = []
    for yr in years:
        tt = t[t["year"] == yr]
        if tt.empty:
            continue
        tid = tt["tournament_id"].iloc[0]
        champ = tt["winner"].iloc[0]
        path = team_path(data, tid, champ, elo_matches)
        entry = {"year": yr, "champion": champ}
        for r in path.itertuples():
            entry[f"G{r.game_no}"] = f"{r.opponent} [{r.stage_short}]"
        rows.append(entry)
    return pd.DataFrame(rows).set_index(["year", "champion"])
