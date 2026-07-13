"""Result / score anomaly detectors.

Two complementary views:

1. :func:`score_surprise` — fit a Poisson goals model driven by the Elo gap, then score every
   match by how improbable its exact scoreline was (Dixon-Coles low-score correction). This
   surfaces the biggest upsets and unexpected blowouts.

2. :func:`detect_convenient_results` — a structural detector for "mutually convenient" group
   results: the last-played match of a group, kicking off *after* a now-eliminated third team
   had finished, whose result let *both* participants advance. This is the shape of the 1982
   "Disgrace of Gijón" (West Germany 1-0 Austria) that led FIFA to make final group matches
   kick off simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

from .elo import team_match_long


# --------------------------------------------------------------------------------------
# Poisson goals model + per-match surprise
# --------------------------------------------------------------------------------------

def _modeling_frame(elo_matches: pd.DataFrame, team_appearances: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match): goals_for, elo gap, host indicator."""
    long = team_match_long(elo_matches)
    ta = team_appearances[["match_id", "team_id", "goals_for", "country_name", "team_name"]]
    frame = long.merge(ta, on=["match_id", "team_id"], suffixes=("", "_ta"))
    frame["elo_diff"] = frame["elo_pre"] - frame["opp_elo_pre"]
    frame["is_host"] = (frame["country_name"] == frame["team_name"]).astype(int)
    return frame.dropna(subset=["goals_for", "elo_diff"])


def fit_goals_model(elo_matches: pd.DataFrame, team_appearances: pd.DataFrame):
    """Fit Poisson GLM: goals_for ~ elo_diff + is_host. Returns (results, frame)."""
    frame = _modeling_frame(elo_matches, team_appearances)
    X = sm.add_constant(frame[["elo_diff", "is_host"]].astype(float))
    model = sm.GLM(frame["goals_for"].astype(float), X, family=sm.families.Poisson())
    return model.fit(), frame


def _poisson_pmf(k: int, lam: float) -> float:
    return float(poisson.pmf(k, lam))


def _dixon_coles_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Low-score dependence correction (Dixon & Coles, 1997)."""
    if hg == 0 and ag == 0:
        return 1.0 - lam_h * lam_a * rho
    if hg == 0 and ag == 1:
        return 1.0 + lam_h * rho
    if hg == 1 and ag == 0:
        return 1.0 + lam_a * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def score_surprise(
    elo_matches: pd.DataFrame,
    team_appearances: pd.DataFrame,
    *,
    rho: float = -0.05,
) -> pd.DataFrame:
    """Per-match surprise = -log P(exact scoreline) under the fitted Poisson model.

    Returns one row per match with expected goals, the actual score, and ``surprise``
    (higher = less probable given the teams' strengths). Also flags whether the surprise
    was an upset (weaker side by Elo won).
    """
    res, _ = fit_goals_model(elo_matches, team_appearances)
    df = elo_matches.copy()

    diff = df["elo_home_pre"] - df["elo_away_pre"]
    host_h = (df["country_name"] == df["home_team_name"]).astype(float)
    host_a = (df["country_name"] == df["away_team_name"]).astype(float)
    b0, b_diff, b_host = res.params["const"], res.params["elo_diff"], res.params["is_host"]
    lam_h = np.exp(b0 + b_diff * diff + b_host * host_h)
    lam_a = np.exp(b0 + b_diff * (-diff) + b_host * host_a)

    surprises = []
    for hg, ag, lh, la in zip(
        df["home_team_score"], df["away_team_score"], lam_h, lam_a
    ):
        if pd.isna(hg) or pd.isna(ag):
            surprises.append(np.nan)
            continue
        hg, ag = int(hg), int(ag)
        p = _poisson_pmf(hg, lh) * _poisson_pmf(ag, la) * _dixon_coles_tau(hg, ag, lh, la, rho)
        p = max(p, 1e-12)
        surprises.append(-np.log(p))

    out = df[[
        "tournament_id", "match_id", "match_name", "stage_name", "match_date",
        "home_team_name", "away_team_name", "home_team_score", "away_team_score",
        "elo_home_pre", "elo_away_pre",
    ]].copy()
    out["exp_goals_home"] = lam_h.to_numpy()
    out["exp_goals_away"] = lam_a.to_numpy()
    out["surprise"] = surprises
    out["year"] = pd.to_datetime(out["match_date"]).dt.year
    # Upset = the pre-match weaker side (by Elo) won outright.
    stronger_home = out["elo_home_pre"] >= out["elo_away_pre"]
    home_win = out["home_team_score"] > out["away_team_score"]
    away_win = out["away_team_score"] > out["home_team_score"]
    out["upset"] = np.where(stronger_home, away_win, home_win)
    # Elo gap the result defied (points): larger = bigger favourite upset.
    out["elo_gap_defied"] = np.where(
        out["upset"], (out["elo_home_pre"] - out["elo_away_pre"]).abs(), 0.0
    )
    return out.dropna(subset=["surprise"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Convenient group results
# --------------------------------------------------------------------------------------

def _match_datetime(matches: pd.DataFrame) -> pd.Series:
    """Combine match_date + match_time into a sortable datetime (time optional)."""
    date = pd.to_datetime(matches["match_date"])
    time = matches["match_time"].fillna("00:00").astype(str).str.slice(0, 5)
    dt = pd.to_datetime(
        date.dt.strftime("%Y-%m-%d") + " " + time, errors="coerce"
    )
    return dt.fillna(date)


def _first_round_advancement(group_standings: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map (tournament_id, team_id) -> advanced flag for the FIRST group stage only.

    We key on team_id rather than group label because in the 1974-1982 two-group-stage
    format the ``group_name`` labels diverge between the matches and standings tables.
    """
    gs = group_standings.copy()
    first_stage = gs.groupby("tournament_id")["stage_number"].transform("min")
    gs = gs[gs["stage_number"] == first_stage]
    return {
        (r.tournament_id, r.team_id): int(r.advanced)
        for r in gs.itertuples()
    }


def detect_convenient_results(
    matches: pd.DataFrame,
    group_standings: pd.DataFrame,
) -> pd.DataFrame:
    """Flag last-played first-round group matches whose result advanced both teams.

    A row is flagged when, within a first group-stage group:
      * it is the chronologically last match of the group,
      * it kicked off *after* at least one non-advancing team had already completed all its
        matches (informational advantage), and
      * *both* participants advanced.

    ``minimality`` (higher = more suspicious) rewards low-event, minimum-margin results — the
    kind that look arranged rather than contested. This is the shape of the 1982 Gijón match.
    """
    # First group stage only: exclude the 1974-82 "second group stage".
    m = matches[
        (matches["group_stage"] == 1) & (matches["stage_name"] != "second group stage")
    ].copy()
    if m.empty:
        return _empty_convenient()
    m["dt"] = _match_datetime(m)

    advanced_map = _first_round_advancement(group_standings)

    flagged = []
    for (tid, group), gm in m.groupby(["tournament_id", "group_name"]):
        gm = gm.sort_values("dt")
        gm["day"] = gm["dt"].dt.normalize()
        last = gm.iloc[-1]
        last_day = last["day"]
        others = gm[gm["dt"] < last["dt"]]
        if others.empty:
            continue

        adv = lambda t: advanced_map.get((tid, t), 0)
        both_advanced = adv(last["home_team_id"]) == 1 and adv(last["away_team_id"]) == 1
        if not both_advanced:
            continue

        # Last match DAY per team in this group (day granularity avoids time-zone / local
        # kickoff-time noise: since 1986 the final pair kicks off simultaneously same-day).
        team_last_day: dict[str, pd.Timestamp] = {}
        team_names: dict[str, str] = {}
        for r in gm.itertuples():
            team_last_day[r.home_team_id] = max(team_last_day.get(r.home_team_id, r.day), r.day)
            team_last_day[r.away_team_id] = max(team_last_day.get(r.away_team_id, r.day), r.day)
            team_names[r.home_team_id] = r.home_team_name
            team_names[r.away_team_id] = r.away_team_name

        # A non-advancing team that finished on an EARLIER day than this match.
        victim = [
            t for t in team_last_day
            if adv(t) == 0 and team_last_day[t] < last_day
        ]
        if not victim:
            continue

        # Non-simultaneous = the group's other final-round match was on an earlier day, so
        # this match was played with full knowledge of the result it needed.
        prev_day = others["day"].max()
        non_simultaneous = last_day > prev_day

        hg, ag = last["home_team_score"], last["away_team_score"]
        total_goals = (hg + ag) if pd.notna(hg) and pd.notna(ag) else np.nan
        margin = abs(hg - ag) if pd.notna(hg) and pd.notna(ag) else np.nan
        minimality = 1.0 / (1.0 + (0 if pd.isna(total_goals) else total_goals))
        if not pd.isna(margin) and margin == 1:
            minimality += 0.25

        victim_names = [team_names[t] for t in victim]
        flagged.append({
            "tournament_id": tid,
            "group_name": group,
            "match_id": last["match_id"],
            "match_name": last["match_name"],
            "match_date": last["match_date"],
            "score": f"{int(hg)}-{int(ag)}" if pd.notna(hg) else "",
            "both_advanced": both_advanced,
            "non_simultaneous": bool(non_simultaneous),
            "eliminated_finished_earlier": ", ".join(map(str, victim_names)),
            "total_goals": total_goals,
            "margin": margin,
            "minimality": minimality,
        })

    if not flagged:
        return _empty_convenient()
    out = pd.DataFrame(flagged)
    # Non-simultaneous, low-event results rank highest.
    out["suspicion"] = out["minimality"] + out["non_simultaneous"].astype(float) * 0.75
    return out.sort_values("suspicion", ascending=False).reset_index(drop=True)


def _empty_convenient() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "tournament_id", "group_name", "match_id", "match_name", "match_date", "score",
        "both_advanced", "non_simultaneous", "eliminated_finished_earlier",
        "total_goals", "margin", "minimality", "suspicion",
    ])
