"""2026 World Cup — a results-only, IN-PROGRESS tracker.

Sourced from ``martj42/international_results`` (live-updated), **not** jfjelstul, and kept
deliberately OUT of the anomaly-detector pipeline for three reasons: the tournament is still
being played, it has no referee or card data anywhere yet, and it is the first 48-team edition
(12 groups → round of 32 → …) whose format the historical detectors don't model.

What this module provides is enough to *show* 2026 alongside the historical context views: the
results so far, inferred stages, and each team's path — coloured, like the rest of the notebook,
by opponents' within-edition strength percentile (from the full-history Elo we already build).

Stages are inferred from the fixed 48-team bracket by chronological order (72 group matches, then
16 + 8 + 4 + 2 + 1 + 1), which is deterministic. Re-run after later rounds to refresh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .intl_elo import load_intl_results
from .reports import canonical_name

TOURNAMENT_START = np.datetime64("2026-06-11")

# Fixed 48-team knockout bracket, in order — used to label matches by chronological position.
_STAGE_SIZES = [
    ("group stage", 72),
    ("round of 32", 16),
    ("round of 16", 8),
    ("quarter-finals", 4),
    ("semi-finals", 2),
    ("third-place match", 1),
    ("final", 1),
]
STAGE_SHORT_2026 = {
    "group stage": "GRP", "round of 32": "R32", "round of 16": "R16",
    "quarter-finals": "QF", "semi-finals": "SF", "third-place match": "3rd", "final": "FIN",
}
_STAGE_RANK_2026 = {name: i for i, (name, _) in enumerate(_STAGE_SIZES)}


def load_2026(*, refresh: bool = False) -> pd.DataFrame:
    """Played 2026 World Cup matches with inferred stages, oldest first.

    Columns: ``date, home_team, away_team, home_score, away_score, stage, stage_short,
    game_index``. Only completed matches are returned.
    """
    r = load_intl_results(refresh=refresh)
    w = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026)]
    w = w.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    for col in ("home_team", "away_team"):
        w[col] = w[col].map(canonical_name)

    stages = []
    for name, size in _STAGE_SIZES:
        stages.extend([name] * size)
    stages = (stages + ["unknown"] * len(w))[: len(w)]
    w = w.copy()
    w["stage"] = stages
    w["stage_short"] = w["stage"].map(STAGE_SHORT_2026).fillna("?")
    w["game_index"] = np.arange(len(w))
    return w


def entering_ratings_2026(elo_timeline: dict, matches_2026: pd.DataFrame) -> dict:
    """Each 2026 team's full-history Elo just before the tournament began (``{team: rating}``)."""
    teams = set(matches_2026["home_team"]) | set(matches_2026["away_team"])
    ratings = {}
    for t in teams:
        tl = elo_timeline.get(t)
        if tl is None:
            ratings[t] = 1500.0
            continue
        dates, vals = tl
        idx = int(np.searchsorted(dates, TOURNAMENT_START, side="left"))
        ratings[t] = float(vals[idx - 1]) if idx > 0 else 1500.0
    return ratings


def pre_tournament_strength_2026(elo_timeline: dict, matches_2026: pd.DataFrame) -> dict:
    """Each 2026 team's within-tournament strength percentile from its entering Elo.

    Returns ``{team: percentile}`` (0–100, 100 = strongest of the 48).
    """
    s = pd.Series(entering_ratings_2026(elo_timeline, matches_2026))
    return (s.rank(pct=True) * 100).to_dict()


def group_membership_2026(matches_2026: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the 12 groups from the group-stage matches: columns ``group, team``.

    In the 48-team format each team plays its three group-mates, so a team plus its group-stage
    opponents is exactly its group of four.
    """
    gs = matches_2026[matches_2026["stage"] == "group stage"]
    opponents: dict[str, set] = {}
    for r in gs.itertuples():
        opponents.setdefault(r.home_team, set()).add(r.away_team)
        opponents.setdefault(r.away_team, set()).add(r.home_team)
    groups = {frozenset({t} | opps) for t, opps in opponents.items()}
    rows = []
    for i, grp in enumerate(sorted(groups, key=lambda s: sorted(s))):
        label = f"Group {chr(65 + i)}"
        rows.extend({"group": label, "team": t} for t in sorted(grp))
    return pd.DataFrame(rows)


def draw_luck_2026(
    matches_2026: pd.DataFrame,
    elo_timeline: dict,
    *,
    n_sims: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Fair-draw Monte-Carlo on the 48-team bracket (12 groups of 4).

    Same method as :func:`worldcup_anomalies.draw.draw_monte_carlo`: hold each group's strongest
    team fixed (the one seeding guarantee) and reshuffle the other 36 teams across the 12 groups
    thousands of times. Returns one row per group anchor with ``draw_luck_pct`` (low = its group
    was softer than most fair draws) and the shared ``rival_clustering_pct`` (high = strong teams
    bunched together more than random). Also flags whether the anchor reached the quarter-finals.
    """
    gm = group_membership_2026(matches_2026)
    ratings = entering_ratings_2026(elo_timeline, matches_2026)
    gm = gm.assign(elo=gm["team"].map(ratings)).reset_index(drop=True)

    groups = sorted(gm["group"].unique())
    ng = len(groups)
    sizes = gm.groupby("group").size()
    gsize = int(sizes.iloc[0])
    if sizes.nunique() != 1:
        raise ValueError("2026 groups are not uniform in size")

    anchor_idx = gm.groupby("group")["elo"].idxmax()
    g_anchor = gm.loc[anchor_idx].set_index("group")
    anchor_elo = np.array([g_anchor.loc[gr, "elo"] for gr in groups])
    anchor_team = {gr: g_anchor.loc[gr, "team"] for gr in groups}
    gtot = gm.groupby("group")["elo"].sum().reindex(groups).to_numpy()
    actual_var = gtot.var()
    actual_nonanchor = gtot - anchor_elo
    pool = gm.loc[~gm.index.isin(anchor_idx), "elo"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_sims, pool.size)), axis=1)
    vals = pool[perm].reshape(n_sims, ng, gsize - 1).sum(axis=2)
    null_var = (vals + anchor_elo[None, :]).var(axis=1)
    rival_clustering_pct = float((null_var <= actual_var).mean() * 100)

    qf = set(quarterfinalists_2026(matches_2026))
    rows = []
    for i, gr in enumerate(groups):
        rows.append({
            "group": gr,
            "team": anchor_team[gr],
            "draw_luck_pct": float((vals[:, i] <= actual_nonanchor[i]).mean() * 100),
            "rival_clustering_pct": rival_clustering_pct,
            "reached_qf": anchor_team[gr] in qf,
        })
    return pd.DataFrame(rows).sort_values("draw_luck_pct").reset_index(drop=True)


def team_path_2026(
    matches_2026: pd.DataFrame, team: str, strength_pct: dict
) -> pd.DataFrame:
    """One team's game-by-game 2026 path, schema-compatible with the notebook path plotter."""
    team = canonical_name(team)
    sub = matches_2026[
        (matches_2026["home_team"] == team) | (matches_2026["away_team"] == team)
    ].sort_values("game_index")
    rows = []
    for i, r in enumerate(sub.itertuples(), start=1):
        if r.home_team == team:
            opp, gf, ga = r.away_team, r.home_score, r.away_score
        else:
            opp, gf, ga = r.home_team, r.away_score, r.home_score
        result = "win" if gf > ga else ("lose" if gf < ga else "draw")
        rows.append({
            "game_no": i,
            "stage": r.stage,
            "stage_short": r.stage_short,
            "opponent": opp,
            "goals_for": int(gf),
            "goals_against": int(ga),
            "result": result,
            "opponent_strength_pct": strength_pct.get(opp, np.nan),
        })
    return pd.DataFrame(rows)


def furthest_stage_2026(matches_2026: pd.DataFrame) -> pd.DataFrame:
    """Furthest stage each team has reached, plus whether they won that game."""
    rows = []
    for team in sorted(set(matches_2026["home_team"]) | set(matches_2026["away_team"])):
        sub = matches_2026[
            (matches_2026["home_team"] == team) | (matches_2026["away_team"] == team)
        ]
        last = sub.sort_values("game_index").iloc[-1]
        if last["home_team"] == team:
            won = last["home_score"] > last["away_score"]
        else:
            won = last["away_score"] > last["home_score"]
        rows.append({
            "team": team,
            "furthest_stage": last["stage"],
            "stage_rank": _STAGE_RANK_2026.get(last["stage"], 0),
            "won_last": bool(won),
        })
    return pd.DataFrame(rows).sort_values("stage_rank", ascending=False).reset_index(drop=True)


def quarterfinalists_2026(matches_2026: pd.DataFrame) -> list[str]:
    """Teams that reached at least the quarter-finals (the deepest, most interesting runs)."""
    qf_teams = matches_2026.loc[
        matches_2026["stage"].map(_STAGE_RANK_2026) >= _STAGE_RANK_2026["quarter-finals"],
        ["home_team", "away_team"],
    ]
    return sorted(set(qf_teams["home_team"]) | set(qf_teams["away_team"]))


def summary_2026(matches_2026: pd.DataFrame) -> dict:
    """Headline status for the in-progress tournament."""
    stage_counts = matches_2026["stage"].value_counts()
    latest = max(_STAGE_RANK_2026, key=lambda s: (_STAGE_RANK_2026[s]
                 if s in stage_counts.index else -1))
    fs = furthest_stage_2026(matches_2026)
    still_in = fs[(fs["stage_rank"] == fs["stage_rank"].max()) & (fs["won_last"])]["team"].tolist()
    return {
        "teams": int(pd.concat([matches_2026["home_team"], matches_2026["away_team"]]).nunique()),
        "games_played": int(len(matches_2026)),
        "games_total": 104,
        "latest_stage": latest,
        "advanced_from_latest": sorted(still_in),
    }
