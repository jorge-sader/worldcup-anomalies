"""Draw-manipulation analysis: distinguishing a *lucky* draw from an *engineered* one.

The group-softness metric in :mod:`worldcup_anomalies.paths` shows *whether* a team got a soft
group, but not whether that softness is surprising given a fair draw — nor whether the other
strong teams were **packed together elsewhere** (so they eliminated each other and cleared the
favoured team's path). That packing is the actual signature of an engineered draw.

We test it with a Monte-Carlo of fair draws. The one thing we can assert without external pot
data is that **the strongest team in each group is separated by design** (seeding keeps top teams
apart). So we hold each group's strongest team ("anchor") fixed and randomly redistribute all the
remaining teams across the groups, thousands of times, and compare the real draw to that null:

- ``draw_luck_pct`` (per anchor): the percentile of its group's opponent strength under fair
  redraws. **Low = the group was softer than almost any fair draw** (a lucky/soft draw).
- ``rival_clustering_pct`` (per edition): the percentile of the *variance of group strength*.
  **High = the strong teams were bunched into some groups and absent from others** more than a
  random draw would produce — i.e. rivals clustered.

The *engineered* pattern is the conjunction: a deep run + a soft group (low ``draw_luck_pct``) +
clustered rivals (high ``rival_clustering_pct``). Any single one alone is ordinary.

Caveat: this null holds only the per-group anchor fixed; real draws also constrain pots 2-4 to one
per group, which this ignores. That makes the null *more* permissive than reality, so the test is
**conservative** — it will under- rather than over-state how extreme a real draw was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import pre_tournament_ratings
from .paths import _furthest_round, group_draw_difficulty


def draw_monte_carlo(
    elo_matches: pd.DataFrame,
    group_standings: pd.DataFrame,
    team_appearances: pd.DataFrame,
    *,
    n_sims: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte-Carlo fair-draw analysis, one row per group anchor (its strongest team).

    Returns: ``tournament_id, year, team_name, group_name, round_label, round_rank,
    draw_luck_pct, rival_clustering_pct``. Editions without uniform group sizes (or groups of
    two) are skipped. The RNG is seeded, so results are reproducible.
    """
    pre = pre_tournament_ratings(elo_matches)
    year_by_t = pre.groupby("tournament_id")["year"].first()
    gs = group_standings.copy()
    first = gs.groupby("tournament_id")["stage_number"].transform("min")
    gs = gs[gs["stage_number"] == first]

    furthest = _furthest_round(team_appearances).set_index(["tournament_id", "team_id"])
    rng = np.random.default_rng(seed)
    rows = []

    for tid, gtab in gs.groupby("tournament_id"):
        g = gtab[["group_name", "team_id", "team_name"]].merge(
            pre[pre["tournament_id"] == tid][["team_id", "elo_pre"]], on="team_id"
        )
        sizes = g.groupby("group_name").size()
        if sizes.nunique() != 1 or sizes.iloc[0] < 3:
            continue
        ng, gsize = g["group_name"].nunique(), int(sizes.iloc[0])
        groups = sorted(g["group_name"].unique())

        anchor_idx = g.groupby("group_name")["elo_pre"].idxmax()
        g_anchor = g.loc[anchor_idx].set_index("group_name")
        anchor_elo = np.array([g_anchor.loc[gr, "elo_pre"] for gr in groups])
        anchor_team = {gr: g_anchor.loc[gr, "team_name"] for gr in groups}
        anchor_id = {gr: g_anchor.loc[gr, "team_id"] for gr in groups}

        gtot = g.groupby("group_name")["elo_pre"].sum().reindex(groups).to_numpy()
        actual_var = gtot.var()
        actual_nonanchor = gtot - anchor_elo  # actual opponent-strength sum per anchor

        pool = g.loc[~g.index.isin(anchor_idx), "elo_pre"].to_numpy()  # ng*(gsize-1) teams
        # Vectorised fair redraws: permute the pool, split into groups, sum each group.
        perm = np.argsort(rng.random((n_sims, pool.size)), axis=1)
        vals = pool[perm].reshape(n_sims, ng, gsize - 1).sum(axis=2)  # (n_sims, ng)
        null_var = (vals + anchor_elo[None, :]).var(axis=1)
        rival_clustering_pct = float((null_var <= actual_var).mean() * 100)

        for i, gr in enumerate(groups):
            draw_luck = float((vals[:, i] <= actual_nonanchor[i]).mean() * 100)
            key = (tid, anchor_id[gr])
            rlabel = furthest.loc[key, "round_label"] if key in furthest.index else None
            rrank = furthest.loc[key, "rank"] if key in furthest.index else np.nan
            rows.append({
                "tournament_id": tid,
                "year": int(year_by_t.get(tid, 0)),
                "team_name": anchor_team[gr],
                "group_name": gr,
                "round_label": rlabel,
                "round_rank": rrank,
                "draw_luck_pct": draw_luck,
                "rival_clustering_pct": rival_clustering_pct,
            })

    return pd.DataFrame(rows).sort_values(["year", "draw_luck_pct"]).reset_index(drop=True)


def edition_draw_summary(
    elo_matches: pd.DataFrame,
    group_standings: pd.DataFrame,
    team_appearances: pd.DataFrame,
    tournaments: pd.DataFrame,
    *,
    n_sims: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per World Cup: how soft was the winner's draw, and how balanced was the draw.

    A myth-buster at a glance. Columns:

        ``champion``                 : the tournament winner
        ``champ_group_softness_pct`` : within-edition, 100 = champion had the softest group
        ``champ_softness_rank``      : the champion's group softness rank (1 = softest of ``n_teams``)
        ``rival_clustering_pct``     : 100 = powers packed into some groups (lopsided draw), 0 = balanced
        ``softest_group_team``       : who actually drew the softest group that edition
        ``softest_team_won``         : did that team go on to win? (usually no)

    Editions without a uniform group format are omitted.
    """
    draws = draw_monte_carlo(
        elo_matches, group_standings, team_appearances, n_sims=n_sims, seed=seed
    )
    gd = group_draw_difficulty(elo_matches, team_appearances)
    gd["softness_rank"] = gd.groupby("tournament_id")["grp_mean_opp_elo"].rank(method="min")
    gd["n_teams"] = gd.groupby("tournament_id")["team_id"].transform("count")
    t = tournaments[["tournament_id", "year", "host_country", "winner"]]

    rows = []
    for tid, g in draws.groupby("tournament_id"):
        info = t[t["tournament_id"] == tid]
        if info.empty:
            continue
        info = info.iloc[0]
        softest = g.loc[g["draw_luck_pct"].idxmin()]
        champ = gd[(gd["tournament_id"] == tid) & (gd["team_name"] == info["winner"])]
        rows.append({
            "year": int(info["year"]),
            "host": info["host_country"],
            "champion": info["winner"],
            "champ_group_softness_pct": (
                float(champ["grp_softness_pct"].iloc[0]) if len(champ) else np.nan
            ),
            "champ_softness_rank": int(champ["softness_rank"].iloc[0]) if len(champ) else -1,
            "n_teams": int(champ["n_teams"].iloc[0]) if len(champ) else -1,
            "rival_clustering_pct": float(g["rival_clustering_pct"].iloc[0]),
            "softest_group_team": softest["team_name"],
            "softest_team_won": bool(softest["team_name"] == info["winner"]),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def engineered_draw_flags(
    draws: pd.DataFrame,
    *,
    soft_max: float = 20.0,
    cluster_min: float = 80.0,
    min_round_rank: int = 6,
) -> pd.DataFrame:
    """The conjunction that would look engineered: deep run + soft group + clustered rivals.

    Filters :func:`draw_monte_carlo` output to anchors that reached at least ``min_round_rank``
    (SF/Final) whose group was softer than ``soft_max`` percentile of fair draws *and* whose
    edition's rivals were more clustered than ``cluster_min`` percentile. Deliberately strict —
    each condition alone is ordinary; only together do they match the threat model.
    """
    d = draws.dropna(subset=["round_rank"])
    return d[
        (d["round_rank"] >= min_round_rank)
        & (d["draw_luck_pct"] <= soft_max)
        & (d["rival_clustering_pct"] >= cluster_min)
    ].reset_index(drop=True)
