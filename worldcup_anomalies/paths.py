"""Per-team "easy path" / seeding-luck detector.

For every team that reached the quarter-final or deeper, we measure the strength of the teams
it actually had to get past — its *strength of schedule* — using each opponent's pre-tournament
Elo. A team that reached the semi-final or final having faced unusually weak opponents (and few
established "powers") floats up as a lucky/easy run, independent of whether it was the host.

Comparison is made *within* the round reached (finalists vs finalists, semi-finalists vs
semi-finalists) so that facing more games in deeper runs does not distort the ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import pre_tournament_ratings

# Map raw stage names to a monotone "depth" rank, tolerant of historical formats
# (1950 final round-robin; 1974-82 second group stage).
_STAGE_RANK = {
    "group stage": 2,
    "first group stage": 2,
    "round of 16": 4,
    "second group stage": 5,
    "quarter-finals": 5,
    "final round": 6,      # 1950 top-4 round robin (semi-final-equivalent depth)
    "semi-finals": 6,
    "third-place match": 6,
    "final": 7,
}

_ROUND_LABEL = {2: "group", 4: "R16", 5: "QF", 6: "SF", 7: "Final"}


def _furthest_round(team_appearances: pd.DataFrame) -> pd.DataFrame:
    """Furthest round rank/label per (tournament_id, team_id)."""
    ta = team_appearances.copy()
    ta["rank"] = ta["stage_name"].map(_STAGE_RANK).fillna(2)
    g = ta.groupby(["tournament_id", "team_id", "team_name"], as_index=False)["rank"].max()
    g["round_label"] = g["rank"].map(_ROUND_LABEL)
    return g


def easy_path_scores(
    elo_matches: pd.DataFrame,
    team_appearances: pd.DataFrame,
    *,
    min_round_rank: int = 5,       # QF or deeper
    power_quantile: float = 0.75,  # "established power" = top quarter of edition by Elo
    min_year: int | None = 1958,   # skip cold-start editions where all Elos are ~1500
) -> pd.DataFrame:
    """Strength-of-schedule for deep runs, with an in-round easiness percentile.

    Returns one row per (tournament, team) that reached at least ``min_round_rank`` (QF+):
    the mean/max opponent pre-tournament Elo, number of established powers faced, and
    ``easiness_pct`` — the percentile of *weakness* within teams that reached the same round
    (100 = weakest path to that round). High ``easiness_pct`` at SF/Final is the flag.

    ``min_year`` drops the earliest editions: because every team enters its first tournament at
    the 1500 base rating, opponent strength is uninformative until Elo has diverged (~1958),
    and including 1930-1938 would fill the "easiest path" list with rating cold-start artefacts.
    Pass ``min_year=None`` to include everything.
    """
    pre = pre_tournament_ratings(elo_matches)  # tournament_id, team_id, elo_pre, year
    pre_lookup = pre.set_index(["tournament_id", "team_id"])["elo_pre"]

    # Edition-level power threshold.
    thr = (
        pre.groupby("tournament_id")["elo_pre"]
        .quantile(power_quantile)
        .rename("power_threshold")
    )

    furthest = _furthest_round(team_appearances)
    deep = furthest[furthest["rank"] >= min_round_rank].copy()
    if min_year is not None:
        keep_ids = set(pre.loc[pre["year"] >= min_year, "tournament_id"])
        deep = deep[deep["tournament_id"].isin(keep_ids)]

    # Opponents faced per (tournament, team).
    ta = team_appearances[["tournament_id", "team_id", "opponent_id"]].dropna()
    rows = []
    for r in deep.itertuples():
        opps = ta[(ta["tournament_id"] == r.tournament_id) & (ta["team_id"] == r.team_id)]
        opp_elos = [
            pre_lookup.get((r.tournament_id, oid), np.nan) for oid in opps["opponent_id"]
        ]
        opp_elos = [e for e in opp_elos if pd.notna(e)]
        if not opp_elos:
            continue
        threshold = thr.get(r.tournament_id, np.nan)
        n_powers = int(np.sum(np.array(opp_elos) >= threshold)) if pd.notna(threshold) else 0
        rows.append({
            "tournament_id": r.tournament_id,
            "team_id": r.team_id,
            "team_name": r.team_name,
            "round_label": r.round_label,
            "round_rank": r.rank,
            "n_opponents": len(opp_elos),
            "mean_opp_elo": float(np.mean(opp_elos)),
            "max_opp_elo": float(np.max(opp_elos)),
            "n_powers_faced": n_powers,
            "own_elo_pre": float(pre_lookup.get((r.tournament_id, r.team_id), np.nan)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(
        pre[["tournament_id", "year"]].drop_duplicates(), on="tournament_id", how="left"
    )

    # Easiness percentile WITHIN the round reached (weaker schedule -> higher percentile).
    out["easiness_pct"] = (
        out.groupby("round_label")["mean_opp_elo"]
        .rank(ascending=False, pct=True)
        .mul(100.0)
    )
    return out.sort_values(["round_rank", "easiness_pct"], ascending=[False, False]).reset_index(
        drop=True
    )


def phase_strength_split(
    elo_matches: pd.DataFrame,
    team_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Split each run's strength-of-schedule into its group vs knockout phases.

    A single whole-path average hides the shape of a run. This separates the two phases so the
    profile is visible: a team can "coast" through a soft group and only meet elite opposition
    once the bracket thins. Returns one row per (tournament, team) that reached the knockouts:

        ``group_opp_elo`` / ``ko_opp_elo``     : mean opponent pre-tournament Elo per phase
        ``ko_max_opp_elo``                     : toughest single knockout opponent
        ``group_softness_pct``                 : within-edition, 100 = softest group that year
        ``ko_toughness_pct``                   : within-edition (knockout teams), 100 = toughest
        ``split_index``                        : mean of the two — high = soft group + hard knockout

    Because Elo inflates over time, both percentiles are computed *within edition*; the raw
    per-phase Elo is returned too for context. Note the knockout mean includes the round of 16,
    so a run that is only brutal in the final one or two games scores moderately.
    """
    pre = pre_tournament_ratings(elo_matches)
    pre_lookup = pre.set_index(["tournament_id", "team_id"])["elo_pre"]

    def _phase_mean(sub: pd.DataFrame) -> float:
        elos = [
            pre_lookup.get((r.tournament_id, r.opponent_id), np.nan)
            for r in sub.itertuples()
        ]
        elos = [e for e in elos if pd.notna(e)]
        return float(np.mean(elos)) if elos else np.nan

    def _phase_max(sub: pd.DataFrame) -> float:
        elos = [
            pre_lookup.get((r.tournament_id, r.opponent_id), np.nan)
            for r in sub.itertuples()
        ]
        elos = [e for e in elos if pd.notna(e)]
        return float(np.max(elos)) if elos else np.nan

    rows = []
    for (tid, team), g in team_appearances.groupby(["tournament_id", "team_id"]):
        ko = g[g["knockout_stage"] == 1]
        if ko.empty:
            continue
        grp = g[g["group_stage"] == 1]
        rows.append({
            "tournament_id": tid,
            "team_id": team,
            "team_name": g["team_name"].iloc[0],
            "group_opp_elo": _phase_mean(grp),
            "ko_opp_elo": _phase_mean(ko),
            "ko_max_opp_elo": _phase_max(ko),
        })

    out = pd.DataFrame(rows).dropna(subset=["group_opp_elo", "ko_opp_elo"])
    if out.empty:
        return out
    out["group_softness_pct"] = 100.0 - out.groupby("tournament_id")["group_opp_elo"].rank(
        pct=True
    ).mul(100.0)
    out["ko_toughness_pct"] = out.groupby("tournament_id")["ko_opp_elo"].rank(pct=True).mul(100.0)
    out["split_index"] = (out["group_softness_pct"] + out["ko_toughness_pct"]) / 2.0

    furthest = _furthest_round(team_appearances)
    out = out.merge(
        furthest[["tournament_id", "team_id", "round_label", "rank"]],
        on=["tournament_id", "team_id"], how="left",
    ).merge(pre[["tournament_id", "year"]].drop_duplicates(), on="tournament_id", how="left")
    return out.sort_values("split_index", ascending=False).reset_index(drop=True)


def group_draw_difficulty(
    elo_matches: pd.DataFrame,
    team_appearances: pd.DataFrame,
    *,
    n_seeds: int = 8,
) -> pd.DataFrame:
    """How soft was each team's GROUP draw — the part of the path the seeding/draw controls.

    Motivation: by the quarter-final it is nearly impossible to avoid strong teams, so a
    whole-path average hides the lever that actually matters — the group draw. If you wanted to
    ease a team's route you would hand it a soft group (and let the other powers cluster
    elsewhere and eliminate each other). This isolates the group stage and, crucially, compares
    **within the same edition** (Elo inflates over time, so cross-era absolute comparison is
    invalid) and **among the top-``n_seeds`` seeds** (top teams are separated into different
    groups by design, so "strong team + soft group" is the norm and must be controlled for).

    Returns one row per (tournament, team): mean group-opponent Elo, number of fellow seeds in
    the group, a within-edition softness percentile (100 = softest group that year), and — for
    the seeds — ``seed_soft_rank`` (1 = softest group among that edition's seeds). Merged with the
    furthest round reached so soft-draw-then-deep-run cases are visible.
    """
    pre = pre_tournament_ratings(elo_matches)
    pre_lookup = pre.set_index(["tournament_id", "team_id"])["elo_pre"]
    seeds = {
        tid: set(g.nlargest(n_seeds, "elo_pre")["team_id"])
        for tid, g in pre.groupby("tournament_id")
    }

    grp = team_appearances[
        team_appearances["stage_name"].isin(["group stage", "first group stage"])
    ]
    rows = []
    for (tid, team), g in grp.groupby(["tournament_id", "team_id"]):
        opp_elos, n_seed_opps = [], 0
        for oid in g["opponent_id"]:
            e = pre_lookup.get((tid, oid), np.nan)
            if pd.notna(e):
                opp_elos.append(e)
            if oid in seeds.get(tid, set()):
                n_seed_opps += 1
        if not opp_elos:
            continue
        rows.append({
            "tournament_id": tid,
            "team_id": team,
            "team_name": g["team_name"].iloc[0],
            "grp_mean_opp_elo": float(np.mean(opp_elos)),
            "n_seeds_in_group": n_seed_opps,
            "is_seed": team in seeds.get(tid, set()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(
        pre[["tournament_id", "year"]].drop_duplicates(), on="tournament_id", how="left"
    )
    # Within-edition softness: 100 = softest group among ALL teams that year.
    out["grp_softness_pct"] = 100.0 - out.groupby("tournament_id")["grp_mean_opp_elo"].rank(
        pct=True
    ).mul(100.0)
    # Seed-controlled: rank among that edition's seeds (1 = softest group among seeds).
    seed_rank = (
        out[out["is_seed"]]
        .groupby("tournament_id")["grp_mean_opp_elo"]
        .rank(method="min")
    )
    out["seed_soft_rank"] = seed_rank
    out["n_seeds_total"] = out.groupby("tournament_id")["is_seed"].transform("sum")

    furthest = _furthest_round(team_appearances)
    out = out.merge(
        furthest[["tournament_id", "team_id", "round_label", "rank"]],
        on=["tournament_id", "team_id"], how="left",
    )
    return out.sort_values(["year", "grp_mean_opp_elo"]).reset_index(drop=True)
