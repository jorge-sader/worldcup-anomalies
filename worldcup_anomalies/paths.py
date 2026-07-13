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
