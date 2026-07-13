"""FIFA-leadership correlation (exploratory, correlational only).

Maps each tournament to the FIFA president in office at kickoff, then aggregates
over/under-performance signals by presidential era. With only 22 tournaments spread over
seven presidents this is *suggestive, never conclusive* — a lens for "which eras are worth a
closer look", not a causal claim. Every function here is descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .elo import pre_tournament_ratings
from .fetch import president_at
from .paths import _furthest_round


def assign_president(
    df: pd.DataFrame, leadership: pd.DataFrame, date_col: str
) -> pd.DataFrame:
    """Add a ``president`` column based on the date in ``date_col``."""
    out = df.copy()
    out["president"] = pd.to_datetime(out[date_col]).apply(
        lambda d: president_at(d, leadership)
    )
    return out


def performance_residuals(
    elo_matches: pd.DataFrame, team_appearances: pd.DataFrame
) -> pd.DataFrame:
    """How far each team went vs what its pre-tournament Elo predicts.

    Fits OLS ``round_rank ~ elo_pre`` across all team-tournaments and returns the residual
    (positive = went further than its strength predicts). This is the engine behind host
    overperformance.
    """
    furthest = _furthest_round(team_appearances)
    pre = pre_tournament_ratings(elo_matches)
    df = furthest.merge(
        pre[["tournament_id", "team_id", "elo_pre", "year"]],
        on=["tournament_id", "team_id"],
        how="left",
    ).dropna(subset=["elo_pre"])

    X = sm.add_constant(df["elo_pre"].astype(float))
    res = sm.OLS(df["rank"].astype(float), X).fit()
    df["expected_rank"] = res.fittedvalues
    df["overperformance"] = df["rank"] - df["expected_rank"]
    return df


def host_overperformance(
    tournaments: pd.DataFrame,
    elo_matches: pd.DataFrame,
    team_appearances: pd.DataFrame,
    leadership: pd.DataFrame,
) -> pd.DataFrame:
    """Per tournament: how much the host over/under-performed its Elo, plus the president."""
    perf = performance_residuals(elo_matches, team_appearances)

    # Host team per tournament (match by host country name). Use start_date (a real date) for
    # president assignment; ``year`` is an integer and must not be parsed as a timestamp.
    t = tournaments[[
        "tournament_id", "year", "start_date", "host_country", "winner", "host_won"
    ]].rename(columns={"year": "t_year"})
    perf_named = perf.drop(columns=["year"], errors="ignore").merge(
        t, on="tournament_id", how="left"
    )
    host = perf_named[
        perf_named.apply(lambda r: str(r["team_name"]) in str(r["host_country"]), axis=1)
    ].copy()

    host = assign_president(host, leadership, "start_date")
    host = host.rename(columns={"t_year": "year"})
    keep = [
        "tournament_id", "year", "team_name", "host_country", "round_label",
        "rank", "expected_rank", "overperformance", "host_won", "president",
    ]
    keep = [c for c in keep if c in host.columns]
    return host[keep].sort_values("overperformance", ascending=False).reset_index(drop=True)


def era_summary(
    tournaments: pd.DataFrame,
    elo_matches: pd.DataFrame,
    team_appearances: pd.DataFrame,
    leadership: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate host performance and titles by presidential era."""
    host = host_overperformance(tournaments, elo_matches, team_appearances, leadership)
    g = host.groupby("president", as_index=False).agg(
        n_tournaments=("tournament_id", "nunique"),
        host_titles=("host_won", "sum"),
        mean_host_overperf=("overperformance", "mean"),
        host_reached_final=("rank", lambda s: int((s >= 7).sum())),
        host_reached_semi=("rank", lambda s: int((s >= 6).sum())),
    )
    # Order eras chronologically.
    order = {p: i for i, p in enumerate(leadership["president"])}
    g["_o"] = g["president"].map(order).fillna(99)
    return g.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def host_overperformance_permutation(
    host: pd.DataFrame,
    era_presidents: list[str],
    *,
    n_perm: int = 20000,
    seed: int = 0,
) -> dict:
    """Permutation test: is mean host overperformance higher in ``era_presidents``?

    Compares the observed mean host overperformance for the named eras against the null of
    randomly reassigning which tournaments fall in those eras. Purely exploratory.
    """
    rng = np.random.default_rng(seed)
    mask = host["president"].isin(era_presidents).to_numpy()
    vals = host["overperformance"].to_numpy()
    k = int(mask.sum())
    if k == 0 or k == len(vals):
        return {"observed_diff": np.nan, "p_value": np.nan, "n_in_era": k}

    observed = vals[mask].mean() - vals[~mask].mean()
    idx = np.arange(len(vals))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(idx)[:k]
        in_mask = np.zeros(len(vals), dtype=bool)
        in_mask[perm] = True
        diff = vals[in_mask].mean() - vals[~in_mask].mean()
        if diff >= observed:
            count += 1
    return {
        "eras": era_presidents,
        "n_in_era": k,
        "observed_diff": float(observed),
        "p_value": (count + 1) / (n_perm + 1),
    }
