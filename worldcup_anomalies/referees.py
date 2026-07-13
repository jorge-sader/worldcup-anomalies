"""Referee discipline anomaly detectors (card data exists 1970+).

Two views:

1. :func:`referee_outliers` — model expected cards per match from context (era trend, knockout
   vs group, how competitive the match is by Elo), then aggregate each referee's standardized
   residual across their matches. Referees whose matches carry far more/fewer cards than the
   model expects float to the top.

2. :func:`host_card_bias` — in matches involving the host nation, does the host systematically
   receive fewer cards than its opponent? Aggregated per tournament and overall with a sign test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def match_card_counts(bookings: pd.DataFrame) -> pd.DataFrame:
    """Cards per match: total card events, and sendings-off."""
    if bookings.empty:
        return pd.DataFrame(columns=["match_id", "n_cards", "n_sendoff"])
    g = bookings.groupby("match_id")
    out = pd.DataFrame({
        "n_cards": g.size(),
        "n_sendoff": g["sending_off"].sum(),
    }).reset_index()
    return out


def _referee_match_frame(
    matches: pd.DataFrame,
    referee_appearances: pd.DataFrame,
    bookings: pd.DataFrame,
    elo_matches: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per match that has card data: cards + context + referee identity."""
    cards = match_card_counts(bookings)
    booked_matches = set(bookings["match_id"].unique())

    m = matches[matches["match_id"].isin(booked_matches)].copy()
    m = m.merge(cards, on="match_id", how="left")
    m["n_cards"] = m["n_cards"].fillna(0)
    m["n_sendoff"] = m["n_sendoff"].fillna(0)
    m["year"] = pd.to_datetime(m["match_date"]).dt.year
    m["knockout"] = m["knockout_stage"].astype(int)

    if elo_matches is not None:
        elo = elo_matches[["match_id", "elo_home_pre", "elo_away_pre"]]
        m = m.merge(elo, on="match_id", how="left")
        m["closeness"] = -(m["elo_home_pre"] - m["elo_away_pre"]).abs()
        m["closeness"] = m["closeness"].fillna(m["closeness"].mean())
    else:
        m["closeness"] = 0.0

    ref = referee_appearances[
        ["match_id", "referee_id", "family_name", "given_name", "country_name"]
    ].rename(columns={"country_name": "referee_country"})
    m = m.merge(ref, on="match_id", how="left")
    m["referee_name"] = (
        m["given_name"].fillna("") + " " + m["family_name"].fillna("")
    ).str.strip()
    return m


def referee_outliers(
    matches: pd.DataFrame,
    referee_appearances: pd.DataFrame,
    bookings: pd.DataFrame,
    elo_matches: pd.DataFrame | None = None,
    *,
    min_matches: int = 5,
) -> pd.DataFrame:
    """Rank referees by how much their matches deviate from expected card counts.

    Fits Poisson ``n_cards ~ year + knockout + closeness`` across all carded matches, then for
    each referee computes z = sum(obs - exp) / sqrt(sum(exp)). Positive z = card-happy relative
    to context; negative = unusually lenient. Only referees with >= ``min_matches`` are returned.
    """
    frame = _referee_match_frame(matches, referee_appearances, bookings, elo_matches)
    frame = frame.dropna(subset=["referee_id"])

    yc = frame["year"] - frame["year"].mean()
    X = sm.add_constant(pd.DataFrame({
        "year_c": yc,
        "knockout": frame["knockout"],
        "closeness": frame["closeness"],
    }))
    res = sm.GLM(frame["n_cards"].astype(float), X, family=sm.families.Poisson()).fit()
    frame = frame.assign(expected=res.fittedvalues)

    grp = frame.groupby(["referee_id", "referee_name", "referee_country"])
    agg = grp.agg(
        n_matches=("match_id", "nunique"),
        obs_cards=("n_cards", "sum"),
        exp_cards=("expected", "sum"),
        sendoffs=("n_sendoff", "sum"),
    ).reset_index()
    agg = agg[agg["n_matches"] >= min_matches].copy()
    agg["z"] = (agg["obs_cards"] - agg["exp_cards"]) / np.sqrt(agg["exp_cards"].clip(lower=1e-6))
    agg["cards_per_match"] = agg["obs_cards"] / agg["n_matches"]
    agg["exp_per_match"] = agg["exp_cards"] / agg["n_matches"]
    return agg.sort_values("z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def host_card_bias(
    matches: pd.DataFrame,
    bookings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Do hosts receive fewer cards than opponents in matches they play in?

    For every carded match involving the host nation (venue country == a participant), count
    cards to the host vs its opponent. Returns (per-tournament table, overall sign-test dict).
    """
    booked = set(bookings["match_id"].unique())
    m = matches[matches["match_id"].isin(booked)].copy()
    # Identify host side per match (host plays in its own country).
    m["host_team_id"] = np.where(
        m["country_name"] == m["home_team_name"], m["home_team_id"],
        np.where(m["country_name"] == m["away_team_name"], m["away_team_id"], None),
    )
    m = m[m["host_team_id"].notna()]
    if m.empty:
        return pd.DataFrame(), {"n": 0}

    cards_by_team = (
        bookings.groupby(["match_id", "team_id"]).size().rename("cards").reset_index()
    )

    rows = []
    for r in m.itertuples():
        host_id = r.host_team_id
        opp_id = r.away_team_id if host_id == r.home_team_id else r.home_team_id
        c = cards_by_team[cards_by_team["match_id"] == r.match_id]
        host_cards = int(c.loc[c["team_id"] == host_id, "cards"].sum())
        opp_cards = int(c.loc[c["team_id"] == opp_id, "cards"].sum())
        rows.append({
            "tournament_id": r.tournament_id,
            "match_id": r.match_id,
            "host_cards": host_cards,
            "opp_cards": opp_cards,
            "diff": host_cards - opp_cards,
        })
    detail = pd.DataFrame(rows)

    per_t = detail.groupby("tournament_id").agg(
        matches=("match_id", "nunique"),
        host_cards=("host_cards", "sum"),
        opp_cards=("opp_cards", "sum"),
        mean_diff=("diff", "mean"),
    ).reset_index()
    per_t["host_minus_opp"] = per_t["host_cards"] - per_t["opp_cards"]

    # Overall sign test: in how many matches did the host get FEWER cards than the opponent?
    decisive = detail[detail["diff"] != 0]
    n = len(decisive)
    host_fewer = int((decisive["diff"] < 0).sum())
    p = stats.binomtest(host_fewer, n, 0.5).pvalue if n else np.nan
    overall = {
        "n_decisive_matches": n,
        "host_fewer_cards": host_fewer,
        "host_more_cards": n - host_fewer,
        "share_host_fewer": host_fewer / n if n else np.nan,
        "sign_test_p": p,
        "total_host_cards": int(detail["host_cards"].sum()),
        "total_opp_cards": int(detail["opp_cards"].sum()),
    }
    return per_t, overall
