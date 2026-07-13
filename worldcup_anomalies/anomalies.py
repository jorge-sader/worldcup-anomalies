"""Unified anomaly scoring.

Each detector speaks a different native language — a surprise in nats, a referee z-score, an
easiness percentile, a suspicion index. This module maps them all onto one comparable
``anomaly_score`` (a standard-normal-style z) so they can be ranked in a single "worth looking
into" table, and attaches a Benjamini-Hochberg q-value where a genuine p-value exists.

IMPORTANT: this is an *exploratory screen*, not a hypothesis test. We compute many statistics
across the same fixed history, so individual scores are inflated by multiple comparisons — the
ranked table is a list of leads to investigate, not proven anomalies. The BH q-values make that
multiplicity explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .elo import compute_elo
from .fetch import WorldCupData
from .leadership import host_overperformance
from .models import detect_convenient_results, score_surprise
from .paths import easy_path_scores, group_draw_difficulty
from .referees import referee_outliers, host_card_bias


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q_ranked, 0, 1)
    q[ok] = out
    return q


def _robust_z(values: pd.Series, *, two_sided: bool = True) -> pd.Series:
    """Median/MAD-based z-score (outlier-robust)."""
    v = values.astype(float)
    med = v.median()
    mad = (v - med).abs().median()
    scale = mad * 1.4826 if mad > 0 else v.std(ddof=0)
    if not scale or np.isnan(scale):
        return pd.Series(np.zeros(len(v)), index=v.index)
    z = (v - med) / scale
    return z.abs() if two_sided else z


def collect_anomalies(
    data: WorldCupData,
    elo_matches: pd.DataFrame | None = None,
    *,
    referee_min_matches: int = 5,
    surprise_top: int = 15,
) -> pd.DataFrame:
    """Run every detector and return one ranked, normalized irregularities table.

    Columns: ``category, subject, detail, tournament_id, statistic, anomaly_score, p_value,
    q_value`` — sorted by ``anomaly_score`` descending. ``anomaly_score`` is a comparable
    z-magnitude; ``p_value`` is populated only where a real test exists (referees, host bias,
    leadership), and ``q_value`` is its BH-adjusted counterpart.
    """
    if elo_matches is None:
        elo_matches = compute_elo(data.matches)

    records: list[dict] = []

    # --- 1. Upsets (results that defied team strength) -------------------------------
    # A favourite thrashing a minnow is statistically "surprising" but not worth looking
    # into; a weaker side beating a strong one is. We therefore screen on genuine upsets
    # from the diverged-Elo era, ranked by how improbable the exact result was.
    ss = score_surprise(elo_matches, data.team_appearances)
    upsets = ss[(ss["upset"]) & (ss["year"] >= 1958)].copy()
    z_upset = (
        _robust_z(upsets["elo_gap_defied"], two_sided=False)
        if len(upsets) else pd.Series(dtype=float)
    )
    ss_top = upsets.sort_values("elo_gap_defied", ascending=False).head(surprise_top)
    for i, r in ss_top.iterrows():
        records.append({
            "category": "upset",
            "subject": r["match_name"],
            "detail": (
                f"{int(r['home_team_score'])}-{int(r['away_team_score'])} in {r['stage_name']}; "
                f"favourite (Elo gap {r['elo_gap_defied']:.0f}) lost, "
                f"expected ~{r['exp_goals_home']:.1f}-{r['exp_goals_away']:.1f}"
            ),
            "tournament_id": r["tournament_id"],
            "statistic": float(r["elo_gap_defied"]),
            "anomaly_score": float(z_upset.loc[i]),
            "p_value": np.nan,
        })

    # --- 2. Convenient group results -------------------------------------------------
    cr = detect_convenient_results(data.matches, data.group_standings)
    for _, r in cr.iterrows():
        records.append({
            "category": "convenient-result",
            "subject": r["match_name"],
            "detail": (
                f"{r['score']}, both advanced; eliminated {r['eliminated_finished_earlier']} "
                f"who finished earlier"
                + ("; non-simultaneous final match" if r["non_simultaneous"] else "")
            ),
            "tournament_id": r["tournament_id"],
            "statistic": float(r["suspicion"]),
            # Map suspicion (~1-1.75) to a comparable z on a fixed scale.
            "anomaly_score": float(2.0 + (r["suspicion"] - 1.0) * 2.0),
            "p_value": np.nan,
        })

    # --- 3. Referee card outliers ----------------------------------------------------
    ro = referee_outliers(
        data.matches, data.referee_appearances, data.bookings, elo_matches,
        min_matches=referee_min_matches,
    )
    for _, r in ro.iterrows():
        p = 2.0 * stats.norm.sf(abs(r["z"]))
        records.append({
            "category": "referee-cards",
            "subject": f"{r['referee_name']} ({r['referee_country']})",
            "detail": (
                f"{int(r['n_matches'])} matches, {int(r['obs_cards'])} cards vs "
                f"{r['exp_cards']:.0f} expected ({r['cards_per_match']:.1f}/match)"
            ),
            "tournament_id": "",
            "statistic": float(r["z"]),
            "anomaly_score": float(abs(r["z"])),
            "p_value": float(p),
        })

    # --- 4. Host card bias (one tournament-agnostic finding) -------------------------
    _, host_bias = host_card_bias(data.matches, data.bookings)
    if host_bias.get("n_decisive_matches", 0):
        p = host_bias["sign_test_p"]
        z = stats.norm.isf(p / 2.0) if p and p > 0 else 0.0
        records.append({
            "category": "host-card-bias",
            "subject": "Hosts receive fewer cards than opponents",
            "detail": (
                f"host got fewer cards in {host_bias['host_fewer_cards']}/"
                f"{host_bias['n_decisive_matches']} decisive matches "
                f"({host_bias['share_host_fewer']:.0%})"
            ),
            "tournament_id": "",
            "statistic": float(host_bias["share_host_fewer"]),
            "anomaly_score": float(z),
            "p_value": float(p),
        })

    # --- 5. Easy paths (SF/Final reached via weak schedule) --------------------------
    ep = easy_path_scores(elo_matches, data.team_appearances)
    deep = ep[ep["round_rank"] >= 6]  # SF or Final
    for _, r in deep[deep["easiness_pct"] >= 80].iterrows():
        records.append({
            "category": "easy-path",
            "subject": f"{int(r['year'])} {r['team_name']} ({r['round_label']})",
            "detail": (
                f"mean opponent Elo {r['mean_opp_elo']:.0f}, faced {int(r['n_powers_faced'])} "
                f"top-quartile teams; weaker than {r['easiness_pct']:.0f}% of "
                f"{r['round_label']} runs"
            ),
            "tournament_id": r["tournament_id"],
            "statistic": float(r["easiness_pct"]),
            # Percentile -> z within the reached round.
            "anomaly_score": float(stats.norm.isf((100.5 - r["easiness_pct"]) / 100.0)),
            "p_value": np.nan,
        })

    # --- 5b. Soft group draw then deep run -------------------------------------------
    # The draw is the real lever: by the quarter-final nobody avoids strong teams, so what
    # matters is whether a team was handed a soft GROUP and coasted to the latter stages.
    # Measured within-edition and among the top-8 seeds (seeding separates strong teams by
    # design, so this is descriptive/structural, NOT proof of manipulation).
    gd = group_draw_difficulty(elo_matches, data.team_appearances)
    soft_deep = gd[(gd["rank"] >= 6) & (gd["is_seed"]) & (gd["seed_soft_rank"] <= 2)]
    for _, r in soft_deep.iterrows():
        records.append({
            "category": "soft-group-draw",
            "subject": f"{int(r['year'])} {r['team_name']} ({r['round_label']})",
            "detail": (
                f"reached {r['round_label']} from the #{int(r['seed_soft_rank'])}-softest group "
                f"among {int(r['n_seeds_total'])} seeds ({int(r['n_seeds_in_group'])} fellow "
                f"seeds in group). STRUCTURAL: soft seeded groups are common — a lead, not proof"
            ),
            "tournament_id": r["tournament_id"],
            "statistic": float(r["grp_softness_pct"]),
            # Softest seed group (rank 1) scores a bit higher than 2nd; kept deliberately modest.
            "anomaly_score": float(2.6 if r["seed_soft_rank"] == 1 else 2.2),
            "p_value": np.nan,
        })

    # --- 6. Host overperformance extremes --------------------------------------------
    ho = host_overperformance(
        data.tournaments, elo_matches, data.team_appearances, data.leadership
    )
    z_over = _robust_z(ho["overperformance"], two_sided=False)
    for i, r in ho.iterrows():
        if abs(z_over.loc[i]) < 1.5:
            continue
        direction = "over" if r["overperformance"] > 0 else "under"
        records.append({
            "category": "host-performance",
            "subject": f"{int(r['year'])} {r['team_name']} (host)",
            "detail": (
                f"{direction}performed Elo by {r['overperformance']:+.1f} rounds "
                f"(reached {r['round_label']}); president: {r['president']}"
            ),
            "tournament_id": r["tournament_id"],
            "statistic": float(r["overperformance"]),
            "anomaly_score": float(abs(z_over.loc[i])),
            "p_value": np.nan,
        })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    # Clip to a common, comparable magnitude so no single heavy-tailed category (e.g. an
    # extreme upset) dwarfs genuinely anomalous findings in other categories.
    out["anomaly_score"] = out["anomaly_score"].clip(lower=0.0, upper=5.0)
    out["q_value"] = _bh_qvalues(out["p_value"].to_numpy())
    out = out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[[
        "rank", "category", "subject", "detail", "tournament_id",
        "statistic", "anomaly_score", "p_value", "q_value",
    ]]
