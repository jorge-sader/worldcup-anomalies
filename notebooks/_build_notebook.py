"""Generate world_cup_irregularities.ipynb from source cells (keeps the notebook in git-diffable
form). Run: ``uv run python notebooks/_build_notebook.py`` then execute the .ipynb."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# World Cup Statistical Irregularities

**Goal.** Pull the full history of the men's FIFA World Cup (1930–2022) — matches, scores,
phases, referees, bookings — plus a FIFA-leadership table, and run statistical models that
surface **irregularities worth looking into**.

**What this is — and isn't.** This is an *exploratory screen*. Every number below is computed
over the same fixed slice of history, so we are making a great many comparisons at once and the
individual scores are inflated by that multiplicity. Nothing here is proof of wrongdoing; the
ranked table is a list of **leads to investigate**, with a Benjamini–Hochberg *q*-value attached
wherever a genuine statistical test exists so the multiple-testing burden is explicit.

**Method in one paragraph.** There is no reliable open historical strength rating for national
teams, so we build one in-repo: a chronological **Elo** engine over every men's World Cup match
(host nations get a home-field bonus at home only). Everything downstream — expected goals,
strength-of-schedule, host over/under-performance — hangs off those ratings. We then run five
detectors and fold their outputs into one comparable **anomaly score**.
""")

code(r"""
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.figsize": (9, 4.5), "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})
# Colour-blind-safe categorical palette (Okabe–Ito).
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
pd.set_option("display.max_colwidth", 90)

from worldcup_anomalies.fetch import load_data
from worldcup_anomalies.elo import compute_elo, pre_tournament_ratings

data = load_data()                 # men's-only, cached under data/raw/
elo_matches = compute_elo(data.matches)
print("Loaded men's World Cup data and computed Elo ratings.")
""")

md(r"""
## 1. What's in the data

The primary source is the [`jfjelstul/worldcup`](https://github.com/jfjelstul/worldcup)
normalized dataset (filtered to men's editions). Card/booking data only exists from **1970**
onward — that's a hard limit of the historical record, so the referee-discipline analysis is a
1970–2022 story. Referee *identity* and match results span the full 1930–2022.
""")

code(r"""
display(data.summary().set_index("table"))
n_t = data.tournaments["tournament_id"].nunique()
print(f"{n_t} men's tournaments, {data.tournaments.year.min()}–{data.tournaments.year.max()}")
card_years = pd.to_datetime(data.bookings.match_date).dt.year
print(f"Card data: {card_years.min()}–{card_years.max()}  |  {len(data.referees)} referees")
""")

md(r"""
## 2. The headline: irregularities worth looking into

`collect_anomalies` runs all five detectors and maps each onto a single, clipped `anomaly_score`
(a z-magnitude) so they can be ranked together. `q_value` is the Benjamini–Hochberg FDR value,
populated only where a real test exists (referee card rates, host card bias).

Read this as a triage list, top to bottom — then use the per-detector sections below to
understand *why* each item was flagged.
""")

code(r"""
from worldcup_anomalies.anomalies import collect_anomalies

ranked = collect_anomalies(data, elo_matches)
print(f"{len(ranked)} flagged items across "
      f"{ranked.category.nunique()} detectors\n")
display(ranked.head(25).set_index("rank")[
    ["category", "subject", "detail", "anomaly_score", "q_value"]
])
""")

code(r"""
# How the flags distribute across detectors and severity.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
cat_order = ranked.category.value_counts()
ax1.barh(cat_order.index[::-1], cat_order.values[::-1],
         color=[PAL[i % len(PAL)] for i in range(len(cat_order))])
ax1.set_title("Flags per detector"); ax1.set_xlabel("count")

for i, cat in enumerate(cat_order.index):
    sub = ranked[ranked.category == cat]
    ax2.scatter(sub.anomaly_score, [i] * len(sub), color=PAL[i % len(PAL)],
                alpha=0.75, s=35)
ax2.set_yticks(range(len(cat_order))); ax2.set_yticklabels(cat_order.index)
ax2.set_xlabel("anomaly score (clipped z)"); ax2.set_title("Score by detector")
ax2.invert_yaxis()
plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Detector — Upsets (results that defied strength)

A favourite thrashing a minnow is *statistically* surprising but not interesting. What's worth a
look is the reverse: a clearly weaker side (by pre-match Elo) **winning**. We rank those by the
Elo gap the result defied. This is a sanity check as much as a detector — it should rediscover
the famous shocks, and it does (South Korea 2-0 Germany 2018, Saudi Arabia beating Argentina
2022, Cameroon over Brazil, and the 2002 South Korea run whose refereeing was itself notorious).
""")

code(r"""
from worldcup_anomalies.models import score_surprise

ss = score_surprise(elo_matches, data.team_appearances)
upsets = ss[(ss.upset) & (ss.year >= 1958)].sort_values("elo_gap_defied", ascending=False)
show = upsets.head(12)[[
    "year", "match_name", "stage_name", "home_team_score", "away_team_score", "elo_gap_defied"
]]
display(show.reset_index(drop=True))

fig, ax = plt.subplots()
ax.scatter(ss.elo_home_pre - ss.elo_away_pre,
           ss.home_team_score - ss.away_team_score,
           s=14, alpha=0.3, color=PAL[0], label="all matches")
ax.scatter(upsets.elo_home_pre - upsets.elo_away_pre,
           upsets.home_team_score - upsets.away_team_score,
           s=28, color=PAL[1], label="upset")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("pre-match Elo gap (home − away)")
ax.set_ylabel("goal margin (home − away)")
ax.set_title("Results vs strength: upsets are the off-diagonal points")
ax.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Detector — "Convenient" group results

The archetype is the 1982 **Disgrace of Gijón**: West Germany beat Austria 1-0, a result that
sent *both* through at Algeria's expense, in a match played the day *after* Algeria had finished.
FIFA's response was to make final group matches kick off simultaneously from 1986 on.

The detector flags the last-played first-round group match whose result advanced **both** teams,
where a non-advancing team had already finished on an earlier day (an informational advantage).
Tellingly, it flags **only 1982** — the structure that made this possible was legislated away in
1986, and the detector independently rediscovers that.
""")

code(r"""
from worldcup_anomalies.models import detect_convenient_results

conv = detect_convenient_results(data.matches, data.group_standings)
display(conv[[
    "tournament_id", "match_name", "match_date", "score",
    "non_simultaneous", "eliminated_finished_earlier", "suspicion"
]])
print("These are the last two World Cups (both 1982) before simultaneous final "
      "group kickoffs were mandated in 1986.")
""")

md(r"""
## 5. Detector — Referee discipline

Two questions. **(a)** Do individual referees hand out far more (or fewer) cards than the match
context warrants? We model expected cards from era, knockout-vs-group, and how competitive the
match is (by Elo), then score each referee's standardized residual. **(b)** Do **hosts** get an
easier ride from officials — fewer cards than their opponents in the matches they play?
""")

code(r"""
from worldcup_anomalies.referees import referee_outliers, host_card_bias

ro = referee_outliers(data.matches, data.referee_appearances, data.bookings,
                      elo_matches, min_matches=5)
top = ro.reindex(ro.z.abs().sort_values(ascending=False).index).head(12)
display(top[["referee_name", "referee_country", "n_matches",
             "obs_cards", "exp_cards", "cards_per_match", "z"]].reset_index(drop=True))

fig, ax = plt.subplots(figsize=(9, 5))
t = top.iloc[::-1]
colors = [PAL[1] if z > 0 else PAL[0] for z in t.z]
ax.barh(t.referee_name + " (" + t.referee_country + ")", t.z, color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("standardized card residual  z  (+ card-happy · − lenient)")
ax.set_title("Referees whose card counts most deviate from expectation")
plt.tight_layout(); plt.show()
""")

code(r"""
per_t, host_bias = host_card_bias(data.matches, data.bookings)
print("HOST CARD BIAS (matches involving the host nation, 1970+):")
for k, v in host_bias.items():
    print(f"  {k}: {v}")
print(f"\nInterpretation: in decisive matches the host received FEWER cards than its opponent "
      f"{host_bias['share_host_fewer']:.0%} of the time "
      f"(sign-test p = {host_bias['sign_test_p']:.3f}). Suggestive of a home-crowd/officiating "
      f"tilt, though card counts also reflect that hosts are often the stronger, less-pressed side.")
""")

md(r"""
## 6. Detector — "Easy path" / seeding luck

For every team that reached the quarter-final or deeper, we measure the strength of the teams it
actually had to beat — the mean pre-tournament Elo of its opponents — and how many established
"powers" (top-quartile that edition) it faced. A run to the semi-final or final past unusually
weak opposition is a lucky/soft draw, *whether or not* the team was the host.

We compare within the round reached (finalists vs finalists) and skip 1930–1938, where every
team enters at the base rating and opponent strength is meaningless (Elo cold-start).
""")

code(r"""
from worldcup_anomalies.paths import easy_path_scores

ep = easy_path_scores(elo_matches, data.team_appearances)
finals = ep[ep.round_label == "Final"].sort_values("mean_opp_elo").head(6)
semis = ep[ep.round_label == "SF"].sort_values("mean_opp_elo").head(6)
print("Easiest routes to the FINAL:")
display(finals[["year", "team_name", "mean_opp_elo", "max_opp_elo",
                "n_powers_faced", "easiness_pct"]].reset_index(drop=True))
print("Easiest routes to the SEMI-FINAL:")
display(semis[["year", "team_name", "mean_opp_elo", "max_opp_elo",
               "n_powers_faced", "easiness_pct"]].reset_index(drop=True))

deep = ep[ep.round_rank >= 6].copy()
fig, ax = plt.subplots()
sc = ax.scatter(deep.mean_opp_elo, deep.n_powers_faced,
                c=deep.round_rank, cmap="viridis", s=45)
for _, r in pd.concat([finals.head(3), semis.head(3)]).iterrows():
    ax.annotate(f"{int(r.year)} {r.team_name}",
                (r.mean_opp_elo, r.n_powers_faced), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("mean opponent Elo (lower = easier)")
ax.set_ylabel("established powers faced")
ax.set_title("Strength of schedule for deep runs (SF/Final)")
plt.colorbar(sc, label="round rank (6=SF, 7=Final)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Detector — FIFA leadership lens (exploratory)

The most speculative view: map each tournament to the FIFA president in office and ask whether
over/under-performance clusters by era. **This is correlational and under-powered** — 22
tournaments across 7 presidents — so treat it as a conversation-starter, not a finding.

The honest result: host over-performance is *higher* in the early decades (when hosting carried a
huge travel/familiarity edge) and *lower* under Blatter/Infantino. The "corruption-era host bias"
hypothesis is **not** supported by the data.
""")

code(r"""
from worldcup_anomalies.leadership import (
    era_summary, host_overperformance, host_overperformance_permutation,
)

es = era_summary(data.tournaments, elo_matches, data.team_appearances, data.leadership)
display(es)

ho = host_overperformance(data.tournaments, elo_matches, data.team_appearances, data.leadership)
perm = host_overperformance_permutation(ho, ["João Havelange", "Sepp Blatter"], n_perm=20000)
print("Permutation test — host over-performance in the Havelange+Blatter era vs the rest:")
print(f"  observed difference: {perm['observed_diff']:+.2f} rounds   p = {perm['p_value']:.3f}")
print("  → no evidence hosts over-performed MORE in that era (if anything, less).")

fig, ax = plt.subplots(figsize=(10, 4))
ho_s = ho.sort_values("year")
colors = [PAL[hash(p) % len(PAL)] for p in ho_s.president]
ax.bar(ho_s.year.astype(str), ho_s.overperformance, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("host over-performance\n(rounds vs Elo expectation)")
ax.set_title("Did the host over- or under-perform? (colour = FIFA president)")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
""")

md(r"""
## 8. Takeaways

- The screen **rediscovers known cases**, which is the point of validating it: the 1982 Gijón
  match tops the convenient-result detector, the 2018 Germany and 2022 Saudi/Argentina shocks top
  the upset detector, and the 2002 South Korea run (notorious for its officiating) surfaces near
  the top of the combined table.
- The one **systemic signal with a real test behind it** is host card bias: hosts receive fewer
  cards than their opponents about 69% of the time (sign-test p ≈ 0.01, BH q ≈ 0.06). Worth a
  deeper, confounder-aware look — hosts are also often the stronger, more possession-dominant side.
- Individual **referee card rates** vary far more than context explains (e.g. very card-happy vs
  very lenient officials), with small-sample caveats — several survive BH correction.
- The **FIFA-leadership lens returns a null**: no era-clustering of host over-performance. Reported
  as-is, because a screen that only ever confirms its priors is worthless.

**Caveats.** Elo is inferred from World Cup matches only (no qualifiers/friendlies), so early
ratings are noisy. Card data starts in 1970. Above all this is a multiple-comparison screen: the
q-values quantify how much of the apparent signal survives that. Treat every row as a lead, not a
verdict.
""")

nb["cells"] = cells
out = Path(__file__).with_name("world_cup_irregularities.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
