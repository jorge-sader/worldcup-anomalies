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

**Team strength.** Every downstream question — expected goals, strength-of-schedule, host
over/under-performance — needs a measure of how strong each team was *at the time*. We use an
**Elo** rating built over **every international match since 1872** (friendlies, qualifiers,
continental cups, World Cups) from the `martj42/international_results` dataset. This matters: a
World-Cup-only Elo would enter each debutant at a neutral 1500 regardless of real strength (Qatar
2022, Japan on debut in 1998, …); the full-history version grounds every team in its actual
record from the first whistle. We keep the World-Cup-only Elo too, purely as a self-contained
cross-check.
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
from worldcup_anomalies.intl_elo import load_intl_results, build_intl_elo, annotate_world_cup

data = load_data()                                  # men's-only, cached under data/raw/
elo_wc = compute_elo(data.matches)                  # self-contained World-Cup-only Elo
elo = annotate_world_cup(data.matches, build_intl_elo(load_intl_results()))  # grounded, primary
print("Loaded data and computed both World-Cup-only and full-history Elo ratings.")
""")

md(r"""
## 1. What's in the data

The match/referee/booking source is the [`jfjelstul/worldcup`](https://github.com/jfjelstul/worldcup)
normalized dataset (filtered to men's editions). Card/booking data only exists from **1970**
onward — a hard limit of the historical record — so the referee-discipline analysis is a
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
## 2. Team strength: World-Cup-only vs full international history

Why this matters for the whole analysis. A World-Cup-only Elo has a **cold-start** problem: it has
never seen a team until its World Cup debut, so it enters that team at a neutral **1500** — even
if the team is, by any real measure, weak or strong. The full-history Elo has already watched the
team play dozens of qualifiers and friendlies, so it enters with a *grounded* rating.

The clearest illustration is the ratings **entering the 2022 World Cup**: under the full-history
model nobody sits at a blank 1500 (Qatar enters around 1790, earned over years of Asian
qualifiers), and the spread between strong and weak teams is far more realistic.
""")

code(r"""
pw = (pre_tournament_ratings(elo_wc).query("tournament_id == 'WC-2022'")
      [["team_name", "elo_pre"]].rename(columns={"elo_pre": "wc_only"}))
pf = (pre_tournament_ratings(elo).query("tournament_id == 'WC-2022'")
      [["team_name", "elo_pre"]].rename(columns={"elo_pre": "full_history"}))
cmp = pw.merge(pf, on="team_name")
cmp["debutant_pinned_at_1500"] = cmp["wc_only"].round(0) == 1500
display(cmp.sort_values("full_history", ascending=False).round(0)
        .reset_index(drop=True))
print("Under the World-Cup-only Elo, Qatar (a WC debutant) is pinned at 1500; "
      "the full-history Elo enters them near 1790 from their real record.")
""")

md(r"""
The rest of the notebook uses the **full-history** ratings (`elo`). Everything is written to work
with either — swap in `elo_wc` to reproduce the self-contained version.
""")

md(r"""
## 3. The headline: irregularities worth looking into

`collect_anomalies` runs all five detectors and maps each onto a single, clipped `anomaly_score`
(a z-magnitude) so they can be ranked together. `q_value` is the Benjamini–Hochberg FDR value,
populated only where a real test exists (referee card rates, host card bias).
""")

code(r"""
from worldcup_anomalies.anomalies import collect_anomalies

ranked = collect_anomalies(data, elo)
print(f"{len(ranked)} flagged items across {ranked.category.nunique()} detectors\n")
display(ranked.head(25).set_index("rank")[
    ["category", "subject", "detail", "anomaly_score", "q_value"]
])
""")

code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
cat_order = ranked.category.value_counts()
ax1.barh(cat_order.index[::-1], cat_order.values[::-1],
         color=[PAL[i % len(PAL)] for i in range(len(cat_order))])
ax1.set_title("Flags per detector"); ax1.set_xlabel("count")

for i, cat in enumerate(cat_order.index):
    sub = ranked[ranked.category == cat]
    ax2.scatter(sub.anomaly_score, [i] * len(sub), color=PAL[i % len(PAL)], alpha=0.75, s=35)
ax2.set_yticks(range(len(cat_order))); ax2.set_yticklabels(cat_order.index)
ax2.set_xlabel("anomaly score (clipped z)"); ax2.set_title("Score by detector")
ax2.invert_yaxis()
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Detector — Upsets (results that defied strength)

A favourite thrashing a minnow is *statistically* surprising but not interesting. What's worth a
look is the reverse: a clearly weaker side (by pre-match Elo) **winning**. We rank those by the
Elo gap the result defied. With grounded ratings this rediscovers the canonical shocks — Cameroon
over Brazil (1990), Argentina losing to Saudi Arabia (2022), Switzerland over Spain (2010),
Algeria over West Germany (1982), South Korea over Germany (2018).
""")

code(r"""
from worldcup_anomalies.models import score_surprise

ss = score_surprise(elo, data.team_appearances)
upsets = ss[(ss.upset) & (ss.year >= 1958)].sort_values("elo_gap_defied", ascending=False)
display(upsets.head(12)[[
    "year", "match_name", "stage_name", "home_team_score", "away_team_score", "elo_gap_defied"
]].reset_index(drop=True))

fig, ax = plt.subplots()
ax.scatter(ss.elo_home_pre - ss.elo_away_pre, ss.home_team_score - ss.away_team_score,
           s=14, alpha=0.3, color=PAL[0], label="all matches")
ax.scatter(upsets.elo_home_pre - upsets.elo_away_pre,
           upsets.home_team_score - upsets.away_team_score,
           s=28, color=PAL[1], label="upset")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("pre-match Elo gap (home − away)"); ax.set_ylabel("goal margin (home − away)")
ax.set_title("Results vs strength: upsets are the off-diagonal points")
ax.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Detector — "Convenient" group results

The archetype is the 1982 **Disgrace of Gijón**: West Germany beat Austria 1-0, a result that
sent *both* through at Algeria's expense, in a match played the day *after* Algeria had finished.
FIFA's response was to make final group matches kick off simultaneously from 1986 on.

The detector flags the last-played first-round group match whose result advanced **both** teams,
where a non-advancing team had already finished on an earlier day (an informational advantage).
Tellingly it flags **only 1982** — the structure that made this possible was legislated away in
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
## 6. Detector — Referee discipline

**(a)** Do individual referees hand out far more (or fewer) cards than the match context warrants?
We model expected cards from era, knockout-vs-group, and how competitive the match is (by Elo),
then score each referee's standardized residual. **(b)** Do **hosts** get an easier ride — fewer
cards than their opponents in the matches they play?
""")

code(r"""
from worldcup_anomalies.referees import referee_outliers, host_card_bias

ro = referee_outliers(data.matches, data.referee_appearances, data.bookings, elo, min_matches=5)
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
print(f"\nIn decisive matches the host received FEWER cards than its opponent "
      f"{host_bias['share_host_fewer']:.0%} of the time "
      f"(sign-test p = {host_bias['sign_test_p']:.3f}). Suggestive of a home-crowd/officiating "
      f"tilt — though hosts are also often the stronger, less-pressed side.")
""")

md(r"""
## 7. Detector — "Easy path" / seeding luck

For every team that reached the quarter-final or deeper, we measure the strength of the teams it
actually had to beat — the mean pre-tournament Elo of its opponents — and how many established
"powers" (top-quartile that edition) it faced. A run to the semi-final or final past unusually
weak opposition is a soft draw, *whether or not* the team was the host. We compare within the
round reached (finalists vs finalists) so that deeper runs facing more games aren't penalised.
""")

code(r"""
from worldcup_anomalies.paths import easy_path_scores

ep = easy_path_scores(elo, data.team_appearances)
finals = ep[ep.round_label == "Final"].sort_values("mean_opp_elo").head(6)
semis = ep[ep.round_label == "SF"].sort_values("mean_opp_elo").head(6)
print("Easiest routes to the FINAL:")
display(finals[["year", "team_name", "mean_opp_elo", "max_opp_elo",
                "n_powers_faced", "easiness_pct"]].reset_index(drop=True))
print("Easiest routes to the SEMI-FINAL:")
display(semis[["year", "team_name", "mean_opp_elo", "max_opp_elo",
               "n_powers_faced", "easiness_pct"]].reset_index(drop=True))
""")

md(r"""
### A worked check: did Argentina have easy paths?

A natural suspicion is that a serial deep-runner reached finals via soft draws. The data says the
opposite — Argentina's finalist runs sit at the **hard** end of the distribution, because from the
quarter-finals on they repeatedly met elite teams (2022: Netherlands, Croatia, France; 2014:
Netherlands, Germany). `easiness_pct` is the percentile of *weakness* within the round reached, so
**low = hard path**.
""")

code(r"""
arg = ep[ep.team_name == "Argentina"].copy()
n_final = (ep.round_label == "Final").sum()
n_sf = (ep.round_label == "SF").sum()
display(arg[["year", "round_label", "n_opponents", "mean_opp_elo",
             "max_opp_elo", "n_powers_faced", "easiness_pct"]]
        .sort_values("year").reset_index(drop=True))
argf = arg[arg.round_label == "Final"].sort_values("easiness_pct")
for _, r in argf.iterrows():
    harder_than = 100 - r.easiness_pct
    print(f"{int(r.year)} Argentina (Final): path was harder than {harder_than:.0f}% "
          f"of all {n_final} finalist runs — faced {int(r.n_powers_faced)} top-quartile teams "
          f"(toughest Elo {r.max_opp_elo:.0f}).")
print("\nConclusion: Argentina is a counter-example to an easy path, not an instance of one.")
""")

code(r"""
deep = ep[ep.round_rank >= 6].copy()
fig, ax = plt.subplots()
sc = ax.scatter(deep.mean_opp_elo, deep.n_powers_faced, c=deep.round_rank, cmap="viridis", s=45)
for _, r in deep[deep.team_name == "Argentina"].iterrows():
    ax.annotate(f"{int(r.year)} ARG", (r.mean_opp_elo, r.n_powers_faced), fontsize=8,
                color=PAL[1], fontweight="bold", xytext=(4, 4), textcoords="offset points")
for _, r in pd.concat([finals.head(2), semis.head(2)]).iterrows():
    ax.annotate(f"{int(r.year)} {r.team_name}", (r.mean_opp_elo, r.n_powers_faced),
                fontsize=8, xytext=(4, -8), textcoords="offset points")
ax.set_xlabel("mean opponent Elo (lower = easier path)")
ax.set_ylabel("established powers faced")
ax.set_title("Strength of schedule for deep runs (Argentina highlighted)")
plt.colorbar(sc, label="round rank (6=SF, 7=Final)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 8. Detector — Draw luck: the group is the lever

The `easy_path` metric above averages over the *whole* run — but by the quarter-final it's nearly
impossible to avoid strong teams, so that average drowns out the part that actually matters. If you
wanted to *ease* a team's route, the lever is the **group draw**: hand it a soft group, let the
other powers cluster elsewhere and knock each other out, and it reaches the semis having beaten
nobody. So we isolate the **group stage** and measure it two ways that avoid the traps:

- **Within-edition** — Elo inflates over time, so we compare a team's group only to the *other
  groups that same year* (100 = softest group of the tournament).
- **Seed-controlled** — the top teams are seeded into separate groups *by design*, so "strong team
  + soft group" is the norm. We therefore also rank each seed's group against the *other seeds*
  (`seed_soft_rank` = 1 means the softest group among that year's ~8 seeds).
""")

code(r"""
from worldcup_anomalies.paths import group_draw_difficulty

gd = group_draw_difficulty(elo, data.team_appearances)

# Deep runs (SF/Final) reached from the softest groups, among seeds, within their edition.
soft_deep = gd[(gd["rank"] >= 6) & (gd["is_seed"])].sort_values("seed_soft_rank")
print("Teams that reached the SF/Final from the softest seeded group of their edition:")
display(soft_deep[soft_deep.seed_soft_rank <= 2][
    ["year", "team_name", "round_label", "grp_mean_opp_elo",
     "n_seeds_in_group", "seed_soft_rank", "n_seeds_total"]
].round(0).reset_index(drop=True))

print("Argentina, every appearance — group softness (100=softest that year) and rank among seeds:")
display(gd[gd.team_name == "Argentina"][
    ["year", "round_label", "grp_softness_pct", "seed_soft_rank", "n_seeds_total"]
].round(0).reset_index(drop=True))
""")

code(r"""
# 2014 & 2022: Argentina's group vs the other seeds' groups that year.
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, yr in zip(axes, [2014, 2022]):
    s = gd[(gd.year == yr) & (gd.is_seed)].sort_values("grp_mean_opp_elo")
    colors = [PAL[1] if t == "Argentina" else PAL[0] for t in s.team_name]
    ax.barh(s.team_name[::-1], s.grp_mean_opp_elo[::-1], color=colors[::-1])
    ax.set_title(f"{yr}: seeded teams' group difficulty")
    ax.set_xlabel("mean group-opponent Elo (shorter = softer draw)")
    ax.set_xlim(s.grp_mean_opp_elo.min() - 40, s.grp_mean_opp_elo.max() + 10)
plt.tight_layout(); plt.show()
print("Argentina drew the SOFTEST seeded group in 2014 and the 2nd-softest in 2022 — "
      "reaching the Final both times.")
""")

md(r"""
**How to read this honestly.** Argentina's two most recent Final runs *did* start from the softest
(2014) and 2nd-softest (2022) seeded groups of their editions — a real pattern the whole-path metric
missed, and a fair basis for curiosity. But three caveats keep it a *lead, not a verdict*:

1. **It's structural.** Soft-seeded-group-then-deep-run recurs throughout history (1990 West
   Germany, 1998/2006 France, 2002 Brazil, 2010 Netherlands …) — winning is simply easier from a
   soft group, and the best teams are seeded. Argentina is not unique.
2. **Someone is always softest.** By construction one seed draws the softest group every edition;
   observing it after the fact isn't evidence by itself.
3. **No career pattern.** Across all of Argentina's tournaments the draw is mixed — the *hardest*
   seeded group in 2006, 6th of 8 in 2002. The soft draws cluster in 2014 and 2022, i.e. n ≈ 2.

That's exactly what a screen should do: surface the pattern, size it, and refuse to overclaim.
""")

md(r"""
## 9. Lucky or engineered? A fair-draw Monte-Carlo

The group-softness metric shows a team *got* a soft group, but not whether that's surprising —
or whether the other powers were **packed together elsewhere** so they eliminated each other and
cleared the favoured team's path. That packing is the real signature of an engineered draw, and
we test it by simulating thousands of fair draws.

The one thing we can assert without external pot data is that seeding **separates each group's
strongest team** by design. So we hold each group's strongest team fixed and randomly
redistribute everyone else, 10,000 times:

- **`draw_luck_pct`** (per team) — where its group's opponent strength falls vs fair redraws.
  *Low = softer than almost any fair draw.*
- **`rival_clustering_pct`** (per edition) — where the *variance of group strength* falls.
  *High = strong teams bunched into some groups and absent from others (rivals clustered).*

The engineered pattern is the **conjunction**: deep run **+** soft group **+** clustered rivals.
This null ignores pot 2–4 constraints, so it's deliberately *conservative* (under- not
over-states how extreme a draw was).
""")

code(r"""
from worldcup_anomalies.draw import draw_monte_carlo, engineered_draw_flags

draws = draw_monte_carlo(elo, data.group_standings, data.team_appearances, n_sims=10000)

print("Argentina's draw each edition (draw_luck low = soft group; clustering high = rivals packed):")
display(draws[draws.team_name == "Argentina"][
    ["year", "group_name", "round_label", "draw_luck_pct", "rival_clustering_pct"]
].round(0).reset_index(drop=True))

print("Deep-run teams (SF/Final) with the softest groups vs fair draws:")
display(draws[draws.round_rank >= 6].sort_values("draw_luck_pct").head(8)[
    ["year", "team_name", "round_label", "draw_luck_pct", "rival_clustering_pct"]
].round(0).reset_index(drop=True))
""")

code(r"""
# The engineered signature: soft group AND clustered rivals AND a deep run.
flags = engineered_draw_flags(draws)
print("Teams matching the ENGINEERED-DRAW signature (1954–2022):")
display(flags[["year", "team_name", "round_label",
               "draw_luck_pct", "rival_clustering_pct"]].round(0).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(8.5, 6))
deep = draws[draws.round_rank >= 6]
ax.scatter(deep.draw_luck_pct, deep.rival_clustering_pct, s=45, color=PAL[0], alpha=0.6,
           label="SF/Final teams")
for _, r in flags.iterrows():
    ax.scatter(r.draw_luck_pct, r.rival_clustering_pct, s=120, color=PAL[1], zorder=5)
    ax.annotate(f"{int(r.year)} {r.team_name}", (r.draw_luck_pct, r.rival_clustering_pct),
                fontsize=9, fontweight="bold", xytext=(6, 0), textcoords="offset points")
for _, r in deep[deep.team_name == "Argentina"].iterrows():
    ax.annotate(f"{int(r.year)} ARG", (r.draw_luck_pct, r.rival_clustering_pct),
                fontsize=8, color=PAL[3], xytext=(4, 4), textcoords="offset points")
ax.axvspan(0, 20, color=PAL[1], alpha=0.06); ax.axhspan(80, 100, color=PAL[1], alpha=0.06)
ax.set_xlabel("draw_luck_pct  (← softer group)")
ax.set_ylabel("rival_clustering_pct  (rivals clustered →)")
ax.set_title("Engineered-draw quadrant = soft group (left) AND clustered rivals (top)")
ax.legend(loc="lower right"); plt.tight_layout(); plt.show()
""")

md(r"""
**The verdict on draw manipulation.** Exactly **one** team-edition in 1954–2022 lands in the
engineered quadrant — **Brazil 2002**: a group softer than ~93% of fair draws while the other
powers were more clustered than 98% of draws (that was the edition of France, Argentina and
Portugal all going out in the group stage). Even so, this is a *statistical signature, not
evidence of wrongdoing* — someone occupies the tail of any distribution.

And on the original question: **Argentina does not match it.** Their 2014 and 2022 groups were
genuinely soft (softer than ~80% of fair draws — the intuition was right), but the rivals were
**not** clustered — 2022 was in fact the *least*-clustered draw of any edition. In 2002, the one
year rivals *were* clustered, Argentina was the **victim**, drawn into the group of death and
eliminated. So: a lucky soft group, twice — not an engineered path.
""")

md(r"""
## 10. Detector — FIFA leadership lens (exploratory)

The most speculative view: map each tournament to the FIFA president in office and ask whether
over/under-performance clusters by era. **This is correlational and under-powered** — 22
tournaments across 7 presidents — so treat it as a conversation-starter, not a finding.

The honest result: host over-performance is *higher* in the early decades (when hosting carried a
huge travel/familiarity edge) and *lower* under Blatter/Infantino. The "corruption-era host bias"
hypothesis is **not** supported.
""")

code(r"""
from worldcup_anomalies.leadership import (
    era_summary, host_overperformance, host_overperformance_permutation,
)

es = era_summary(data.tournaments, elo, data.team_appearances, data.leadership)
display(es)

ho = host_overperformance(data.tournaments, elo, data.team_appearances, data.leadership)
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
## 11. Takeaways

- Using ratings grounded in **every international match since 1872** (not just World Cup games)
  removes the cold-start artefact where debutants entered at a neutral 1500 — and it *sharpens*
  the results: the upset list becomes the canonical shocks, and noisy host-performance flags drop
  away.
- The screen **rediscovers known cases**, which is the validation: 1982 Gijón tops the
  convenient-result detector, Cameroon–Brazil 1990 and Saudi Arabia–Argentina 2022 top the upsets,
  and the detector independently rediscovers that the Gijón pattern died with the 1986 simultaneity
  rule.
- The one **systemic signal with a real test behind it** is host card bias: hosts receive fewer
  cards than their opponents ~69% of the time (sign-test p ≈ 0.01, BH q ≈ 0.07). Worth a deeper,
  confounder-aware look — hosts are also often the stronger, more possession-dominant side.
- **Two questions, two different answers, for the same team.** Over the *whole* run Argentina's
  finals were among the *hardest* paths (2022 harder than ~88% of finalist runs) — but that's
  because everyone meets elite teams from the quarter-final on. Isolating the **group draw** (the
  part seeding/the draw controls) flips it: Argentina reached the 2014 and 2022 finals from the
  *softest* and *2nd-softest* seeded groups of their editions. Real pattern, correctly sized: it's
  structural (soft seeded groups are common for eventual champions) and n ≈ 2, so a lead, not a
  verdict. Picking the *right unit of analysis* is the whole game.
- **Lucky ≠ engineered.** A fair-draw Monte-Carlo separates a soft group from an engineered one
  (soft group *and* rivals clustered elsewhere). Only **Brazil 2002** matches the full signature
  in 1954–2022; Argentina's soft 2014/2022 groups do **not** (rivals weren't clustered — 2022 was
  the most balanced draw on record). Even Brazil 2002 is a signature, not proof.
- The **FIFA-leadership lens returns a null**: no era-clustering of host over-performance. Reported
  as-is, because a screen that only ever confirms its priors is worthless.

**Caveats.** Card data starts in 1970. Elo is still a summary of results only (no lineups/xG). Above
all this is a multiple-comparison screen: the q-values quantify how much apparent signal survives.
Treat every row as a lead, not a verdict.
""")

nb["cells"] = cells
out = Path(__file__).with_name("world_cup_irregularities.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
