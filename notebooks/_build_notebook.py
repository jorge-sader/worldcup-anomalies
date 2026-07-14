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
# --- Environment bootstrap ---------------------------------------------------------
# This notebook imports the `worldcup_anomalies` package. Locally it's installed via uv;
# on Colab (or any fresh kernel) it isn't present, so install it straight from GitHub.
try:
    import worldcup_anomalies  # noqa: F401
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/jorge-sader/worldcup-anomalies.git",
    ])
    import worldcup_anomalies  # noqa: F401
print("worldcup_anomalies is ready — data is fetched from the web on first use.")
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
elo_timeline = build_intl_elo(load_intl_results())  # full-history Elo, per-team over 1872–2026
elo = annotate_world_cup(data.matches, elo_timeline)  # grounded ratings per WC match (primary)
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
## 2. General context: tournaments, paths, and how the Elo works

Before the anomaly detectors, three orienting views: a reference table of every World Cup, a
visual comparison of how hard each champion's road to the title was, and a plain-language primer
on the Elo rating that drives everything downstream.
""")

md(r"""
### 2a. Every World Cup at a glance

Hosts, field size, games, penalty shootouts, cards, and the top-four finishers. Card columns are
**0 before 1970** — booking data simply does not exist further back. The in-progress **2026**
edition is tracked separately in §2f (it uses a different, live source and the new 48-team format).
""")

code(r"""
from worldcup_anomalies.reports import tournament_overview, team_path, champions_paths

overview = tournament_overview(data)
display(
    overview.style
    .format({"teams": "{:.0f}", "games": "{:.0f}", "penalty_shootouts": "{:.0f}",
             "yellow": "{:.0f}", "red": "{:.0f}"})
    .background_gradient(subset=["games", "yellow", "red"], cmap="Blues")
    .set_caption("Men's FIFA World Cups, 1930–2022")
    .hide(axis="index")
)
""")

md(r"""
### 2b. Road to the title — comparing champions' paths

Each row is a champion; each column a game in order. **Colour = the opponent's strength as a
*within-edition percentile*** (100 = one of the strongest teams that tournament) — an era-fair
measure, since raw Elo inflates over the decades. The knockout rounds are boxed: blue = R16,
orange = QF, red = SF/Final. This "path" view makes strength-of-schedule legible in a way a table
of numbers can't — and the same helper drives the interactive team explorer just below.
""")

code(r"""
import matplotlib.patches as mpatches

STAGE_EDGE = {"R32": "#56B4E9", "R16": "#0072B2", "QF": "#E69F00",
              "SF": "#D55E00", "FIN": "#D55E00"}

def plot_paths_grid(entries, title, row_h=0.62):
    # entries: list of (row_label, path_df). Colour = opponent within-edition strength pct.
    maxg = max(len(p) for _, p in entries)
    nrows = len(entries)
    E = np.full((nrows, maxg), np.nan)
    labels = np.empty((nrows, maxg), dtype=object); labels[:] = ""
    stages = np.empty((nrows, maxg), dtype=object); stages[:] = ""
    ylabels = []
    for i, (lab, p) in enumerate(entries):
        ylabels.append(lab)
        for r in p.itertuples():
            E[i, r.game_no - 1] = r.opponent_strength_pct
            labels[i, r.game_no - 1] = f"{r.opponent}\n{r.result[:1].upper()}"
            stages[i, r.game_no - 1] = r.stage_short
    fig, ax = plt.subplots(figsize=(1.5 * maxg + 3, max(2.6, row_h * nrows + 1)))
    im = ax.imshow(E, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    for i in range(nrows):
        for j in range(maxg):
            if not labels[i, j]:
                continue
            v = E[i, j]
            tc = "white" if (not np.isnan(v) and v > 62) else "black"
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=7, color=tc)
            edge = STAGE_EDGE.get(stages[i, j])
            if edge:
                lw = 3 if stages[i, j] in ("SF", "FIN") else 2
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                edgecolor=edge, lw=lw))
    ax.set_xticks(range(maxg)); ax.set_xticklabels([f"Game {k+1}" for k in range(maxg)])
    ax.set_yticks(range(nrows)); ax.set_yticklabels(ylabels)
    ax.set_title(title, fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("opponent strength\n(within-edition pct)")
    legend = [mpatches.Patch(edgecolor=c, facecolor="none", label=s, linewidth=2)
              for s, c in [("R16", "#0072B2"), ("QF", "#E69F00"), ("SF / Final", "#D55E00")]]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout()
    return fig, ax

champ_years = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
champ_entries = []
for yr in champ_years:
    row = data.tournaments[data.tournaments.year == yr]
    tid, champ = row.tournament_id.iloc[0], row.winner.iloc[0]
    champ_entries.append((f"{yr}  {champ}", team_path(data, tid, champ, elo)))
plot_paths_grid(champ_entries,
                "Road to the title — colour = opponent strength (within-edition percentile)")
plt.show()
""")

md(r"""
### 2c. Explore any team's historical path (interactive)

Pick a team from the alphabetical dropdown to see **its** road through *every* World Cup it has
played — one row per edition, same encoding as above (colour = opponent strength percentile,
knockout rounds boxed). This is a team's historical strength-of-schedule at a glance: soft groups
show up as pale left-hand cells, deep runs as long rows ending in red knockout boxes.

Same-nation successor names are folded into **continuous entries**: "Germany" spans its West
Germany years, "Russia" the Soviet Union, "Serbia" folds in Serbia & Montenegro — each row still
labelled with the name used in that era (e.g. `1974 · West Germany`). East Germany, Yugoslavia and
Czechoslovakia are kept as their own separate entries.

*The dropdown needs a live kernel* (open in Colab, or run locally). On the static GitHub/nbviewer
view, the **Germany** default is rendered below as a still image (note the West Germany rows folded
in); open in Colab to switch teams, or just call `show_team_history("Brazil")` in a cell.
""")

code(r"""
import ipywidgets as widgets
from worldcup_anomalies.reports import all_teams, team_history, canonical_name
from worldcup_anomalies.worldcup_2026 import (
    load_2026, pre_tournament_strength_2026, team_path_2026,
)

# The in-progress 2026 edition (results-only) is appended as a final row where applicable.
m2026 = load_2026()
pct2026 = pre_tournament_strength_2026(elo_timeline, m2026)
teams2026 = set(m2026["home_team"]) | set(m2026["away_team"])

def show_team_history(team):
    entries = team_history(data, team, elo)
    if canonical_name(team) in teams2026:
        entries.append(("2026 · in progress", team_path_2026(m2026, team, pct2026)))
    if not entries:
        print(f"{team}: no World Cup appearances on record."); return
    plot_paths_grid(
        entries,
        f"{team} — every World Cup path (colour = opponent strength, within-edition percentile)",
    )
    plt.show()

# Static default so the chart is visible even on non-interactive viewers (GitHub/nbviewer):
show_team_history("Germany")

# Live dropdown (needs a running kernel — Colab or local Jupyter); includes 2026 debutants:
widgets.interact(
    show_team_history,
    team=widgets.Dropdown(options=sorted(set(all_teams(data)) | teams2026),
                          value="Germany", description="Team:"),
);
""")

md(r"""
### 2d. How the Elo is calculated — a worked example

The rating is simple arithmetic applied after every match:

1. **Expected result** from the rating gap: `E = 1 / (1 + 10^((opp − you)/400))` — level ratings
   give `E = 0.5`; a 400-point edge gives ≈ 0.91.
2. **Actual result**: win = 1, draw = 0.5, loss = 0.
3. **Margin multiplier** `G`: 1 for a 1-goal win, 1.5 for 2, `(11+margin)/8` for 3+.
4. **Update**: `Δ = K · G · (actual − expected)` with `K = 40`; you gain Δ, the opponent loses it.

Below, three real matches worked end to end (World-Cup-only Elo, so numbers sit near the 1500
baseline and are easy to follow). Note how an upset moves ratings far more than an expected result.
""")

code(r"""
from worldcup_anomalies.elo import _gd_multiplier, K_FACTOR

def work_match(match_name, year):
    tid = data.tournaments[data.tournaments.year == year].tournament_id.iloc[0]
    sel = elo_wc[(elo_wc.tournament_id == tid) & (elo_wc.match_name == match_name)]
    assert len(sel) == 1, f"match not found: {match_name} ({year})"
    r = sel.iloc[0]
    exp_h = r.elo_prob_home
    actual_h = 1.0 if r.home_team_score > r.away_team_score else (
        0.0 if r.home_team_score < r.away_team_score else 0.5)
    mult = _gd_multiplier(r.home_team_score - r.away_team_score)
    delta = K_FACTOR * mult * (actual_h - exp_h)
    return {
        "match": f"{match_name} ({year})",
        "score": f"{int(r.home_team_score)}–{int(r.away_team_score)}",
        "home_pre": round(r.elo_home_pre), "away_pre": round(r.elo_away_pre),
        "P(home)": round(exp_h, 2), "actual": actual_h, "G(margin)": mult,
        "Δ_home": round(delta, 1),
        "home_post": round(r.elo_home_pre + delta), "away_post": round(r.elo_away_pre - delta),
    }

worked = pd.DataFrame([
    work_match("Argentina vs Saudi Arabia", 2022),   # huge upset (favourite lost)
    work_match("Argentina vs Croatia", 2022),        # strong favourite delivers
    work_match("Brazil vs Germany", 2014),           # 1–7: margin multiplier bites
])
display(worked.set_index("match"))
print("Argentina (heavy favourite) LOST to Saudi Arabia → a big negative Δ_home, with the "
      "same points gained by low-rated Saudi Arabia. Argentina beating Croatia as favourites "
      "→ small Δ. Germany's 7–1 → the margin multiplier amplifies an already-expected win.")
""")

md(r"""
### 2e. Ratings over time

The full-history engine tracks every national team from 1872. Here are a few traditional powers —
you can see eras rise and fade (Hungary's 1950s peak, Brazil's long dominance, Spain's late-2000s
surge), which is exactly the signal the strength-of-schedule and upset detectors lean on.
""")

code(r"""
traj_teams = ["Brazil", "Germany", "Argentina", "France", "Italy", "Spain", "Hungary"]
fig, ax = plt.subplots(figsize=(11, 5))
for i, tm in enumerate(traj_teams):
    if tm not in elo_timeline:
        continue
    dates, ratings = elo_timeline[tm]
    yrs = pd.to_datetime(dates).year
    keep = yrs >= 1920
    ax.plot(yrs[keep], np.array(ratings)[keep], label=tm, color=PAL[i % len(PAL)], lw=1.4)
ax.axhline(1500, color="k", lw=0.6, ls="--", alpha=0.5)
ax.set_xlabel("year"); ax.set_ylabel("Elo rating (full international history)")
ax.set_title("How team strength evolves — full-history Elo, 1920–2022")
ax.legend(ncol=4, fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
### 2f. The 2026 World Cup — live tracker (in progress)

A bonus view of the tournament being played *right now*. This comes from the live-updated
`martj42/international_results` feed (not the jfjelstul dataset, which ends at 2022) and is
**results-only**: no referee or card data exists yet, standings aren't final, and it's the first
**48-team** edition (12 groups → round of 32 → …). For those reasons it is kept **out of the
anomaly detectors** — the historical formats they assume don't apply. Stages are inferred from the
fixed bracket; re-run to refresh as later rounds are played.
""")

code(r"""
from worldcup_anomalies.worldcup_2026 import (
    load_2026, summary_2026, quarterfinalists_2026,
    pre_tournament_strength_2026, team_path_2026,
)

m26 = load_2026()
s26 = summary_2026(m26)
print(f"2026 World Cup — {s26['games_played']}/{s26['games_total']} games played; "
      f"latest completed round: {s26['latest_stage']}.")
print(f"Advanced from the {s26['latest_stage']}: {', '.join(s26['advanced_from_latest'])}")

# A tournaments-table-style row for 2026 (standings still TBD).
row26 = pd.DataFrame([{
    "Year": 2026, "Host": "Canada · Mexico · USA", "Teams": s26["teams"],
    "Games (played)": s26["games_played"], "Cards": "—  (no data yet)",
    "Status": f"in progress — {s26['latest_stage']}",
}])
display(row26.style.hide(axis="index"))
""")

code(r"""
# The deepest 2026 runs (teams that reached the quarter-finals), same path encoding as above.
pct26 = pre_tournament_strength_2026(elo_timeline, m26)
qf26 = quarterfinalists_2026(m26)
entries26 = [(t, team_path_2026(m26, t, pct26)) for t in qf26]
entries26.sort(key=lambda e: e[1]["opponent_strength_pct"].mean())  # easiest schedule first
plot_paths_grid(
    entries26,
    "2026 quarter-finalists' paths so far (colour = opponent strength, within-edition percentile)",
)
plt.show()
print("Read like the champions chart above — pale rows = soft schedules. The pattern from 2014 "
      "and 2022 recurs: Argentina's 2026 route so far is the palest of all eight quarter-finalists "
      "(lowest mean opponent strength) — a soft draw again, worth revisiting once the draw-luck "
      "Monte-Carlo can be run on the completed 48-team bracket.")
""")

md(r"""
## 3. Team strength: World-Cup-only vs full international history

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
## 4. The headline: irregularities worth looking into

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
## 5. Detector — Upsets (results that defied strength)

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
## 6. Detector — "Convenient" group results

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
## 7. Detector — Referee discipline

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
## 8. Detector — "Easy path" / seeding luck

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
### The group / knockout split

A whole-path average hides the *shape* of a run. Splitting strength-of-schedule into the group
phase vs the knockout phase makes the "coast then earn it" profile explicit: `group_softness_pct`
(100 = softest group that year) against `ko_toughness_pct` (100 = toughest knockout opponents),
with `split_index` high when both hold. The knockout mean includes the round of 16, so a run
that's only brutal in the final match or two scores moderately.
""")

code(r"""
from worldcup_anomalies.paths import phase_strength_split

split = phase_strength_split(elo, data.team_appearances)
deep_split = split[split["rank"] >= 6]
print("Purest 'soft group, brutal knockout' profiles among SF/Final teams:")
display(deep_split.sort_values("split_index", ascending=False).head(8)[
    ["year", "team_name", "round_label", "group_softness_pct",
     "ko_toughness_pct", "split_index"]
].round(0).reset_index(drop=True))

print("Argentina — group vs knockout, each knockout appearance:")
display(split[split.team_name == "Argentina"].sort_values("year")[
    ["year", "round_label", "group_opp_elo", "ko_max_opp_elo",
     "group_softness_pct", "ko_toughness_pct", "split_index"]
].round(0).reset_index(drop=True))
print("Argentina's 2014 & 2022 groups were among the softest (softness ~88), but knockout "
      "toughness was only moderate — the R16 (Switzerland, Australia) was beatable and the "
      "brutality was concentrated in the last one or two games. A soft group, not a brutal path.")
""")

md(r"""
## 9. Detector — Draw luck: the group is the lever

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
## 10. Lucky or engineered? A fair-draw Monte-Carlo

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
### Draw softness by World Cup — the myth-buster table

One row per tournament: how soft the **champion's** group was (rank within its field), how
**balanced vs clustered** the whole draw was, and — the decisive column — *who actually drew the
softest group that year* and whether they won. Use it to sanity-check any "team X had a rigged
easy draw" claim directly.
""")

code(r"""
from worldcup_anomalies.draw import edition_draw_summary

summary = edition_draw_summary(elo, data.group_standings, data.team_appearances, data.tournaments)
display(summary.style.format({
    "champ_group_softness_pct": "{:.0f}", "rival_clustering_pct": "{:.0f}",
}).hide(axis="index"))

won = int(summary.softest_team_won.sum()); n = len(summary)
print(f"MYTH-BUSTER: the team that drew the SOFTEST group won only {won}/{n} editions "
      f"({won/n:.0%}) — both times a host.")
print(f"Champions' group-softness rank ranged from 1st (softest) to "
      f"{int(summary.champ_softness_rank.max())}th of their field — "
      f"2018 France won from a hard group; a soft draw is neither necessary nor sufficient.")
print(f"2022 was the most BALANCED draw on record (rival clustering "
      f"{summary.loc[summary.year==2022,'rival_clustering_pct'].iloc[0]:.0f}%): "
      f"no engineered concentration of rivals.")
""")

md(r"""
## 11. Detector — FIFA leadership lens (exploratory)

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
## 12. Takeaways

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
