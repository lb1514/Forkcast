"""
Forkcast: a little forecast for your next favourite bite.
PPOL 5205 Concept Showcase · Laalithya Bhiminani

Screens: landing -> quiz (one question at a time) -> results.
The recommender logic (build_profile, score_restaurants, explain, apply_feedback) is unchanged
from the original version:
  1. Quiz              -> starting taste profile (+1 chosen, -0.5 rejected)
  2. Association rules -> expand the profile with related restaurant types
  3. LDA themes        -> what each restaurant is praised for / criticised for
  4. Scoring           -> taste match + praised themes - dealbreakers + adjusted rating
  5. Feedback          -> Love it / Pass update the profile and re-rank
"""
import html
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="Forkcast", page_icon="🍴", layout="centered", initial_sidebar_state="collapsed")

@st.cache_data
def load_data():
    r = pd.read_csv(HERE / "restaurants_app.csv")
    r["taste_groups"] = r["taste_groups"].fillna("").apply(lambda s: [g for g in s.split("|") if g])
    r["description"] = r["description"].fillna("")
    r["price"] = r["price"].fillna("")
    rules = pd.read_csv(HERE / "rules.csv")
    return r, rules

restaurants, rules = load_data()

# Names for the LDA themes (pos_0 and neg_6 are Google-translated text, so they are not used)
POS_THEMES = {
    "pos_1": "quick & convenient service", "pos_2": "burgers & beer", "pos_3": "seafood & trying new things",
    "pos_4": "great atmosphere & service", "pos_5": "comfort food", "pos_6": "coffee & sweets",
    "pos_7": "Asian food & good value", "pos_8": "a lively vibe", "pos_9": "pizza & tacos",
}
NEG_THEMES = {
    "neg_0": "disappointing food", "neg_1": "delivery & order problems", "neg_2": "rude staff",
    "neg_3": "slow table service", "neg_4": "long waits & cold food", "neg_5": "poor quality for the price",
    "neg_7": "rude counter staff",
}

# ----------------------------------------------------------------------------
# Quiz: each option maps to taste groups (from categories) and praised themes (from LDA)
# ----------------------------------------------------------------------------
QUESTIONS = [
    {"key": "q1", "text": "Ramen & dumplings, or tacos & arepas?",
     "options": {"🍜 Ramen & dumplings": {"groups": ["Japanese & sushi", "Chinese", "Korean", "Southeast Asian", "Asian fusion"]},
                 "🌮 Tacos & arepas": {"groups": ["Mexican", "Latin American & Caribbean"]}}},
    {"key": "q2", "text": "Pasta & mezze, or burgers & BBQ?",
     "options": {"🍝 Pasta & mezze": {"groups": ["Italian", "Mediterranean & Middle Eastern", "European"]},
                 "🍔 Burgers & BBQ": {"groups": ["American classics", "Southern, BBQ & Cajun", "Chicken & wings"]}}},
    {"key": "q3", "text": "A cozy café, or a lively bar?",
     "options": {"☕ Cozy café": {"groups": ["Coffee & tea", "Bakery & desserts"], "themes": ["pos_6"]},
                 "🍸 Lively bar": {"groups": ["Bars & pubs", "Cocktails & wine", "Nightlife"], "themes": ["pos_8"]}}},
    {"key": "q4", "text": "Weekend brunch, or a late-night bite?",
     "options": {"🥞 Weekend brunch": {"groups": ["Breakfast & brunch"]},
                 "🌙 Late-night bite": {"groups": ["Nightlife", "Pizza", "Fast food"]}}},
    {"key": "q5", "text": "Quick & casual, or a sit-down treat?",
     "options": {"🥪 Quick & casual": {"groups": ["Fast food", "Sandwiches & delis", "Pizza"], "themes": ["pos_1"]},
                 "🍷 Sit-down treat": {"groups": ["Fine dining", "Steakhouse", "Seafood"], "themes": ["pos_4"]}}},
]
NEW_OR_FAMILIAR = ["⭐ My usual favourites", "🧭 Something new"]
PRIORITIES = {"Great atmosphere & service": ["pos_4"], "Quick & convenient": ["pos_1"],
              "A lively vibe": ["pos_8"], "Good value": ["pos_7", "pos_9"]}
DEALBREAKERS = {"Slow table service": ["neg_3"], "Rude staff": ["neg_2", "neg_7"],
                "Long waits & cold food": ["neg_4"], "Delivery problems": ["neg_1"],
                "Disappointing food or value": ["neg_0", "neg_5"]}
LESS_COMMON = {"Korean", "Southeast Asian", "African", "Indian & South Asian", "Mediterranean & Middle Eastern"}

# ----------------------------------------------------------------------------
# The recommender
# ----------------------------------------------------------------------------
def build_profile(answers):
    """Stage 1 (quiz) + Stage 2 (association rules)."""
    groups, themes, sources = {}, {}, {}
    for q in QUESTIONS:
        picked = answers[q["key"]]
        for label, info in q["options"].items():
            w = 1.0 if label == picked else -0.5
            for g in info["groups"]:
                groups[g] = groups.get(g, 0) + w
                if w > 0:
                    sources[g] = ("quiz", label)
            if w > 0:
                for t in info.get("themes", []):
                    themes[t] = themes.get(t, 0) + 1.0
    for t in PRIORITIES[answers["priority"]]:
        themes[t] = themes.get(t, 0) + 1.5

    adventurous = answers["novelty"] == NEW_OR_FAMILIAR[1]
    strength = 0.9 if adventurous else 0.6
    liked = {g: s for g, s in groups.items() if s > 0}
    for _, r in rules.iterrows():
        a, b = r["if_user_likes"], r["they_also_like"]
        if a in liked and groups.get(b, 0) == 0:
            add = strength * r["confidence"] * min(liked[a], 1.5)
            if add > groups.get(b, 0):
                groups[b] = add
                sources[b] = ("rule", a, r["confidence"], r["lift"])
    if adventurous:
        for g in LESS_COMMON:
            if groups.get(g, 0) >= 0:
                groups[g] = groups.get(g, 0) + 0.2
    dealbreakers = [t for d in answers["dealbreakers"] for t in DEALBREAKERS[d]]
    return {"groups": groups, "themes": themes, "sources": sources, "dealbreakers": dealbreakers}


def score_restaurants(df, profile):
    """Stage 3 + 4: taste match (A) + praised themes (B) - dealbreakers (C) + adjusted rating (D)."""
    g = profile["groups"]
    A = df["taste_groups"].apply(lambda gs: float(np.mean([np.clip(g.get(x, 0), -1, 1.5) for x in gs])) if gs else 0.0)
    B = pd.Series(0.0, index=df.index)
    if profile["themes"]:
        total = sum(profile["themes"].values())
        for t, w in profile["themes"].items():
            B += (w / total) * df[t] * 3          # theme shares are small, so scale up
    C = pd.Series(0.0, index=df.index)
    for t in profile["dealbreakers"]:
        C += df[t] * (df["share_negative"] / 0.08)  # complaint theme x how often reviews are negative (0.08 = average)
    D = ((df["adj_rating"] - 4.0) / 0.8).clip(0, 1)
    out = df.copy()
    out["A"], out["B"], out["C"], out["D"] = A, B.clip(0, 1.5), C.clip(0, 2), D
    out["score"] = 1.0 * out["A"] + 0.6 * out["B"] - 0.6 * out["C"] + 0.6 * out["D"]
    return out.sort_values("score", ascending=False)


def explain(row, profile):
    """'Why this?': the strongest reasons behind a recommendation."""
    reasons = []
    best = sorted(row["taste_groups"], key=lambda x: profile["groups"].get(x, 0), reverse=True)
    if best and profile["groups"].get(best[0], 0) > 0:
        src = profile["sources"].get(best[0])
        if src and src[0] == "rule":
            reasons.append(f"People who enjoy <b>{src[1]}</b> often enjoy <b>{best[0]}</b> "
                           f"({src[2]:.0%} of them, {src[3]:.1f}× more than average)")
        else:
            reasons.append(f"Matches your taste for <b>{best[0]}</b>")
    praised = [(row[t], name) for t, name in POS_THEMES.items() if t in profile["themes"]]
    if praised:
        share, name = max(praised)
        if share > 0.15:
            reasons.append(f"Reviewers often praise its <b>{name}</b>")
    if profile["dealbreakers"] and row["C"] < 0.15:
        reasons.append("Rarely criticised for your dealbreakers")
    if row["n_reviews"] < 60 and row["our_avg_rating"] >= 4.5:
        reasons.append("A well-rated <b>hidden gem</b> with few reviews")
    elif row["D"] > 0.6:
        reasons.append(f"Consistently well rated ({row['our_avg_rating']:.1f}★ from {int(row['n_reviews']):,} reviews)")
    return reasons[:3]


def apply_feedback(profile, row, liked):
    """Stage 5: thumbs up/down nudge the taste groups and themes of that restaurant."""
    step = 0.3 if liked else -0.3
    for g in row["taste_groups"]:
        profile["groups"][g] = profile["groups"].get(g, 0) + step
    top_themes = sorted(POS_THEMES, key=lambda t: row[t], reverse=True)[:2]
    for t in top_themes:
        profile["themes"][t] = max(0.0, profile["themes"].get(t, 0) + step)

# ============================================================================
# Interface (design, navigation and presentation only)
# ============================================================================
INK, MUTED, TERRA, SAGE, CREAM, CARD, LINE = "#302E2A", "#858078", "#B7654B", "#89977A", "#FAF7F0", "#FFFDF9", "#E8DFD2"
ss = st.session_state
ss.setdefault("stage", "landing")     # landing -> quiz -> results
ss.setdefault("step", 0)
ss.setdefault("ans", {})


def food_pattern(opacity):
    """A faint, hand-drawn line pattern of food (bowl, cup, fork, pizza, taco), drawn in SVG so it never breaks."""
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280' viewBox='0 0 280 280'>
<g fill='none' stroke='{TERRA}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' opacity='{opacity}'>
<path d='M24 70 h64 a32 32 0 0 1 -64 0 z'/><path d='M44 60 l32 -34 M54 62 l36 -30'/>
<path d='M176 44 h42 v32 a12 12 0 0 1 -12 12 h-18 a12 12 0 0 1 -12 -12 z'/><path d='M218 52 h6 a9 9 0 0 1 0 18 h-6'/>
<path d='M188 32 q5 -7 0 -14 M202 32 q5 -7 0 -14'/>
<path d='M76 160 v68 M64 160 v20 a12 12 0 0 0 24 0 v-20 M76 160 v20'/>
<path d='M180 150 l56 22 l-44 54 z'/><circle cx='202' cy='178' r='3.5'/><circle cx='210' cy='198' r='3.5'/>
<path d='M118 268 a30 30 0 0 1 60 0 z'/><path d='M126 259 q6 -7 13 0 q6 -7 13 0 q6 -7 13 0'/>
</g></svg>"""
    return 'url("data:image/svg+xml;utf8,' + quote(svg) + '")'


if ss.stage == "landing":
    background = (f"background-color:{CREAM}; background-image: radial-gradient(circle at 15% 20%, rgba(183,101,75,0.10), transparent 40%),"
                  f" radial-gradient(circle at 85% 80%, rgba(137,151,122,0.12), transparent 45%), {food_pattern(0.10)};"
                  " background-size: auto, auto, 280px 280px;")
else:
    background = (f"background-color:{CREAM}; background-image: linear-gradient({CREAM} 0, rgba(250,247,240,0.94) 100%), {food_pattern(0.06)};"
                  " background-size: auto, 280px 280px; background-attachment: fixed;")
max_width = "1080px" if ss.stage == "results" else "760px"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, p, label, li, button, input {{ font-family: 'Inter', sans-serif; color: {INK}; }}
.stApp {{ {background} }}
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{ display: none !important; }}
.block-container {{ max-width: {max_width}; padding-top: 3.2rem; padding-bottom: 4rem; }}
.serif {{ font-family: 'Fraunces', serif; }}

/* buttons: default = quiet outline, primary = terracotta */
.stButton button, .stLinkButton a {{ border-radius: 999px; border: 1px solid {LINE}; background: {CARD}; color: {INK};
  font-weight: 500; transition: all .18s ease; box-shadow: none; }}
.stButton button:hover, .stLinkButton a:hover {{ border-color: {TERRA}; color: {TERRA}; background: {CARD}; }}
.stButton button[data-testid="stBaseButton-primary"] {{ background: {TERRA}; border-color: {TERRA}; color: #fff; }}
.stButton button[data-testid="stBaseButton-primary"]:hover {{ background: #A45A42; color: #fff; }}
.stButton button[data-testid="stBaseButton-primary"] p {{ color: #fff; }}
.stButton button:disabled {{ opacity: .45; }}

/* landing */
.landing {{ text-align: center; padding: 11vh 0 2rem 0; }}
.mark {{ display:inline-block; font-size: .78rem; letter-spacing: .22em; text-transform: uppercase; color: {SAGE};
  border: 1px solid {LINE}; border-radius: 999px; padding: 6px 14px; background: rgba(255,253,249,.7); }}
.landing h1 {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: clamp(3.4rem, 9vw, 5.6rem); line-height: 1;
  margin: 26px 0 10px 0; color: {INK}; letter-spacing: -0.03em; }}
.landing .tag {{ font-family: 'Fraunces', serif; font-style: italic; font-size: 1.35rem; color: {TERRA}; margin-bottom: 18px; }}
.landing .intro {{ color: {MUTED}; font-size: 1.05rem; max-width: 460px; margin: 0 auto 34px auto; line-height: 1.6; }}
.st-key-start button {{ padding: 14px 34px; font-size: 1.05rem; }}

/* quiz */
.qtop {{ display:flex; justify-content: space-between; align-items: center; color: {MUTED}; font-size: .85rem; margin-bottom: 10px; }}
.qtop .brand {{ font-family: 'Fraunces', serif; font-size: 1.15rem; color: {INK}; }}
.track {{ height: 3px; background: {LINE}; border-radius: 3px; overflow: hidden; margin-bottom: 46px; }}
.fill {{ height: 100%; background: {TERRA}; border-radius: 3px; transition: width .35s ease; }}
.qtitle {{ font-family: 'Fraunces', serif; font-size: clamp(1.9rem, 5vw, 2.5rem); line-height: 1.15; text-align: center;
  margin: 0 0 10px 0; letter-spacing: -0.01em; }}
.qsub {{ text-align: center; color: {MUTED}; margin-bottom: 34px; }}
.st-key-options button {{ min-height: 168px; width: 100%; border-radius: 22px; background: {CARD}; border: 1px solid {LINE};
  display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 18px 14px; }}
.st-key-options button p {{ font-family: 'Fraunces', serif; font-size: 1.22rem; font-weight: 400; color: {INK}; }}
.st-key-options button:hover {{ transform: translateY(-2px); border-color: {TERRA}; }}
.st-key-options .stButton button[data-testid="stBaseButton-primary"],
.st-key-options .stButton button[data-testid="stBaseButton-primary"]:hover {{ background: #F6E9E3; border: 2px solid {TERRA}; }}
.st-key-options .stButton button[data-testid="stBaseButton-primary"] p {{ color: {INK}; }}
.st-key-chips button {{ min-height: 58px; border-radius: 16px; justify-content: flex-start; padding: 12px 18px; }}
.st-key-chips button p {{ font-size: .98rem; }}
.st-key-chips .stButton button[data-testid="stBaseButton-primary"],
.st-key-chips .stButton button[data-testid="stBaseButton-primary"]:hover {{ background: #EEF1EA; border: 2px solid {SAGE}; }}
.st-key-chips .stButton button[data-testid="stBaseButton-primary"] p {{ color: {INK}; }}
.st-key-nav {{ margin-top: 34px; }}

/* results */
.rhead {{ text-align: center; margin-bottom: 26px; }}
.rhead .eyebrow {{ font-size: .78rem; letter-spacing: .22em; text-transform: uppercase; color: {SAGE}; }}
.rhead h1 {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: clamp(2.4rem, 6vw, 3.4rem); margin: 10px 0 8px 0; letter-spacing: -0.02em; }}
.rhead p {{ color: {MUTED}; font-size: 1.05rem; }}
[class*="st-key-rcard_"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 22px; padding: 22px 22px 16px 22px;
  margin-bottom: 18px; transition: border-color .2s ease; }}
[class*="st-key-rcard_"]:hover {{ border-color: #D9C6B6; }}
.r-top {{ display:flex; justify-content: space-between; align-items:center; margin-bottom: 8px; }}
.r-rank {{ font-family: 'Fraunces', serif; color: {TERRA}; font-size: .95rem; letter-spacing: .08em; }}
.gem {{ font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: #5E6B52; background: #EEF1EA;
  border-radius: 999px; padding: 3px 10px; }}
.gem.new {{ color: #A4503A; background: #F6E9E3; }}
.r-name {{ font-family: 'Fraunces', serif; font-size: 1.55rem; line-height: 1.15; margin-bottom: 6px; }}
.r-meta {{ font-size: .92rem; color: {INK}; margin-bottom: 4px; }}
.r-meta .star {{ color: {TERRA}; font-weight: 600; }}
.r-sub {{ font-size: .82rem; color: {MUTED}; margin-bottom: 12px; }}
.r-desc {{ font-size: .9rem; color: #57524B; line-height: 1.5; margin-bottom: 12px; display: -webkit-box;
  -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }}
.r-tags span {{ display:inline-block; font-size: .74rem; color: {MUTED}; border: 1px solid {LINE}; border-radius: 999px;
  padding: 2px 10px; margin: 0 5px 6px 0; }}
.r-why {{ border-top: 1px solid {LINE}; margin-top: 10px; padding-top: 12px; font-size: .86rem; color: #57524B; min-height: 92px; }}
.r-why .t {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {SAGE}; margin-bottom: 6px; }}
.r-why div.reason {{ margin-bottom: 3px; }}
[class*="st-key-rcard_"] .stButton button, [class*="st-key-rcard_"] .stLinkButton a {{ font-size: .85rem; padding: 4px 10px; min-height: 36px; }}
.note {{ text-align:center; color: {MUTED}; font-size: .88rem; margin: -6px 0 18px 0; }}
.bar {{ display:flex; align-items:center; gap: 12px; margin: 6px 0; font-size: .88rem; }}
.bar .lbl {{ width: 210px; color: {INK}; }}
.bar .bg {{ flex: 1; height: 8px; background: {LINE}; border-radius: 8px; overflow: hidden; }}
.bar .fg {{ height: 100%; background: {TERRA}; border-radius: 8px; }}
.bar .src {{ width: 90px; color: {MUTED}; font-size: .75rem; }}
table.rules {{ width:100%; border-collapse: collapse; font-size: .86rem; margin: 8px 0; }}
table.rules th, table.rules td {{ text-align:left; padding: 6px 8px; border-bottom: 1px solid {LINE}; }}
table.rules th {{ color: {MUTED}; font-weight: 500; }}
[data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; }}
[data-testid="stExpander"] summary p {{ font-family: 'Fraunces', serif; font-size: 1.08rem; }}
</style>
""", unsafe_allow_html=True)

# ---------------- quiz content (wording and display only; values are the original labels) ----------------
CITIES = [("New York City", "🗽", "Bagels to omakase"), ("Chicago", "🌆", "Deep dish and beyond"),
          ("Miami", "🌴", "Cuban, seafood, sunshine"), ("New Orleans", "🎷", "Creole, Cajun, jazz"),
          ("Washington DC", "🏛️", "Ethiopian to embassy row")]
CAPTIONS = {
    "🍜 Ramen & dumplings": "noodles, sushi, bao, pho", "🌮 Tacos & arepas": "tacos, arepas, empanadas",
    "🍝 Pasta & mezze": "Italian, Mediterranean, European", "🍔 Burgers & BBQ": "burgers, barbecue, wings",
    "☕ Cozy café": "coffee, pastries, quiet corners", "🍸 Lively bar": "cocktails, music, good company",
    "🥞 Weekend brunch": "eggs, pancakes, slow mornings", "🌙 Late-night bite": "pizza, snacks, after hours",
    "🥪 Quick & casual": "sandwiches, slices, grab-and-go", "🍷 Sit-down treat": "fine dining, steak, seafood",
    "⭐ My usual favourites": "tried-and-true", "🧭 Something new": "new cuisines, hidden corners",
}
PRIORITY_ICONS = {"Great atmosphere & service": ("✨", "the whole experience"), "Quick & convenient": ("⚡", "in and out"),
                  "A lively vibe": ("🎶", "buzz and energy"), "Good value": ("💸", "worth every dollar")}
NONE = "None of these"


def two_options(q):
    return [(label, label.split(" ", 1)[0], label.split(" ", 1)[1], CAPTIONS.get(label, "")) for label in q["options"]]


STEPS = [
    {"key": "city", "title": "Where are we eating today?", "sub": "Pick a city to explore.", "kind": "single",
     "options": [(c, e, c, cap) for c, e, cap in CITIES], "cols": 3},
    {"key": "q1", "title": "What's calling your name?", "sub": "Go with your gut.", "kind": "single", "options": two_options(QUESTIONS[0]), "cols": 2},
    {"key": "q2", "title": "Pasta & mezze or burgers & BBQ?", "sub": "Just this once, you have to choose.", "kind": "single", "options": two_options(QUESTIONS[1]), "cols": 2},
    {"key": "q3", "title": "What's your kind of setting?", "sub": "Where do you feel most at home?", "kind": "single", "options": two_options(QUESTIONS[2]), "cols": 2},
    {"key": "q4", "title": "What's your ideal food plan?", "sub": "Morning person or night owl?", "kind": "single", "options": two_options(QUESTIONS[3]), "cols": 2},
    {"key": "q5", "title": "How do you like to eat?", "sub": "Depends on the day, but usually…", "kind": "single", "options": two_options(QUESTIONS[4]), "cols": 2},
    {"key": "novelty", "title": "Sticking to what you know, or exploring something new?", "sub": "No wrong answers here.",
     "kind": "single", "options": [(v, v.split(" ", 1)[0], v.split(" ", 1)[1], CAPTIONS[v]) for v in NEW_OR_FAMILIAR], "cols": 2},
    {"key": "priority", "title": "What matters most when choosing somewhere to eat?", "sub": "Pick the one you care about most.",
     "kind": "single", "options": [(p, PRIORITY_ICONS[p][0], p, PRIORITY_ICONS[p][1]) for p in PRIORITIES], "cols": 2, "compact": True},
    {"key": "dealbreakers", "title": "Anything you'd rather avoid?", "sub": "Choose as many as you like.", "kind": "multi",
     "options": list(DEALBREAKERS) + [NONE]},
]


def choose(key, value):
    ss.ans[key] = value


def toggle_dealbreaker(value):
    current = [d for d in ss.ans.get("dealbreakers", [])]
    if value == NONE:
        current = [] if NONE in current else [NONE]
    else:
        current = [d for d in current if d != NONE]
        current = [d for d in current if d != value] if value in current else current + [value]
    ss.ans["dealbreakers"] = [d for d in list(DEALBREAKERS) + [NONE] if d in current]


def answered(step):
    return bool(ss.ans.get("dealbreakers")) if step["kind"] == "multi" else step["key"] in ss.ans


def go(stage=None, step=None):
    if stage is not None:
        ss.stage = stage
    if step is not None:
        ss.step = step


def finish_quiz():
    a = ss.ans
    answers = {q["key"]: a[q["key"]] for q in QUESTIONS}
    answers["novelty"] = a["novelty"]
    answers["priority"] = a["priority"]
    answers["dealbreakers"] = [d for d in a.get("dealbreakers", []) if d != NONE]
    ss.profile = build_profile(answers)          # unchanged recommender
    ss.city = a["city"]
    ss.hidden, ss.feedback = set(), []
    ss.stage = "results"


def restart():
    for k in ["profile", "city", "hidden", "feedback"]:
        ss.pop(k, None)
    ss.ans, ss.step, ss.stage = {}, 0, "quiz"


def option_css(step):
    """Illustration (emoji) above and a short caption below each option card."""
    rules_css = []
    for i, (_, emoji, _, cap) in enumerate(step["options"]):
        k = f"opt_{step['key']}_{i}"
        rules_css.append(f'.st-key-{k} button::before {{ content: "{emoji}"; font-size: 2.6rem; display:block; margin-bottom: 10px; }}')
        if cap:
            rules_css.append(f'.st-key-{k} button::after {{ content: "{cap}"; display:block; font-size: .82rem; color: {MUTED}; margin-top: 6px; }}')
    return "<style>" + "\n".join(rules_css) + "</style>"


def short_address(addr, name):
    parts = [p.strip() for p in str(addr).split(",") if p.strip() and str(addr) != "nan"]
    if parts and parts[0] == str(name):
        parts = parts[1:]
    return ", ".join(parts[:2])


def primary_group(gs, profile):
    """The restaurant's taste group that best matches this user's profile."""
    return max(gs, key=lambda x: profile["groups"].get(x, 0)) if gs else None


def diversify(scored, profile, k=10, per_group=2, discovery_slots=3):
    """Pick a varied top-k from the scored list (scores are unchanged):
    - one branch per restaurant name (no chain repeated),
    - at most `per_group` places per taste group,
    - up to `discovery_slots` places from related tastes added by association rules."""
    picked, names, group_count = [], set(), {}
    rows = list(scored.iterrows())

    def try_add(row, cap, discovery):
        name = str(row["name"]).strip().lower()
        g = primary_group(row["taste_groups"], profile)
        if name in names or (cap is not None and g is not None and group_count.get(g, 0) >= cap):
            return False
        picked.append((row, discovery))
        names.add(name)
        group_count[g] = group_count.get(g, 0) + 1
        return True

    # 1. discovery slots: best places whose top-matching taste was added by an association rule
    n_disc = 0
    for _, row in rows:
        if n_disc >= discovery_slots:
            break
        g = primary_group(row["taste_groups"], profile)
        src = profile["sources"].get(g)
        if g and src and src[0] == "rule" and profile["groups"].get(g, 0) > 0:
            n_disc += try_add(row, 1, True)
    # 2. fill the rest in score order, with the per-group cap (relaxed if there aren't enough places)
    for cap in (per_group, None):
        for _, row in rows:
            if len(picked) >= k:
                break
            try_add(row, cap, False)
    # keep score order on the page
    picked.sort(key=lambda p: p[0]["score"], reverse=True)
    out = pd.DataFrame([p[0] for p in picked[:k]]).reset_index(drop=True)
    out["discovery"] = [p[1] for p in picked[:k]]
    return out


# ============================================================================
# Screen 1: landing
# ============================================================================
if ss.stage == "landing":
    st.markdown("""
    <div class="landing">
      <span class="mark">Restaurant discovery</span>
      <h1>Forkcast</h1>
      <div class="tag">A little forecast for your next favourite bite.</div>
      <div class="intro">Tell us what you're craving. We'll help you find somewhere worth discovering.</div>
    </div>""", unsafe_allow_html=True)
    c = st.columns([1, 1.4, 1])
    c[1].button("Let's find my next bite →", key="start", type="primary", width="stretch",
                on_click=go, kwargs={"stage": "quiz", "step": 0})

# ============================================================================
# Screen 2: quiz, one question at a time
# ============================================================================
elif ss.stage == "quiz":
    step = STEPS[ss.step]
    n = len(STEPS)
    st.markdown(f"""
    <div class="qtop"><span class="brand">Forkcast</span><span>Question {ss.step + 1} of {n}</span></div>
    <div class="track"><div class="fill" style="width:{(ss.step + 1) / n * 100:.0f}%"></div></div>
    <div class="qtitle">{step['title']}</div>
    <div class="qsub">{step['sub']}</div>""", unsafe_allow_html=True)

    if step["kind"] == "single":
        st.markdown(option_css(step), unsafe_allow_html=True)
        with st.container(key="options"):
            opts = step["options"]
            per_row = step["cols"]
            for r in range(0, len(opts), per_row):
                cols = st.columns(per_row)
                for j, (value, emoji, label, cap) in enumerate(opts[r:r + per_row]):
                    i = r + j
                    selected = ss.ans.get(step["key"]) == value
                    cols[j].button(label, key=f"opt_{step['key']}_{i}", width="stretch",
                                   type="primary" if selected else "secondary",
                                   on_click=choose, args=(step["key"], value))
    else:
        with st.container(key="chips"):
            chosen = ss.ans.get("dealbreakers", [])
            opts = step["options"]
            for r in range(0, len(opts), 2):
                cols = st.columns(2)
                for j, value in enumerate(opts[r:r + 2]):
                    on = value in chosen
                    cols[j].button(("✓  " if on else "") + value, key=f"chip_{r + j}", width="stretch",
                                   type="primary" if on else "secondary",
                                   on_click=toggle_dealbreaker, args=(value,))

    with st.container(key="nav"):
        left, _, right = st.columns([1, 1.2, 1.3])
        if ss.step > 0:
            left.button("← Back", key="back", width="stretch", on_click=go, kwargs={"step": ss.step - 1})
        else:
            left.button("← Home", key="home", width="stretch", on_click=go, kwargs={"stage": "landing"})
        last = ss.step == n - 1
        if last:
            right.button("See my matches →", key="next", type="primary", width="stretch",
                         disabled=not answered(step), on_click=finish_quiz)
        else:
            right.button("Continue →", key="next", type="primary", width="stretch",
                         disabled=not answered(step), on_click=go, kwargs={"step": ss.step + 1})

# ============================================================================
# Screen 3: results
# ============================================================================
else:
    profile = ss.profile
    city = ss.city
    st.markdown(f"""
    <div class="rhead">
      <div class="eyebrow">Forkcast · {city}</div>
      <h1>Your kind of places.</h1>
      <p>Based on what you told us, here are a few places we think you'll love.</p>
    </div>""", unsafe_allow_html=True)

    ctrl = st.columns([0.6, 1.2, 1, 0.6], vertical_alignment="center")
    gems_only = ctrl[1].toggle("Hidden gems only", key="gems",
                               help="Well-rated places (4.5★ or more) with fewer than 60 reviews")
    ctrl[2].button("↺ Retake quiz", key="retake", width="stretch", on_click=restart)

    pool = restaurants[(restaurants["city"] == city) & (~restaurants["gmap_id"].isin(ss.hidden))]
    if gems_only:
        pool = pool[(pool["n_reviews"] < 60) & (pool["our_avg_rating"] >= 4.5)]
    ranked = diversify(score_restaurants(pool, profile), profile)

    if ss.feedback:
        st.markdown(f"<div class='note'>Updated with your feedback · {html.escape(ss.feedback[-1])}</div>",
                    unsafe_allow_html=True)

    def react(row, liked):
        apply_feedback(ss.profile, row, liked)
        ss.feedback.append(("Loved " if liked else "Passed on ") + str(row["name"]))
        if not liked:
            ss.hidden.add(row["gmap_id"])

    if ranked.empty:
        st.info("No places match these filters yet. Try turning off hidden gems.")

    grid = st.columns(2, gap="medium")
    for i, row in ranked.iterrows():
        gem = row["n_reviews"] < 60 and row["our_avg_rating"] >= 4.5
        reasons = explain(row, profile)
        price = f" · {html.escape(str(row['price']))}" if row["price"] else ""
        desc = html.escape(row["description"]) if row["description"] else ""
        tags = "".join(f"<span>{html.escape(g)}</span>" for g in row["taste_groups"][:3])
        why = "".join(f"<div class='reason'>{r}</div>" for r in reasons) or "<div class='reason'>A strong overall match</div>"
        card = f"""
        <div class="r-top"><span class="r-rank">{i + 1:02d}</span>{'<span class="gem">Hidden gem</span>' if gem else ('<span class="gem new">New for you</span>' if row["discovery"] else '')}</div>
        <div class="r-name">{html.escape(str(row['name']))}</div>
        <div class="r-meta">{html.escape(str(row['main_category']))}{price} · <span class="star">★ {row['adj_rating']:.1f}</span>
          <span style="color:{MUTED}">({int(row['n_reviews']):,} reviews)</span></div>
        <div class="r-sub">{html.escape(short_address(row['address'], row['name']))}</div>
        {'<div class="r-desc">' + desc + '</div>' if desc else ''}
        <div class="r-tags">{tags}</div>
        <div class="r-why"><div class="t">Why this made your list</div>{why}</div>
        """.replace("\n", " ")
        with grid[i % 2]:
            with st.container(key=f"rcard_{i}"):
                st.markdown(card, unsafe_allow_html=True)
                b1, b2, b3 = st.columns([1, 1, 1.25])
                b1.button("♥ Love it", key=f"up_{row['gmap_id']}", width="stretch", on_click=react, args=(row, True))
                b2.button("Pass", key=f"down_{row['gmap_id']}", width="stretch", on_click=react, args=(row, False))
                b3.link_button("Maps ↗", f"https://www.google.com/maps/place//data=!4m2!3m1!1s{row['gmap_id']}",
                               width="stretch")

    st.write("")
    with st.expander("Your taste profile"):
        prof = pd.Series(profile["groups"])
        prof = prof[prof > 0].sort_values(ascending=False).head(8)
        if len(prof):
            bars = ""
            for g, v in prof.items():
                src = profile["sources"].get(g, ("",))[0]
                label = "from your answers" if src == "quiz" else ("related taste" if src == "rule" else "")
                bars += (f"<div class='bar'><span class='lbl'>{html.escape(g)}</span><span class='bg'>"
                         f"<span class='fg' style='display:block;width:{v / prof.max() * 100:.0f}%'></span></span>"
                         f"<span class='src'>{label}</span></div>")
            st.markdown(bars, unsafe_allow_html=True)
        st.caption("“Related tastes” were added because people who share your answers often enjoy them too. "
                   "Every Love it or Pass nudges this profile.")

    with st.expander("How your matches were selected"):
        st.markdown(
            "Your answers sketch a **taste profile**. Forkcast then looks for kinds of places that people with "
            "similar tastes also enjoy, reads what reviewers **praise** (like atmosphere or value) and **complain "
            "about** (like slow service), and checks that each place is consistently well rated, not just lucky "
            "with a handful of reviews. Places that match your tastes, earn praise for what you care about, and "
            "rarely trigger your dealbreakers rise to the top.")

    with st.expander("See them on a map"):
        if len(ranked):
            try:
                st.map(ranked.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]], color=TERRA, size=60)
            except Exception:
                st.caption("The map isn't available on this computer, but it works in the online version.")

    with st.expander("How Forkcast works (the data science)"):
        st.markdown(
            "Forkcast is built on **Google Local Reviews** (McAuley Lab, UC San Diego): 9 million reviews of 47,813 "
            "food and drink spots in New York City, Chicago, Miami, New Orleans and Washington DC, through 2021.\n\n"
            "1. **Taste profile.** Each answer maps to restaurant types: types you pick score +1, types you pass on −0.5.\n"
            "2. **Association rule mining.** From 283,680 reviewers' liked restaurants, Forkcast learns which types are "
            "enjoyed together and adds related types you never mentioned.\n"
            "3. **Topic modelling (LDA).** Recurring themes in positive reviews (what places are praised for) and "
            "negative reviews (common complaints).\n"
            "4. **Scoring.** Taste match + praised themes you value − your dealbreakers + a rating adjusted for how "
            "many reviews support it (ratings shown on cards are these adjusted ratings).\n"
            "5. **Feedback.** Love it and Pass nudge your profile, and the list re-ranks.")
        top_rules = rules.sort_values("lift", ascending=False).head(8)
        rows_html = "".join(
            f"<tr><td>{html.escape(r.if_user_likes)}</td><td>{html.escape(r.they_also_like)}</td>"
            f"<td>{r.confidence:.0%}</td><td>{r.lift:.2f}</td></tr>" for r in top_rules.itertuples())
        st.markdown("<table class='rules'><tr><th>If you like</th><th>You may also like</th><th>Share of those fans</th>"
                    f"<th>Lift</th></tr>{rows_html}</table>", unsafe_allow_html=True)
        st.caption("Lift = how many times more likely than chance. Above 1 means the two go together.")
