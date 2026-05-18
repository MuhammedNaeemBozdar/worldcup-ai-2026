"""
utils/simulator.py
FIFA World Cup 2026 tournament simulation using Monte Carlo.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_models, build_prediction_features

# ── MODULE-LEVEL CACHED MODEL ─────────────────────────────────────────────────
_MODELS = None
_SCALER = None
_FEAT_COLS = None
_PROB_TABLE = {}   # (team_a, team_b) -> (home_p, draw_p, away_p)

def _get_model():
    global _MODELS, _SCALER, _FEAT_COLS
    if _MODELS is None:
        _MODELS, _SCALER, _FEAT_COLS, _ = load_models()
    return _MODELS["XGBoost"]


def _precompute_probs():
    """Batch-predict all WC 2026 matchup probabilities using symmetric neutral-venue averaging."""
    global _PROB_TABLE
    if _PROB_TABLE:
        return

    from utils.data_loader import _build_team_cache
    _build_team_cache()
    model = _get_model()

    all_teams = list(set(t for g in WC2026_GROUPS.values() for t in g))
    rows, keys = [], []

    
    # Collect all feature rows
    feat_ref = None
    for ta in all_teams:
        for tb in all_teams:
            if ta == tb:
                continue
            X = build_prediction_features(ta, tb, neutral=True)
            if X is not None:
                if feat_ref is None:
                    feat_ref = list(X.columns)
                rows.append(X.values[0])
                keys.append((ta, tb))

    if not rows:
        return

    import numpy as np
    import pandas as pd
    X_batch = pd.DataFrame(rows, columns=feat_ref)
    probs_batch = model.predict_proba(X_batch)  # shape (N, 3)

    # Build lookup (ta, tb) -> raw probs
    raw = {}
    for (ta, tb), probs in zip(keys, probs_batch):
        if len(probs) == 3:
            raw[(ta, tb)] = (float(probs[0]), float(probs[1]), float(probs[2]))
        else:
            raw[(ta, tb)] = (float(probs[0]), 0.20, float(probs[1]))

    # Average both directions for symmetric neutral-venue result
    for ta in all_teams:
        for tb in all_teams:
            if ta == tb:
                continue
            ab = raw.get((ta, tb), (0.40, 0.30, 0.30))
            ba = raw.get((tb, ta), (0.40, 0.30, 0.30))
            h = (ab[0] + ba[2]) / 2
            d = (ab[1] + ba[1]) / 2
            a = (ab[2] + ba[0]) / 2
            total = h + d + a
            _PROB_TABLE[(ta, tb)] = (h / total, d / total, a / total)

    # Fill any remaining missing pairs
    for ta in all_teams:
        for tb in all_teams:
            if ta != tb and (ta, tb) not in _PROB_TABLE:
                _PROB_TABLE[(ta, tb)] = (0.40, 0.30, 0.30)


def predict_match_fast(team_a: str, team_b: str) -> tuple:
    """Instant lookup from pre-computed probability table."""
    if not _PROB_TABLE:
        _precompute_probs()
    return _PROB_TABLE.get((team_a, team_b), (0.40, 0.30, 0.30))

# ──────────────────────────────────────────────────────────────────────────────
# FIFA WORLD CUP 2026 — 48 TEAMS, 12 GROUPS OF 4
# ──────────────────────────────────────────────────────────────────────────────

WC2026_GROUPS = {
    "A": ["United States", "Mexico", "Canada", "Jamaica"],
    "B": ["Brazil", "Argentina", "Ecuador", "Bolivia"],
    "C": ["France", "Portugal", "Croatia", "Albania"],
    "D": ["England", "Netherlands", "Denmark", "Hungary"],
    "E": ["Spain", "Germany", "Austria", "Serbia"],
    "F": ["Belgium", "Italy", "Switzerland", "Slovakia"],
    "G": ["Morocco", "Senegal", "Cameroon", "Tunisia"],
    "H": ["Japan", "South Korea", "Australia", "Iran"],
    "I": ["Uruguay", "Colombia", "Peru", "Venezuela"],
    "J": ["Nigeria", "Ghana", "Egypt", "Algeria"],
    "K": ["Saudi Arabia", "Qatar", "Iraq", "Bahrain"],
    "L": ["Poland", "Czech Republic", "Romania", "Bulgaria"],
}

QUALIFIED_TEAMS = [t for group in WC2026_GROUPS.values() for t in group]


def simulate_group_match(team_a: str, team_b: str) -> tuple:
    """
    Returns (goals_a, goals_b) based on predicted probabilities.
    Simulates scoreline using Poisson-like logic.
    """
    home_p, draw_p, away_p = predict_match_fast(team_a, team_b)

    # Normalize probabilities (cast to float64 to avoid numpy float32 precision issues)
    home_p, draw_p, away_p = float(home_p), float(draw_p), float(away_p)
    total = home_p + draw_p + away_p
    if total <= 0:
        home_p, draw_p, away_p = 0.34, 0.33, 0.33
        total = 1.0
    p0 = home_p / total
    p1 = draw_p / total
    p2 = max(0.0, 1.0 - p0 - p1)
    p0 = max(0.0, p0)
    p1 = max(0.0, p1)
    # Final renorm
    s = p0 + p1 + p2
    probs = [p0/s, p1/s, p2/s]

    # Sample outcome
    outcome = np.random.choice(["home", "draw", "away"], p=probs)

    base_goals = 1.3
    if outcome == "home":
        ga = np.random.poisson(base_goals * 1.4)
        gb = np.random.poisson(base_goals * 0.7)
        if ga <= gb:
            ga = gb + 1
    elif outcome == "away":
        ga = np.random.poisson(base_goals * 0.7)
        gb = np.random.poisson(base_goals * 1.4)
        if gb <= ga:
            gb = ga + 1
    else:
        ga = np.random.poisson(base_goals)
        gb = ga  # guarantee draw
    return int(ga), int(gb)


def simulate_group_stage(groups: dict) -> dict:
    """Simulate all group matches, return standings dict."""
    standings = {}
    for group, teams in groups.items():
        table = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0} for t in teams}
        n = len(teams)
        for i in range(n):
            for j in range(i + 1, n):
                ta, tb = teams[i], teams[j]
                ga, gb = simulate_group_match(ta, tb)
                table[ta]["gf"] += ga; table[ta]["ga"] += gb; table[ta]["gd"] += ga - gb
                table[tb]["gf"] += gb; table[tb]["ga"] += ga; table[tb]["gd"] += gb - ga
                if ga > gb:
                    table[ta]["pts"] += 3; table[ta]["w"] += 1; table[tb]["l"] += 1
                elif ga == gb:
                    table[ta]["pts"] += 1; table[tb]["pts"] += 1; table[ta]["d"] += 1; table[tb]["d"] += 1
                else:
                    table[tb]["pts"] += 3; table[tb]["w"] += 1; table[ta]["l"] += 1
        # Sort: pts → gd → gf
        sorted_teams = sorted(table.keys(),
                              key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]),
                              reverse=True)
        standings[group] = [(t, table[t]) for t in sorted_teams]
    return standings


def get_qualifiers(standings: dict, top_n: int = 2) -> list:
    qualifiers = []
    for group, rows in standings.items():
        qualifiers.extend([rows[i][0] for i in range(min(top_n, len(rows)))])
    return qualifiers


def simulate_knockout_match(team_a: str, team_b: str) -> str:
    """Returns winner (no draws allowed — penalty shootout)."""
    home_p, draw_p, away_p = predict_match_fast(team_a, team_b)
    # In knockouts redistribute draw probability
    total = home_p + away_p
    if total == 0:
        return np.random.choice([team_a, team_b])
    ha = home_p / total
    return team_a if np.random.random() < ha else team_b


def simulate_knockout_bracket(qualifiers: list) -> dict:
    """Simulate R32 → R16 → QF → SF → Final."""
    bracket = {"Round of 32": [], "Round of 16": [], "Quarterfinals": [],
                "Semifinals": [], "Final": [], "Champion": None}

    # R32 (32 teams — first 32 of 48 qualifiers, two from each group)
    r32_teams = qualifiers[:32] if len(qualifiers) >= 32 else qualifiers
    np.random.shuffle(r32_teams)
    r32_winners = []
    for i in range(0, len(r32_teams), 2):
        if i + 1 < len(r32_teams):
            w = simulate_knockout_match(r32_teams[i], r32_teams[i+1])
        else:
            w = r32_teams[i]
        bracket["Round of 32"].append((r32_teams[i], r32_teams[i+1] if i+1 < len(r32_teams) else "BYE", w))
        r32_winners.append(w)

    # R16
    r16_winners = []
    for i in range(0, len(r32_winners), 2):
        if i + 1 < len(r32_winners):
            w = simulate_knockout_match(r32_winners[i], r32_winners[i+1])
        else:
            w = r32_winners[i]
        bracket["Round of 16"].append((r32_winners[i], r32_winners[i+1] if i+1 < len(r32_winners) else "BYE", w))
        r16_winners.append(w)

    # QF
    qf_winners = []
    for i in range(0, len(r16_winners), 2):
        if i + 1 < len(r16_winners):
            w = simulate_knockout_match(r16_winners[i], r16_winners[i+1])
        else:
            w = r16_winners[i]
        bracket["Quarterfinals"].append((r16_winners[i], r16_winners[i+1] if i+1 < len(r16_winners) else "BYE", w))
        qf_winners.append(w)

    # SF
    sf_winners = []
    for i in range(0, len(qf_winners), 2):
        if i + 1 < len(qf_winners):
            w = simulate_knockout_match(qf_winners[i], qf_winners[i+1])
        else:
            w = qf_winners[i]
        bracket["Semifinals"].append((qf_winners[i], qf_winners[i+1] if i+1 < len(qf_winners) else "BYE", w))
        sf_winners.append(w)

    # Final
    if len(sf_winners) >= 2:
        champ = simulate_knockout_match(sf_winners[0], sf_winners[1])
        bracket["Final"].append((sf_winners[0], sf_winners[1], champ))
        bracket["Champion"] = champ
    elif sf_winners:
        bracket["Champion"] = sf_winners[0]

    return bracket


def run_monte_carlo(n_simulations: int = 500) -> dict:
    """
    Run n_simulations full WC tournaments.
    Returns champion_counts dict and semifinalist_counts dict.
    """
    _precompute_probs()   # batch-predict all pairs once upfront

    champion_counts = defaultdict(int)
    finalist_counts = defaultdict(int)
    semifinalist_counts = defaultdict(int)

    for _ in range(n_simulations):
        standings = simulate_group_stage(WC2026_GROUPS)
        qualifiers = get_qualifiers(standings, top_n=2)
        # Extend with 3rd-place best teams (8 of 12 groups) to reach 32
        third_place = []
        for group, rows in standings.items():
            if len(rows) >= 3:
                third_place.append((rows[2][0], rows[2][1]["pts"], rows[2][1]["gd"]))
        third_place.sort(key=lambda x: (x[1], x[2]), reverse=True)
        qualifiers.extend([t[0] for t in third_place[:8]])

        bracket = simulate_knockout_bracket(qualifiers)

        if bracket["Champion"]:
            champion_counts[bracket["Champion"]] += 1
        if bracket["Final"]:
            for team in bracket["Final"][0][:2]:
                finalist_counts[team] += 1
        if bracket["Semifinals"]:
            for match in bracket["Semifinals"]:
                for team in match[:2]:
                    semifinalist_counts[team] += 1

    return {
        "champion_counts": dict(champion_counts),
        "finalist_counts": dict(finalist_counts),
        "semifinalist_counts": dict(semifinalist_counts),
        "n_simulations": n_simulations,
    }


def simulate_once():
    """Run a single full tournament and return bracket + standings."""
    _precompute_probs()   # no-op after first call
    standings = simulate_group_stage(WC2026_GROUPS)
    qualifiers = get_qualifiers(standings, top_n=2)
    third_place = []
    for group, rows in standings.items():
        if len(rows) >= 3:
            third_place.append((rows[2][0], rows[2][1]["pts"], rows[2][1]["gd"]))
    third_place.sort(key=lambda x: (x[1], x[2]), reverse=True)
    qualifiers.extend([t[0] for t in third_place[:8]])
    bracket = simulate_knockout_bracket(qualifiers)
    return standings, bracket
