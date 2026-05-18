"""
utils/insights.py
AI-generated match and team insights.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_match_features, load_teams_form, load_player_aggregates


def generate_match_insights(home_team: str, away_team: str,
                             home_prob: float, draw_prob: float, away_prob: float) -> list:
    """Generate natural language insights for a matchup."""
    df = load_match_features()
    form_df = load_teams_form()
    insights = []

    def get_elo(team):
        rows = df[df["_home_team"] == team].sort_values("_date")
        if len(rows):
            return rows["home_elo"].iloc[-1]
        rows = df[df["_away_team"] == team].sort_values("_date")
        if len(rows):
            return rows["away_elo"].iloc[-1]
        return 1500.0

    def get_form(team):
        rows = form_df[form_df["team"] == team].sort_values("match_date")
        if len(rows):
            return rows.iloc[-1]
        return None

    h_elo = get_elo(home_team)
    a_elo = get_elo(away_team)
    h_form = get_form(home_team)
    a_form = get_form(away_team)

    # Elo insight
    elo_diff = h_elo - a_elo
    if elo_diff > 150:
        insights.append(f"⚡ {home_team} holds a significant Elo rating advantage of {int(elo_diff)} points — historically a strong predictor of victory.")
    elif elo_diff < -150:
        insights.append(f"⚡ {away_team} holds a substantial Elo superiority of {int(abs(elo_diff))} points entering this clash.")
    else:
        insights.append(f"🔄 Both sides are closely matched on Elo rating ({int(h_elo)} vs {int(a_elo)}), signaling a competitive contest.")

    # Form insights
    if h_form is not None and a_form is not None:
        h_wr = h_form["win_rate"]
        a_wr = a_form["win_rate"]
        if h_wr > a_wr + 0.2:
            insights.append(f"📈 {home_team} arrives in superior form with a {h_wr:.0%} recent win rate versus {a_wr:.0%} for {away_team}.")
        elif a_wr > h_wr + 0.2:
            insights.append(f"📈 {away_team} is the in-form side, boasting a {a_wr:.0%} win rate in recent matches.")
        
        h_gs = h_form["avg_goals_scored"]
        a_gs = a_form["avg_goals_scored"]
        if h_gs > a_gs + 0.5:
            insights.append(f"⚽ {home_team}'s attack has been prolific, averaging {h_gs:.1f} goals per game — significantly more than {away_team}'s {a_gs:.1f}.")
        elif a_gs > h_gs + 0.5:
            insights.append(f"⚽ {away_team} brings sharper offensive output ({a_gs:.1f} goals/game) compared to {home_team} ({h_gs:.1f}).")

        h_gc = h_form["avg_goals_conceded"]
        a_gc = a_form["avg_goals_conceded"]
        if h_gc < a_gc - 0.4:
            insights.append(f"🛡️ {home_team}'s defense looks more resilient — conceding just {h_gc:.1f} per game vs {a_gc:.1f} for {away_team}.")
        elif a_gc < h_gc - 0.4:
            insights.append(f"🛡️ {away_team} shows the sturdier backline, conceding {a_gc:.1f} per game against {home_team}'s {h_gc:.1f}.")

    # Prediction insight
    if home_prob > 0.55:
        insights.append(f"🏆 The model gives {home_team} a commanding {home_prob:.0%} chance of winning — they are clear favorites.")
    elif away_prob > 0.55:
        insights.append(f"🏆 Despite the neutral venue, {away_team} is the heavy favorite with {away_prob:.0%} win probability.")
    elif draw_prob > 0.28:
        insights.append(f"🤝 A draw is a realistic outcome here ({draw_prob:.0%} probability) — this fixture could go either way.")

    if len(insights) < 3:
        insights.append(f"🔍 Historical head-to-head data and current form metrics point toward a closely contested match between {home_team} and {away_team}.")

    return insights


def generate_team_profile(team: str) -> dict:
    """Generate textual profile highlights for a team."""
    df = load_match_features()
    home = df[df["_home_team"] == team]
    away = df[df["_away_team"] == team]

    total = len(home) + len(away)
    wins = ((home["home_goals"] > home["away_goals"]).sum() +
            (away["away_goals"] > away["home_goals"]).sum())
    goals_f = int(home["home_goals"].sum() + away["away_goals"].sum())
    goals_a = int(home["away_goals"].sum() + away["home_goals"].sum())

    elo = 1500.0
    if len(home):
        elo = home.sort_values("_date")["home_elo"].iloc[-1]

    strengths = []
    weaknesses = []
    win_rate = wins / total if total else 0

    if win_rate > 0.55:
        strengths.append("consistently dominant win rate")
    if goals_f / total > 1.8:
        strengths.append("prolific attacking output")
    if (goals_a / total) < 1.0:
        strengths.append("solid defensive record")
    if elo > 1800:
        strengths.append("elite Elo rating")

    if win_rate < 0.35:
        weaknesses.append("inconsistent results")
    if goals_f / total < 1.0:
        weaknesses.append("limited scoring threat")
    if (goals_a / total) > 1.8:
        weaknesses.append("vulnerable defensively")

    return {
        "elo": round(elo, 1),
        "win_rate": round(win_rate, 3),
        "total_matches": total,
        "goals_for": goals_f,
        "goals_against": goals_a,
        "strengths": strengths if strengths else ["competitive in most fixtures"],
        "weaknesses": weaknesses if weaknesses else ["no major identified weaknesses"],
    }
