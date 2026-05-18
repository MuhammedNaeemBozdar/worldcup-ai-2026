"""
pages/prediction.py
Match Prediction page — AI-powered win probabilities, radar, insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import predict_match, get_all_teams, load_match_features
from utils.insights import generate_match_insights, generate_team_profile
from visualizations.charts import (win_probability_donut, radar_chart, GOLD, GREEN, RED)


def _prob_pill(team, prob, color):
    return f"""
    <div style='background:#12122a;border-radius:12px;padding:1.2rem;text-align:center;border:1px solid {color}40;'>
        <div style='color:#aaa;font-size:0.8rem;margin-bottom:0.3rem;'>{team}</div>
        <div style='color:{color};font-size:2.4rem;font-weight:800;font-family:Rajdhani,sans-serif;'>{prob:.1%}</div>
        <div style='background:#1a1a30;border-radius:4px;height:6px;margin-top:0.6rem;'>
            <div style='background:{color};width:{prob*100:.1f}%;height:6px;border-radius:4px;'></div>
        </div>
    </div>"""


def show():
    st.markdown("## ⚽ Match Prediction Engine")

    all_teams = get_all_teams()

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        home_team = st.selectbox("🏠 Team A", all_teams,
            index=all_teams.index("Brazil") if "Brazil" in all_teams else 0, key="pred_home")
    with col_vs:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1.8rem;"
                    "font-weight:900;color:#f0b429;font-family:Rajdhani,sans-serif;'>VS</div>",
                    unsafe_allow_html=True)
    with col2:
        away_team = st.selectbox("✈️ Team B", all_teams,
            index=all_teams.index("France") if "France" in all_teams else 1, key="pred_away")

    neutral = st.toggle("Neutral Venue (World Cup)", value=True)

    if home_team == away_team:
        st.warning("Please select two different teams.")
        return

    if st.button("🔮 Predict Match",use_container_width=True, type="primary"):
        with st.spinner("Running AI model..."):
            h_prob, d_prob, a_prob = predict_match(home_team, away_team, neutral)

        # Winner banner
        if h_prob >= a_prob and h_prob >= d_prob:
            winner, wcolor, wconf = home_team, GREEN, h_prob
        elif a_prob >= h_prob and a_prob >= d_prob:
            winner, wcolor, wconf = away_team, RED, a_prob
        else:
            winner, wcolor, wconf = "Draw likely", GOLD, d_prob

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d0d1a 0%,#12122a 100%);
                    border-radius:16px;padding:1.8rem 2rem;margin:1rem 0;
                    border:1px solid {wcolor}40;text-align:center;'>
            <div style='color:#888;font-size:0.8rem;letter-spacing:3px;text-transform:uppercase;'>AI Prediction</div>
            <div style='color:{wcolor};font-size:2.8rem;font-weight:900;
                        font-family:Rajdhani,sans-serif;margin:0.4rem 0;'>{winner}</div>
            <div style='color:#bbb;font-size:0.9rem;'>Confidence: <b style="color:{wcolor};">{wconf:.1%}</b></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for col, team, prob, color in [(c1, home_team, h_prob, GREEN), (c2, "Draw", d_prob, GOLD), (c3, away_team, a_prob, RED)]:
            with col:
                st.markdown(_prob_pill(team, prob, color), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        chart_col, radar_col = st.columns(2)

        with chart_col:
            st.markdown("##### Win Probability Distribution")
            fig = win_probability_donut([home_team, "Draw", away_team], [h_prob, d_prob, a_prob], colors=[GREEN, GOLD, RED])
            st.plotly_chart(fig, use_container_width=True)

        with radar_col:
            st.markdown("##### Team Attributes Radar")
            df = load_match_features()
            def get_attrs(team, is_home=True):
                pfx = "home" if is_home else "away"
                col_key = f"_{pfx}_team"
                rows = df[df[col_key] == team].sort_values("_date")
                if len(rows):
                    r = rows.iloc[-1]
                    return [r.get(f"{pfx}_avg_overall",70), r.get(f"{pfx}_avg_attack",70),
                            r.get(f"{pfx}_avg_defense",70), r.get(f"{pfx}_avg_pace",70),
                            r.get(f"{pfx}_avg_shooting",65), r.get(f"{pfx}_avg_passing",68)]
                return [70,70,70,70,65,68]
            cats = ["Overall","Attack","Defense","Pace","Shooting","Passing"]
            fig2 = radar_chart({home_team: get_attrs(home_team,True), away_team: get_attrs(away_team,False)}, cats, range_=(55,90))
            st.plotly_chart(fig2, use_container_width=True)

        # AI Insights
        st.markdown("### 🤖 AI Match Insights")
        insights = generate_match_insights(home_team, away_team, h_prob, d_prob, a_prob)
        for insight in insights:
            st.markdown(f"<div style='background:#12122a;border-left:3px solid #f0b429;"
                        f"padding:0.85rem 1.2rem;border-radius:0 8px 8px 0;margin:0.4rem 0;"
                        f"color:#ddd;font-size:0.95rem;'>{insight}</div>", unsafe_allow_html=True)

        # Team Profiles
        st.markdown("### 📋 Team Profiles")
        pc1, pc2 = st.columns(2)
        for col, team, color in [(pc1, home_team, GREEN), (pc2, away_team, RED)]:
            profile = generate_team_profile(team)
            with col:
                strengths_html = "".join(f"<div style='color:#ccc;font-size:0.78rem;'>✅ {s}</div>" for s in profile["strengths"])
                weaknesses_html = "".join(f"<div style='color:#ccc;font-size:0.78rem;'>⚠️ {w}</div>" for w in profile["weaknesses"])
                st.markdown(f"""
                <div style='background:#12122a;border-radius:12px;padding:1.4rem;border:1px solid {color}35;'>
                    <div style='color:{color};font-size:1.15rem;font-weight:700;font-family:Rajdhani,sans-serif;margin-bottom:0.8rem;'>{team}</div>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.8rem;'>
                        <div style='color:#aaa;font-size:0.8rem;'>Elo Rating</div>
                        <div style='color:#fff;font-size:0.8rem;font-weight:600;'>{profile["elo"]}</div>
                        <div style='color:#aaa;font-size:0.8rem;'>Win Rate</div>
                        <div style='color:#fff;font-size:0.8rem;font-weight:600;'>{profile["win_rate"]:.1%}</div>
                        <div style='color:#aaa;font-size:0.8rem;'>Total Matches</div>
                        <div style='color:#fff;font-size:0.8rem;font-weight:600;'>{profile["total_matches"]}</div>
                        <div style='color:#aaa;font-size:0.8rem;'>Goals Scored</div>
                        <div style='color:#fff;font-size:0.8rem;font-weight:600;'>{profile["goals_for"]}</div>
                    </div>
                    <div style='border-top:1px solid #1e1e3a;padding-top:0.6rem;'>
                        {strengths_html}{weaknesses_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Expected Score
        st.markdown("### 🎯 Expected Score Estimate")
        home_xg = max(0.3, round(1.3 + (h_prob - 0.33) * 2.5, 1))
        away_xg = max(0.3, round(1.3 + (a_prob - 0.33) * 2.5, 1))
        home_goals_est = int(np.round(home_xg))
        away_goals_est = int(np.round(away_xg))
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#12122a,#0d0d20);border-radius:14px;
                    padding:1.8rem;text-align:center;border:1px solid #ffffff10;'>
            <div style='color:#888;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.8rem;'>Expected Scoreline</div>
            <div style='display:flex;justify-content:center;align-items:center;gap:1.5rem;'>
                <div>
                    <div style='color:#4CAF50;font-size:3.5rem;font-weight:900;font-family:Rajdhani,sans-serif;'>{home_goals_est}</div>
                    <div style='color:#888;font-size:0.85rem;'>{home_team}</div>
                </div>
                <div style='color:#444;font-size:2rem;'>—</div>
                <div>
                    <div style='color:#e74c3c;font-size:3.5rem;font-weight:900;font-family:Rajdhani,sans-serif;'>{away_goals_est}</div>
                    <div style='color:#888;font-size:0.85rem;'>{away_team}</div>
                </div>
            </div>
            <div style='color:#555;font-size:0.75rem;margin-top:0.8rem;'>xG model: {home_xg:.2f} – {away_xg:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
