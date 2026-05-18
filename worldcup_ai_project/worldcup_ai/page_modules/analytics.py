"""
pages/analytics.py
Team Analytics Dashboard page.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import (get_team_stats, load_match_features,
                                load_teams_form, get_all_teams)


def show():
    st.markdown("## 📊 Team Analytics Dashboard")

    all_teams = get_all_teams()
    selected_team = st.selectbox("Select a team", all_teams,
                                  index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)

    with st.spinner("Loading team analytics..."):
        df = load_match_features()
        form_df = load_teams_form()

        home = df[df["_home_team"] == selected_team].copy()
        away = df[df["_away_team"] == selected_team].copy()

        total   = len(home) + len(away)
        wins    = int((home["home_goals"] > home["away_goals"]).sum() +
                      (away["away_goals"] > away["home_goals"]).sum())
        draws   = int((home["home_goals"] == home["away_goals"]).sum() +
                      (away["home_goals"] == away["away_goals"]).sum())
        losses  = total - wins - draws
        gf      = int(home["home_goals"].sum() + away["away_goals"].sum())
        ga      = int(home["away_goals"].sum() + away["home_goals"].sum())
        win_pct = wins / total if total else 0

        elo = 1500.0
        if len(home):
            elo = home.sort_values("_date")["home_elo"].iloc[-1]

    # ── KPI CARDS ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Matches", total, "⚽", "#4CAF50"),
        (c2, "Win Rate", f"{win_pct:.1%}", "🏆", "#f0b429"),
        (c3, "Goals Scored", gf, "⚡", "#3498db"),
        (c4, "Goals Conceded", ga, "🛡️", "#e74c3c"),
        (c5, "Elo Rating", int(elo), "📈", "#9b59b6"),
    ]
    for col, label, val, icon, color in kpis:
        with col:
            st.markdown(f"""
            <div style='background:#12122a;border-radius:12px;padding:1rem;text-align:center;
                        border-top:3px solid {color};'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='color:{color};font-size:1.5rem;font-weight:800;'>{val}</div>
                <div style='color:#888;font-size:0.75rem;text-transform:uppercase;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── W/D/L PIE ─────────────────────────────────────────────────────────
    col_pie, col_goals = st.columns(2)

    with col_pie:
        st.markdown("#### Win / Draw / Loss Breakdown")
        fig = go.Figure(go.Pie(
            labels=["Wins", "Draws", "Losses"],
            values=[wins, draws, losses],
            hole=0.5,
            marker=dict(colors=["#4CAF50", "#f0b429", "#e74c3c"],
                        line=dict(color="#0d0d1a", width=3)),
            textinfo="percent+label",
            textfont=dict(color="white", size=13),
        ))
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_goals:
        st.markdown("#### Goals Per Match (Form Trend)")
        team_form = form_df[form_df["team"] == selected_team].sort_values("match_date").tail(30)
        if len(team_form) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=team_form["match_date"], y=team_form["avg_goals_scored"],
                mode="lines+markers", name="Goals Scored",
                line=dict(color="#4CAF50", width=2), marker=dict(size=5),
                fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
            ))
            fig2.add_trace(go.Scatter(
                x=team_form["match_date"], y=team_form["avg_goals_conceded"],
                mode="lines+markers", name="Goals Conceded",
                line=dict(color="#e74c3c", width=2), marker=dict(size=5),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d0d1a",
                xaxis=dict(color="#888", gridcolor="#1e1e3a"),
                yaxis=dict(color="#888", gridcolor="#1e1e3a"),
                legend=dict(font=dict(color="white")),
                height=300,
                margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No form data available for this team.")

    # ── ELO HISTORY ────────────────────────────────────────────────────────
    st.markdown("#### 📈 Elo Rating History")
    home_elo_hist = home[["_date", "home_elo"]].rename(columns={"home_elo": "elo"})
    away_elo_hist = away[["_date", "away_elo"]].rename(columns={"away_elo": "elo"})
    elo_hist = pd.concat([home_elo_hist, away_elo_hist]).sort_values("_date").tail(100)

    fig3 = go.Figure(go.Scatter(
        x=elo_hist["_date"], y=elo_hist["elo"],
        mode="lines", line=dict(color="#f0b429", width=2),
        fill="tozeroy", fillcolor="rgba(240,180,41,0.1)",
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
        xaxis=dict(color="#888", gridcolor="#1e1e3a"),
        yaxis=dict(color="#888", gridcolor="#1e1e3a", title="Elo"),
        height=280, margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── LAST 10 MATCHES ────────────────────────────────────────────────────
    st.markdown("#### 🕐 Recent Match History")
    home_recent = home[["_date", "_away_team", "home_goals", "away_goals", "_tournament"]].copy()
    home_recent["opponent"] = home_recent["_away_team"]
    home_recent["score"] = home_recent["home_goals"].astype(int).astype(str) + " - " + home_recent["away_goals"].astype(int).astype(str)
    home_recent["result"] = home_recent.apply(
        lambda r: "✅ Win" if r["home_goals"] > r["away_goals"]
        else ("🟡 Draw" if r["home_goals"] == r["away_goals"] else "❌ Loss"), axis=1)
    home_recent["venue"] = "Home"

    away_recent = away[["_date", "_home_team", "home_goals", "away_goals", "_tournament"]].copy()
    away_recent["opponent"] = away_recent["_home_team"]
    away_recent["score"] = away_recent["away_goals"].astype(int).astype(str) + " - " + away_recent["home_goals"].astype(int).astype(str)
    away_recent["result"] = away_recent.apply(
        lambda r: "✅ Win" if r["away_goals"] > r["home_goals"]
        else ("🟡 Draw" if r["away_goals"] == r["home_goals"] else "❌ Loss"), axis=1)
    away_recent["venue"] = "Away"

    recent = pd.concat([
        home_recent[["_date", "opponent", "score", "result", "venue", "_tournament"]],
        away_recent[["_date", "opponent", "score", "result", "venue", "_tournament"]],
    ]).sort_values("_date", ascending=False).head(10)
    recent.columns = ["Date", "Opponent", "Score", "Result", "Venue", "Tournament"]
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(recent, hide_index=True,use_container_width=True)

    # ── ALL TEAMS LEADERBOARD ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏅 All-Time Team Leaderboard (Top 30 by Win Rate, min 50 matches)")
    with st.spinner("Computing global stats..."):
        stats_df = get_team_stats()
    filtered = stats_df[stats_df["matches"] >= 50].sort_values("win_rate", ascending=False).head(30)
    filtered = filtered.reset_index(drop=True)
    filtered.index += 1
    filtered.columns = [c.replace("_", " ").title() for c in filtered.columns]
    st.dataframe(filtered,use_container_width=True)
