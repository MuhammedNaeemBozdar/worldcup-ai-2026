"""
pages/historical.py
Historical match data explorer page.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_match_features, get_all_teams


def show():
    st.markdown("## 📜 Historical Match Data")

    df = load_match_features()
    df["year"] = df["_date"].dt.year

    # ── FILTER BAR ─────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        all_teams = get_all_teams()
        with col1:
            filter_team = st.selectbox("Filter by Team (optional)", ["All"] + all_teams)
        with col2:
            year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()),
                                    (2000, int(df["year"].max())))
        with col3:
            tournaments = ["All"] + sorted(df["_tournament"].dropna().unique().tolist())
            filter_tourn = st.selectbox("Tournament", tournaments)

    filtered = df[
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1])
    ]
    if filter_team != "All":
        filtered = filtered[
            (filtered["_home_team"] == filter_team) |
            (filtered["_away_team"] == filter_team)
        ]
    if filter_tourn != "All":
        filtered = filtered[filtered["_tournament"] == filter_tourn]

    # ── KPIs ───────────────────────────────────────────────────────────────
    total_matches = len(filtered)
    avg_goals = (filtered["home_goals"].sum() + filtered["away_goals"].sum()) / max(total_matches, 1)
    home_wins = (filtered["home_goals"] > filtered["away_goals"]).sum()
    home_win_pct = home_wins / max(total_matches, 1)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, icon, color in [
        (c1, "Total Matches", f"{total_matches:,}", "⚽", "#4CAF50"),
        (c2, "Avg Goals/Match", f"{avg_goals:.2f}", "🥅", "#f0b429"),
        (c3, "Home Win Rate", f"{home_win_pct:.1%}", "🏠", "#3498db"),
        (c4, "Year Range", f"{year_range[0]}–{year_range[1]}", "📅", "#9b59b6"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#12122a;border-radius:12px;padding:1rem;text-align:center;
                        border-top:3px solid {color};'>
                <div style='font-size:1.4rem;'>{icon}</div>
                <div style='color:{color};font-size:1.5rem;font-weight:800;'>{val}</div>
                <div style='color:#888;font-size:0.75rem;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MATCHES PER YEAR ───────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 📅 Matches Per Year")
        by_year = filtered.groupby("year").size().reset_index(name="count")
        fig = go.Figure(go.Bar(
            x=by_year["year"], y=by_year["count"],
            marker_color="#f0b429", opacity=0.85,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
            xaxis=dict(color="#888", gridcolor="#1e1e3a"),
            yaxis=dict(color="#888", gridcolor="#1e1e3a"),
            height=300, margin=dict(t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 🏆 Matches by Tournament Type")
        tourn_counts = filtered["_tournament"].value_counts().head(12)
        fig2 = go.Figure(go.Bar(
            y=tourn_counts.index[::-1], x=tourn_counts.values[::-1],
            orientation="h", marker_color="#3498db", opacity=0.85,
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
            xaxis=dict(color="#888", gridcolor="#1e1e3a"),
            yaxis=dict(color="white", tickfont=dict(size=11)),
            height=300, margin=dict(t=10, b=20, l=160),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── GOALS DISTRIBUTION ─────────────────────────────────────────────────
    st.markdown("#### 🥅 Goals Scored Distribution")
    total_goals = pd.concat([
        filtered["home_goals"].rename("goals"),
        filtered["away_goals"].rename("goals")
    ])
    goals_dist = total_goals.value_counts().sort_index().reset_index()
    goals_dist.columns = ["goals", "count"]
    goals_dist = goals_dist[goals_dist["goals"] <= 10]

    fig3 = go.Figure(go.Bar(
        x=goals_dist["goals"], y=goals_dist["count"],
        marker=dict(color=goals_dist["goals"], colorscale="Viridis"),
        text=goals_dist["count"],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
        xaxis=dict(color="white", title="Goals in a Match", dtick=1),
        yaxis=dict(color="#888", gridcolor="#1e1e3a"),
        height=300, margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── RAW DATA TABLE ─────────────────────────────────────────────────────
    st.markdown("#### 🗃️ Match Records")
    display_cols = ["_date", "_home_team", "home_goals", "away_goals", "_away_team", "_tournament"]
    display = filtered[display_cols].sort_values("_date", ascending=False).head(200).copy()
    display.columns = ["Date", "Home Team", "HG", "AG", "Away Team", "Tournament"]
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display,use_container_width=True, hide_index=True)
