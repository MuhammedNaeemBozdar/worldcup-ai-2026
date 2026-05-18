"""
pages/player_analytics.py
Player analytics page using player_aggregates.csv
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_player_aggregates


def show():
    st.markdown("## 👤 Player Analytics")

    df = load_player_aggregates()

    # Latest FIFA version per country
    latest = df.sort_values("fifa_version").groupby("country").last().reset_index()

    tabs = st.tabs(["🌍 Country Rankings", "🔍 Country Deep Dive", "📊 Attribute Comparison"])

    # ── TAB 1: TOP COUNTRIES ────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🏆 Top Countries by Average Player Rating")
        top30 = latest.sort_values("avg_overall", ascending=False).head(30)

        fig = go.Figure(go.Bar(
            y=top30["country"][::-1],
            x=top30["avg_overall"][::-1],
            orientation="h",
            marker=dict(
                color=top30["avg_overall"][::-1],
                colorscale="YlOrRd",
                line=dict(color="#0d0d1a", width=0.5),
            ),
            text=[f"{v:.1f}" for v in top30["avg_overall"][::-1]],
            textposition="outside",
            textfont=dict(color="white", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
            xaxis=dict(color="#888", gridcolor="#1e1e3a", range=[55, 90]),
            yaxis=dict(color="white"),
            height=700,
            margin=dict(l=150, r=80, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top attack vs defense
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⚔️ Top Attack Nations")
            top_att = latest.sort_values("avg_attack_overall", ascending=False).head(10)[
                ["country", "avg_attack_overall", "avg_shooting"]].reset_index(drop=True)
            top_att.index += 1
            top_att.columns = ["Country", "Attack Rating", "Shooting"]
            st.dataframe(top_att,use_container_width=True)

        with col2:
            st.markdown("#### 🛡️ Top Defensive Nations")
            top_def = latest.sort_values("avg_defense_overall", ascending=False).head(10)[
                ["country", "avg_defense_overall", "avg_defending"]].reset_index(drop=True)
            top_def.index += 1
            top_def.columns = ["Country", "Defense Rating", "Defending"]
            st.dataframe(top_def,use_container_width=True)

    # ── TAB 2: COUNTRY DEEP DIVE ─────────────────────────────────────────
    with tabs[1]:
        countries = sorted(latest["country"].unique())
        selected = st.selectbox("Select Country", countries,
                                 index=countries.index("Brazil") if "Brazil" in countries else 0)
        country_data = df[df["country"] == selected].sort_values("fifa_version")
        country_latest = latest[latest["country"] == selected]

        if len(country_latest) == 0:
            st.warning("No data for this country.")
            return

        row = country_latest.iloc[0]

        # KPI row
        attrs = ["avg_overall", "avg_pace", "avg_shooting", "avg_passing",
                 "avg_dribbling", "avg_defending", "avg_physic"]
        labels = ["Overall", "Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physic"]
        colors = ["#f0b429", "#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#1abc9c", "#e67e22"]

        kpi_cols = st.columns(7)
        for col, attr, label, color in zip(kpi_cols, attrs, labels, colors):
            with col:
                val = row.get(attr, 0)
                st.markdown(f"""
                <div style='background:#12122a;border-radius:10px;padding:0.7rem;text-align:center;
                            border-top:2px solid {color};'>
                    <div style='color:{color};font-size:1.4rem;font-weight:800;'>{val:.0f}</div>
                    <div style='color:#888;font-size:0.65rem;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # Radar
        st.markdown("#### 🎯 Attribute Radar")
        radar_attrs = ["avg_pace", "avg_shooting", "avg_passing",
                       "avg_dribbling", "avg_defending", "avg_physic"]
        radar_labels = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physic"]
        values = [row.get(a, 0) for a in radar_attrs]

        fig_r = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            line=dict(color="#f0b429", width=2),
            fillcolor="rgba(240,180,41,0.2)",
            name=selected,
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="#0d0d1a",
                radialaxis=dict(visible=True, range=[40, 95], color="#555"),
                angularaxis=dict(color="#888"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Rating trend over FIFA versions
        if len(country_data) > 1:
            st.markdown("#### 📈 Rating Trend Across FIFA Versions")
            fig_t = go.Figure()
            for attr, label, color in [
                ("avg_overall", "Overall", "#f0b429"),
                ("avg_attack_overall", "Attack", "#e74c3c"),
                ("avg_defense_overall", "Defense", "#4CAF50"),
            ]:
                if attr in country_data.columns:
                    fig_t.add_trace(go.Scatter(
                        x=country_data["fifa_version"],
                        y=country_data[attr],
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                    ))
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
                xaxis=dict(color="#888", gridcolor="#1e1e3a", title="FIFA Version"),
                yaxis=dict(color="#888", gridcolor="#1e1e3a", title="Rating"),
                legend=dict(font=dict(color="white")),
                height=300, margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig_t, use_container_width=True)

    # ── TAB 3: COMPARE COUNTRIES ─────────────────────────────────────────
    with tabs[2]:
        st.markdown("### ⚔️ Compare Two Countries")
        countries_list = sorted(latest["country"].unique())
        col_a, col_b = st.columns(2)
        with col_a:
            team_a = st.selectbox("Country A", countries_list,
                                   index=countries_list.index("Brazil") if "Brazil" in countries_list else 0,
                                   key="cmp_a")
        with col_b:
            team_b = st.selectbox("Country B", countries_list,
                                   index=countries_list.index("France") if "France" in countries_list else 1,
                                   key="cmp_b")

        if team_a != team_b:
            radar_attrs = ["avg_pace", "avg_shooting", "avg_passing",
                           "avg_dribbling", "avg_defending", "avg_physic"]
            radar_labels = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physic"]

            a_row = latest[latest["country"] == team_a].iloc[0]
            b_row = latest[latest["country"] == team_b].iloc[0]

            fig_cmp = go.Figure()
            for team, row, color in [(team_a, a_row, "#4CAF50"), (team_b, b_row, "#e74c3c")]:
                vals = [row.get(a, 0) for a in radar_attrs]
                fig_cmp.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                    fill="toself", name=team,
                    line=dict(color=color, width=2),
                    fillcolor=f"rgba(76,175,80,0.2)" if color=="#4CAF50" else f"rgba(231,76,60,0.2)",
                ))
            fig_cmp.update_layout(
                polar=dict(
                    bgcolor="#0d0d1a",
                    radialaxis=dict(visible=True, range=[40, 95], color="#555"),
                    angularaxis=dict(color="#888"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="white")),
                height=420, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Bar comparison
            comp_data = {
                "Attribute": radar_labels,
                team_a: [a_row.get(a, 0) for a in radar_attrs],
                team_b: [b_row.get(a, 0) for a in radar_attrs],
            }
            comp_df = pd.DataFrame(comp_data)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name=team_a, x=comp_df["Attribute"], y=comp_df[team_a],
                                     marker_color="#4CAF50"))
            fig_bar.add_trace(go.Bar(name=team_b, x=comp_df["Attribute"], y=comp_df[team_b],
                                     marker_color="#e74c3c"))
            fig_bar.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
                xaxis=dict(color="white"), yaxis=dict(color="#888", gridcolor="#1e1e3a"),
                legend=dict(font=dict(color="white")),
                height=350, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
