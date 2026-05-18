"""
pages/simulator.py
FIFA World Cup 2026 Tournament Simulator page.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.simulator import (WC2026_GROUPS, simulate_once, run_monte_carlo)


def show():
    st.markdown("## 🏆 FIFA World Cup 2026 Simulator")

    tabs = st.tabs(["🎲 Single Simulation", "📊 Monte Carlo Analysis", "🗺️ Group Stage Setup"])

    # ── TAB 1: SINGLE SIM ──────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Run a full World Cup simulation")
        if st.button("▶️ Simulate World Cup 2026",use_container_width=True, type="primary"):
            with st.spinner("Simulating all group matches and knockout rounds..."):
                standings, bracket = simulate_once()

            # Champion banner
            champion = bracket.get("Champion", "Unknown")
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#f0b429,#e67e22);
                        border-radius:16px;padding:2rem;text-align:center;margin:1rem 0;'>
                <div style='color:#1a1a1a;font-size:0.9rem;letter-spacing:3px;text-transform:uppercase;font-weight:700;'>🏆 World Cup 2026 Champion</div>
                <div style='color:#1a1a1a;font-size:3rem;font-weight:900;margin:0.5rem 0;'>{champion}</div>
            </div>
            """, unsafe_allow_html=True)

            # Group standings
            st.markdown("### 📋 Group Stage Standings")
            groups_list = list(standings.items())
            for row_start in range(0, len(groups_list), 3):
                cols = st.columns(3)
                for col_idx, col in enumerate(cols):
                    g_idx = row_start + col_idx
                    if g_idx >= len(groups_list):
                        break
                    gname, gdata = groups_list[g_idx]
                    with col:
                        st.markdown(f"**Group {gname}**")
                        rows = []
                        for i, (team, stats) in enumerate(gdata):
                            medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
                            rows.append({
                                "Team": f"{medal} {team}",
                                "Pts": stats["pts"],
                                "W": stats["w"], "D": stats["d"], "L": stats["l"],
                                "GD": stats["gd"],
                            })
                        gdf = pd.DataFrame(rows)
                        st.dataframe(gdf, hide_index=True,use_container_width=True,
                                     column_config={"Team": st.column_config.TextColumn(width="medium")})

            # Knockout results
            st.markdown("### ⚡ Knockout Stage Results")
            stage_order = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
            for stage in stage_order:
                matches = bracket.get(stage, [])
                if not matches:
                    continue
                st.markdown(f"#### {stage}")
                match_cols = st.columns(min(len(matches), 4))
                for idx, (ta, tb, winner) in enumerate(matches):
                    with match_cols[idx % 4]:
                        ta_style = "color:#f0b429;font-weight:700;" if winner == ta else "color:#888;"
                        tb_style = "color:#f0b429;font-weight:700;" if winner == tb else "color:#888;"
                        st.markdown(f"""
                        <div style='background:#12122a;border-radius:10px;padding:0.7rem;
                                    border:1px solid #ffffff15;margin:0.3rem 0;font-size:0.85rem;'>
                            <div style='{ta_style}'>{ta}</div>
                            <div style='color:#555;font-size:0.7rem;text-align:center;'>vs</div>
                            <div style='{tb_style}'>{tb}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── TAB 2: MONTE CARLO ──────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### Monte Carlo Tournament Analysis")
        st.info("💡 Probabilities are pre-computed once per session — simulations run in ~2ms each.")
        n_sim = st.slider("Number of simulations", min_value=100, max_value=2000, value=500, step=100)
        if st.button("🎰 Run Monte Carlo",use_container_width=True, type="primary"):
            with st.spinner(f"Pre-computing match probabilities + running {n_sim} simulations..."):
                results = run_monte_carlo(n_simulations=n_sim)

            champ = results["champion_counts"]
            total = results["n_simulations"]

            top_champions = sorted(champ.items(), key=lambda x: x[1], reverse=True)[:15]
            teams_c = [t[0] for t in top_champions]
            probs_c = [t[1] / total * 100 for t in top_champions]

            fig = go.Figure(go.Bar(
                y=teams_c[::-1], x=probs_c[::-1],
                orientation="h",
                marker=dict(
                    color=probs_c[::-1],
                    colorscale="Oranges",
                    line=dict(color="#0d0d1a", width=1),
                ),
                text=[f"{p:.1f}%" for p in probs_c[::-1]],
                textposition="outside",
                textfont=dict(color="white", size=12),
            ))
            fig.update_layout(
                title=dict(text="🏆 Championship Probability (Top 15 Teams)", font=dict(color="white")),
                xaxis=dict(title="Win %", color="#888", gridcolor="#1e1e3a"),
                yaxis=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d0d1a",
                height=550,
                margin=dict(l=140, r=80, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Finalist probability
            st.markdown("#### 🎯 Finalist Reach Probability")
            fin = results["finalist_counts"]
            top_fin = sorted(fin.items(), key=lambda x: x[1], reverse=True)[:10]
            fin_teams = [t[0] for t in top_fin]
            fin_probs = [t[1] / total * 100 for t in top_fin]

            fig2 = go.Figure(go.Bar(
                x=fin_teams, y=fin_probs,
                marker=dict(color="#4CAF50", opacity=0.85),
                text=[f"{p:.1f}%" for p in fin_probs],
                textposition="outside",
                textfont=dict(color="white"),
            ))
            fig2.update_layout(
                xaxis=dict(color="white"),
                yaxis=dict(title="Reach Final %", color="#888", gridcolor="#1e1e3a"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d0d1a",
                height=380,
                margin=dict(t=30, b=60),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Stats summary
            top3 = top_champions[:3]
            c1, c2, c3 = st.columns(3)
            for col, (team, cnt), medal in zip([c1, c2, c3], top3, ["🥇", "🥈", "🥉"]):
                with col:
                    st.markdown(f"""
                    <div style='background:#12122a;border-radius:12px;padding:1.2rem;text-align:center;
                                border:1px solid #f0b42940;'>
                        <div style='font-size:2rem;'>{medal}</div>
                        <div style='color:#f0b429;font-size:1rem;font-weight:700;'>{team}</div>
                        <div style='color:#aaa;font-size:0.85rem;'>{cnt/total:.1%} chance</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 3: GROUP STAGE SETUP ────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🗺️ FIFA World Cup 2026 — 48 Teams in 12 Groups")
        cols = st.columns(3)
        for idx, (group, teams) in enumerate(WC2026_GROUPS.items()):
            with cols[idx % 3]:
                teams_html = "".join(
                    f"<div style='padding:0.35rem 0;border-bottom:1px solid #1e1e3a;color:#ddd;font-size:0.88rem;'>"
                    f"<span style='color:#f0b429;margin-right:0.5rem;'>{'🥇' if i==0 else '⚽'}</span>{t}"
                    f"</div>"
                    for i, t in enumerate(teams)
                )
                st.markdown(f"""
                <div style='background:#12122a;border-radius:12px;padding:1rem;margin:0.5rem 0;
                            border:1px solid #f0b42920;'>
                    <div style='color:#f0b429;font-size:1rem;font-weight:700;margin-bottom:0.5rem;'>
                        Group {group}
                    </div>
                    {teams_html}
                </div>
                """, unsafe_allow_html=True)
