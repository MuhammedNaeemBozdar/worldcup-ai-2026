"""
pages/model_insights.py
ML Model performance and insights page.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_models, train_and_save_models


def show():
    st.markdown("## 🤖 Machine Learning Model Insights")

    tabs = st.tabs(["📈 Model Performance", "⚙️ Training Pipeline", "🔢 Feature Importance"])

    with tabs[0]:
        st.markdown("### Model Comparison & Metrics")

        col_train, _ = st.columns([2, 3])
        with col_train:
            if st.button("🔄 Retrain All Models", type="primary"):
                with st.spinner("Training Logistic Regression, Random Forest, XGBoost..."):
                    _, _, _, metrics = train_and_save_models()
                st.success("✅ Models retrained successfully!")
                st.session_state["metrics"] = metrics

        with st.spinner("Loading model metrics..."):
            try:
                _, _, _, metrics = load_models()
            except Exception:
                with st.spinner("Training models for the first time..."):
                    _, _, _, metrics = train_and_save_models()

        model_names = list(metrics.keys())
        accs  = [metrics[m]["accuracy"] for m in model_names]
        precs = [metrics[m]["precision"] for m in model_names]
        recs  = [metrics[m]["recall"] for m in model_names]

        # Metrics bar chart
        fig = go.Figure()
        for label, vals, color in [
            ("Accuracy",  accs,  "#f0b429"),
            ("Precision", precs, "#4CAF50"),
            ("Recall",    recs,  "#3498db"),
        ]:
            fig.add_trace(go.Bar(name=label, x=model_names, y=[v*100 for v in vals],
                                 marker_color=color,
                                 text=[f"{v:.1%}" for v in vals],
                                 textposition="outside",
                                 textfont=dict(color="white")))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
            xaxis=dict(color="white"), yaxis=dict(color="#888", gridcolor="#1e1e3a", title="%"),
            legend=dict(font=dict(color="white")),
            height=380, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Best model highlight
        best_model = model_names[np.argmax(accs)]
        best_acc   = max(accs)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#12122a,#1a1a3e);border-radius:12px;
                    padding:1.2rem 2rem;border:1px solid #f0b42940;text-align:center;'>
            <div style='color:#aaa;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;'>Best Performing Model</div>
            <div style='color:#f0b429;font-size:2rem;font-weight:800;'>{best_model}</div>
            <div style='color:#ccc;'>Accuracy: {best_acc:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confusion matrices
        st.markdown("### 🧮 Confusion Matrices")
        cm_cols = st.columns(3)
        for col, mname in zip(cm_cols, model_names):
            with col:
                cm = metrics[mname]["confusion_matrix"]
                st.markdown(f"**{mname}**")
                fig_cm = ff.create_annotated_heatmap(
                    z=cm.tolist(),
                    x=["Home Win", "Draw", "Away Win"],
                    y=["Home Win", "Draw", "Away Win"],
                    colorscale="Blues",
                    showscale=False,
                    font_colors=["white"],
                )
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(t=20, b=20, l=60, r=10),
                    xaxis=dict(color="white"),
                    yaxis=dict(color="white"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    with tabs[1]:
        st.markdown("### ⚙️ ML Pipeline Architecture")
        steps = [
            ("1. Data Loading", "Load 43,000+ historical matches with team stats, Elo ratings, and form metrics.", "#3498db"),
            ("2. Feature Engineering", "Create Elo diff, momentum diff, form win rates, attack/defense differentials.", "#9b59b6"),
            ("3. Target Variable", "Encode result as: 0=Home Win, 1=Draw, 2=Away Win (3-class classification).", "#e67e22"),
            ("4. Train/Test Split", "80% training data, 20% test data with stratification to preserve class balance.", "#4CAF50"),
            ("5. Standardisation", "StandardScaler applied to Logistic Regression features only.", "#f0b429"),
            ("6. Model Training", "Train Logistic Regression, Random Forest (200 trees), XGBoost (300 estimators).", "#e74c3c"),
            ("7. Evaluation", "Compute Accuracy, Precision, Recall and Confusion Matrix for each model.", "#1abc9c"),
            ("8. Model Persistence", "Save best model (XGBoost) + scaler + feature columns with joblib.", "#3498db"),
        ]
        for title, desc, color in steps:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:1rem;padding:0.8rem;
                        background:#12122a;border-radius:10px;margin:0.5rem 0;border-left:3px solid {color};'>
                <div style='color:{color};font-weight:700;font-size:0.95rem;min-width:220px;'>{title}</div>
                <div style='color:#ccc;font-size:0.9rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📐 Feature Set Used")
        features = [
            ("elo_diff", "Difference in Elo ratings between home and away team"),
            ("overall_diff", "FIFA overall rating difference"),
            ("attack_diff", "Attack rating differential"),
            ("defense_diff", "Defense rating differential"),
            ("home/away_form_win_rate", "Recent win rate from last matches"),
            ("home/away_form_scored", "Average goals scored in recent matches"),
            ("home/away_form_conceded", "Average goals conceded in recent matches"),
            ("momentum_diff", "Derived: form win rate gap"),
            ("goal_diff_form", "Derived: recent goal difference gap"),
            ("elo_advantage", "Binary: is Elo gap > 100 points"),
            ("is_neutral / is_world_cup", "Venue and tournament context flags"),
        ]
        feat_df = pd.DataFrame(features, columns=["Feature", "Description"])
        feat_df.index += 1
        st.dataframe(feat_df,use_container_width=True, hide_index=False)

    with tabs[2]:
        st.markdown("### 🔢 XGBoost Feature Importance")
        with st.spinner("Loading feature importance..."):
            try:
                models, _, feat_cols, _ = load_models()
                xgb_model = models["XGBoost"]
                importances = xgb_model.feature_importances_
                feat_imp = pd.DataFrame({
                    "Feature": feat_cols,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)

                fig = go.Figure(go.Bar(
                    y=feat_imp["Feature"][::-1],
                    x=feat_imp["Importance"][::-1],
                    orientation="h",
                    marker=dict(
                        color=feat_imp["Importance"][::-1],
                        colorscale="YlOrRd",
                    ),
                    text=[f"{v:.3f}" for v in feat_imp["Importance"][::-1]],
                    textposition="outside",
                    textfont=dict(color="white", size=11),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d1a",
                    xaxis=dict(color="#888", gridcolor="#1e1e3a"),
                    yaxis=dict(color="white"),
                    height=500,
                    margin=dict(l=200, r=80, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Train the models first. Error: {e}")
