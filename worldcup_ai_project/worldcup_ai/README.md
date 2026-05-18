# ⚽ FIFA World Cup 2026 AI Prediction System

A professional-grade, ML-powered football analytics dashboard built with Python and Streamlit.

## 🚀 Quick Start

```bash
# 1. Clone / navigate to project
cd worldcup_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the ML models (first time only — ~30 seconds)
python train.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
worldcup_ai/
│
├── app.py                    ← Main Streamlit dashboard
├── train.py                  ← Standalone model training script
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── teams_match_features.csv   ← 43,000+ historical matches
│   │   ├── teams_form.csv             ← Team form metrics
│   │   └── player_aggregates.csv      ← FIFA player ratings by country
│   └── processed/
│
├── models/
│   └── trained_models/
│       ├── xgboost_model.pkl
│       ├── rf_model.pkl
│       ├── lr_model.pkl
│       ├── scaler.pkl
│       ├── feature_cols.pkl
│       └── metrics.pkl
│
├── utils/
│   ├── data_loader.py    ← Data loading, feature engineering, prediction
│   ├── simulator.py      ← WC 2026 tournament simulation (Monte Carlo)
│   └── insights.py       ← AI-generated match insights
│
└── pages/
    ├── prediction.py         ← Match Prediction page
    ├── simulator.py          ← Tournament Simulator page
    ├── analytics.py          ← Team Analytics page
    ├── player_analytics.py   ← Player / Country Analytics page
    ├── model_insights.py     ← ML Model Performance page
    └── historical.py         ← Historical Match Data explorer
```

---

## 🧠 Machine Learning Pipeline

| Step | Description |
|------|-------------|
| **Data** | 43,364 international matches from 1872–2024 |
| **Features** | Elo ratings, FIFA player stats, recent form, momentum |
| **Target** | 3-class: Home Win (0), Draw (1), Away Win (2) |
| **Models** | Logistic Regression, Random Forest, **XGBoost** (best) |
| **Accuracy** | ~62–63% (vs 33% random baseline) |

### Features Used
- `elo_diff` — Elo rating gap between teams
- `overall_diff`, `attack_diff`, `defense_diff` — FIFA attribute gaps
- `home/away_form_win_rate` — Recent form
- `home/away_form_scored/conceded` — Goal-scoring form
- `momentum_diff` — Derived form gap
- `goal_diff_form` — Recent net goals advantage
- `is_world_cup`, `is_neutral`, `is_continental` — Context flags

---

## 🏆 FIFA World Cup 2026 Groups

| Group | Teams |
|-------|-------|
| A | United States, Mexico, Canada, Jamaica |
| B | Brazil, Argentina, Ecuador, Bolivia |
| C | France, Portugal, Croatia, Albania |
| D | England, Netherlands, Denmark, Hungary |
| E | Spain, Germany, Austria, Serbia |
| F | Belgium, Italy, Switzerland, Slovakia |
| G | Morocco, Senegal, Cameroon, Tunisia |
| H | Japan, South Korea, Australia, Iran |
| I | Uruguay, Colombia, Peru, Venezuela |
| J | Nigeria, Ghana, Egypt, Algeria |
| K | Saudi Arabia, Qatar, Iraq, Bahrain |
| L | Poland, Czech Republic, Romania, Bulgaria |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview, dataset stats, group stage preview |
| 🔮 **Match Prediction** | AI win probabilities, radar charts, insights |
| 🏆 **Tournament Simulator** | Full WC 2026 simulation + Monte Carlo |
| 📊 **Team Analytics** | Elo history, form trends, match records |
| 👤 **Player Analytics** | Country ratings, attribute comparisons |
| 🤖 **Model Insights** | Accuracy, confusion matrices, feature importance |
| 📜 **Historical Data** | 43K+ match database with filters |

---

## 🎨 Design

- **Theme**: Dark sports analytics (inspired by Sofascore / FotMob)
- **Fonts**: Rajdhani (headings) + Inter (body)
- **Colors**: Gold (#f0b429) accent on deep navy (#0d0d1a)
- **Charts**: Plotly interactive with dark backgrounds
