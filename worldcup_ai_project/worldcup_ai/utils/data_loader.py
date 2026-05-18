"""
utils/data_loader.py
Handles all data loading, cleaning, feature engineering, and model training.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained_models")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CACHE
# ──────────────────────────────────────────────────────────────────────────────
_MATCH_DF = None
_FORM_DF = None
_PLAYER_DF = None
# Per-team last-row cache for fast feature building
_TEAM_HOME_ROW = {}
_TEAM_AWAY_ROW = {}
_FEAT_COLS_CACHE = None

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & CLEAN
# ──────────────────────────────────────────────────────────────────────────────

def load_match_features() -> pd.DataFrame:
    """Load the main match features dataset."""
    global _MATCH_DF
    if _MATCH_DF is not None:
        return _MATCH_DF
    path = os.path.join(RAW_DIR, "teams_match_features.csv")
    df = pd.read_csv(path, parse_dates=["_date"])
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["result"] = df.apply(
        lambda r: 0 if r["home_goals"] > r["away_goals"]
        else (1 if r["home_goals"] == r["away_goals"] else 2),
        axis=1
    )
    _MATCH_DF = df
    return df


def load_teams_form() -> pd.DataFrame:
    global _FORM_DF
    if _FORM_DF is not None:
        return _FORM_DF
    path = os.path.join(RAW_DIR, "teams_form.csv")
    _FORM_DF = pd.read_csv(path, parse_dates=["match_date"])
    return _FORM_DF


def load_player_aggregates() -> pd.DataFrame:
    global _PLAYER_DF
    if _PLAYER_DF is not None:
        return _PLAYER_DF
    path = os.path.join(RAW_DIR, "player_aggregates.csv")
    _PLAYER_DF = pd.read_csv(path)
    return _PLAYER_DF


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff",
    "overall_diff",
    "attack_diff",
    "defense_diff",
    "home_form_scored",
    "home_form_conceded",
    "home_form_win_rate",
    "away_form_scored",
    "away_form_conceded",
    "away_form_win_rate",
    "is_neutral",
    "is_world_cup",
    "is_continental",
    "home_avg_pace",
    "home_avg_shooting",
    "home_avg_passing",
    "away_avg_pace",
    "away_avg_shooting",
    "away_avg_passing",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["momentum_diff"] = df["home_form_win_rate"] - df["away_form_win_rate"]
    df["goal_diff_form"] = (df["home_form_scored"] - df["home_form_conceded"]) - \
                           (df["away_form_scored"] - df["away_form_conceded"])
    df["elo_advantage"] = df["elo_diff"].apply(lambda x: 1 if x > 100 else (0 if x < -100 else 0.5))
    return df


def get_feature_cols(df: pd.DataFrame):
    extra = ["momentum_diff", "goal_diff_form", "elo_advantage"]
    base = [c for c in FEATURE_COLS if c in df.columns]
    return base + [c for c in extra if c in df.columns]


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save_models():
    """Full ML pipeline: load → engineer → train → save → return metrics."""
    df = load_match_features()
    df = engineer_features(df)
    df = df.dropna(subset=get_feature_cols(df))

    feat_cols = get_feature_cols(df)
    X = df[feat_cols]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost":             xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                                  use_label_encoder=False, eval_metric="mlogloss",
                                                  random_state=42),
    }

    metrics = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, preds)
        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm}

    # Save all artifacts
    joblib.dump(models["XGBoost"],            os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    joblib.dump(models["Random Forest"],      os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(models["Logistic Regression"],os.path.join(MODEL_DIR, "lr_model.pkl"))
    joblib.dump(scaler,                        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feat_cols,                     os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(metrics,                       os.path.join(MODEL_DIR, "metrics.pkl"))

    return models, scaler, feat_cols, metrics


def load_models():
    """Load pre-trained models from disk, training if needed."""
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        train_and_save_models()
    xgb_model  = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    rf_model   = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    lr_model   = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feat_cols  = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    metrics    = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))
    return {"XGBoost": xgb_model, "Random Forest": rf_model, "Logistic Regression": lr_model}, \
           scaler, feat_cols, metrics


# ──────────────────────────────────────────────────────────────────────────────
# TEAM STATS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_TEAM_STATS_CACHE = None

def get_team_stats() -> pd.DataFrame:
    """Build per-team aggregate stats from match features (cached)."""
    global _TEAM_STATS_CACHE
    if _TEAM_STATS_CACHE is not None:
        return _TEAM_STATS_CACHE
    df = load_match_features()
    form_df = load_teams_form()

    records = []
    all_teams = sorted(set(df["_home_team"]) | set(df["_away_team"]))

    for team in all_teams:
        home = df[df["_home_team"] == team]
        away = df[df["_away_team"] == team]

        home_wins  = (home["home_goals"] > home["away_goals"]).sum()
        away_wins  = (away["away_goals"] > away["home_goals"]).sum()
        home_draws = (home["home_goals"] == home["away_goals"]).sum()
        away_draws = (away["home_goals"] == away["away_goals"]).sum()
        total_matches = len(home) + len(away)
        total_wins    = home_wins + away_wins
        total_draws   = home_draws + away_draws
        total_losses  = total_matches - total_wins - total_draws

        goals_scored    = home["home_goals"].sum() + away["away_goals"].sum()
        goals_conceded  = home["away_goals"].sum() + away["home_goals"].sum()

        latest_elo = None
        if len(home) > 0:
            latest_elo = home.sort_values("_date")["home_elo"].iloc[-1]
        elif len(away) > 0:
            latest_elo = away.sort_values("_date")["away_elo"].iloc[-1]

        # Get last form
        team_form = form_df[form_df["team"] == team].sort_values("match_date")
        last_win_rate = team_form["win_rate"].iloc[-1] if len(team_form) > 0 else 0.0
        last_avg_scored = team_form["avg_goals_scored"].iloc[-1] if len(team_form) > 0 else 0.0

        records.append({
            "team": team,
            "matches": total_matches,
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_losses,
            "win_rate": round(total_wins / total_matches, 3) if total_matches else 0,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_diff": goals_scored - goals_conceded,
            "elo": round(latest_elo, 1) if latest_elo else 1500,
            "form_win_rate": last_win_rate,
            "form_avg_goals": last_avg_scored,
        })

    _TEAM_STATS_CACHE = pd.DataFrame(records)
    return _TEAM_STATS_CACHE


def get_team_row(team: str, match_df: pd.DataFrame, is_home: bool) -> dict:
    """Extract latest feature row for a given team (for prediction)."""
    col = "_home_team" if is_home else "_away_team"
    rows = match_df[match_df[col] == team].sort_values("_date")
    if len(rows) == 0:
        return {}
    return rows.iloc[-1].to_dict()


def _build_team_cache():
    """Pre-build per-team last-row lookup for fast feature access."""
    global _TEAM_HOME_ROW, _TEAM_AWAY_ROW
    if _TEAM_HOME_ROW:
        return
    df = load_match_features()
    df = engineer_features(df)
    for team, grp in df.groupby("_home_team"):
        _TEAM_HOME_ROW[team] = grp.sort_values("_date").iloc[-1].to_dict()
    for team, grp in df.groupby("_away_team"):
        _TEAM_AWAY_ROW[team] = grp.sort_values("_date").iloc[-1].to_dict()


def build_prediction_features(home_team: str, away_team: str, neutral: bool = True) -> pd.DataFrame:
    """Build a single-row feature DataFrame for match prediction (fast cached version)."""
    global _FEAT_COLS_CACHE
    _build_team_cache()

    if _FEAT_COLS_CACHE is None:
        feat_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
        if os.path.exists(feat_path):
            _FEAT_COLS_CACHE = joblib.load(feat_path)
        else:
            return None

    feat_cols = _FEAT_COLS_CACHE
    h_row = _TEAM_HOME_ROW.get(home_team) or _TEAM_AWAY_ROW.get(home_team, {})
    a_row = _TEAM_AWAY_ROW.get(away_team) or _TEAM_HOME_ROW.get(away_team, {})

    if not h_row or not a_row:
        return None

    row = {
        "elo_diff":             float(h_row.get("home_elo", 1500)) - float(a_row.get("away_elo", 1500)),
        "overall_diff":         float(h_row.get("home_avg_overall", 70)) - float(a_row.get("away_avg_overall", 70)),
        "attack_diff":          float(h_row.get("home_avg_attack", 70)) - float(a_row.get("away_avg_attack", 70)),
        "defense_diff":         float(h_row.get("home_avg_defense", 70)) - float(a_row.get("away_avg_defense", 70)),
        "home_form_scored":     float(h_row.get("home_form_scored", 1.0)),
        "home_form_conceded":   float(h_row.get("home_form_conceded", 1.0)),
        "home_form_win_rate":   float(h_row.get("home_form_win_rate", 0.5)),
        "away_form_scored":     float(a_row.get("away_form_scored", 1.0)),
        "away_form_conceded":   float(a_row.get("away_form_conceded", 1.0)),
        "away_form_win_rate":   float(a_row.get("away_form_win_rate", 0.5)),
        "is_neutral":           int(neutral),
        "is_world_cup":         1,
        "is_continental":       0,
        "home_avg_pace":        float(h_row.get("home_avg_pace", 72)),
        "home_avg_shooting":    float(h_row.get("home_avg_shooting", 65)),
        "home_avg_passing":     float(h_row.get("home_avg_passing", 68)),
        "away_avg_pace":        float(a_row.get("away_avg_pace", 72)),
        "away_avg_shooting":    float(a_row.get("away_avg_shooting", 65)),
        "away_avg_passing":     float(a_row.get("away_avg_passing", 68)),
        "momentum_diff":        float(h_row.get("home_form_win_rate", 0.5)) - float(a_row.get("away_form_win_rate", 0.5)),
        "goal_diff_form":       (float(h_row.get("home_form_scored", 1)) - float(h_row.get("home_form_conceded", 1))) -
                                (float(a_row.get("away_form_scored", 1)) - float(a_row.get("away_form_conceded", 1))),
        "elo_advantage":        1.0 if float(h_row.get("home_elo", 1500)) - float(a_row.get("away_elo", 1500)) > 100
                                else (0.0 if float(h_row.get("home_elo", 1500)) - float(a_row.get("away_elo", 1500)) < -100 else 0.5),
    }

    available = [c for c in feat_cols if c in row]
    return pd.DataFrame([row])[available]


def predict_match(home_team: str, away_team: str, neutral: bool = True):
    """Return (home_prob, draw_prob, away_prob) using XGBoost.
    For neutral venues, averages both team-order directions for a symmetric result.
    """
    models, scaler, feat_cols, _ = load_models()
    model = models["XGBoost"]

    X_ab = build_prediction_features(home_team, away_team, neutral)
    if X_ab is None:
        return 0.40, 0.30, 0.30
    probs_ab = model.predict_proba(X_ab)[0]
    if len(probs_ab) == 2:
        probs_ab = [probs_ab[0], 0.20, probs_ab[1]]

    if neutral:
        # Also predict with teams swapped, then average symmetrically
        X_ba = build_prediction_features(away_team, home_team, neutral)
        if X_ba is not None:
            probs_ba = model.predict_proba(X_ba)[0]
            if len(probs_ba) == 2:
                probs_ba = [probs_ba[0], 0.20, probs_ba[1]]
            # probs_ba[0] = P(away_team wins), probs_ba[2] = P(home_team wins)
            h = (float(probs_ab[0]) + float(probs_ba[2])) / 2
            d = (float(probs_ab[1]) + float(probs_ba[1])) / 2
            a = (float(probs_ab[2]) + float(probs_ba[0])) / 2
            total = h + d + a
            return h / total, d / total, a / total

    total = sum(float(p) for p in probs_ab)
    return float(probs_ab[0]) / total, float(probs_ab[1]) / total, float(probs_ab[2]) / total


def get_all_teams() -> list:
    df = load_match_features()
    return sorted(set(df["_home_team"]) | set(df["_away_team"]))
