"""
End-to-end training pipeline:

1. Load match data and compute pre-match Elo ratings (src/elo.py)
2. Engineer rolling-form and head-to-head features (src/features.py)
3. Hold out actual World Cup matches (2022, 2026) as the test set
4. Train Logistic Regression and XGBoost, evaluate against a naive baseline

Run with:  python src/train.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from elo import compute_elo_history
from features import add_result_label, add_rolling_form, add_head_to_head, build_feature_frame, FEATURE_COLUMNS

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper() == "TRUE"
    return df


def build_dataset(df):
    df = compute_elo_history(df)
    df = add_result_label(df)
    df = add_rolling_form(df, window=5)
    df = add_head_to_head(df)
    df = build_feature_frame(df)
    return df


def temporal_split(df, test_tournament="FIFA World Cup", test_years=(2022, 2026)):
    played = df.dropna(subset=["result"]).copy()
    unplayed = df[df["result"].isna()].copy()
    is_test_wc = (played["tournament"] == test_tournament) & (played["date"].dt.year.isin(test_years))
    test = played[is_test_wc]
    train = played[~is_test_wc]
    return train, test, unplayed


def train_models(train, test):
    """
    Scale features, fit Logistic Regression and XGBoost, and evaluate both
    against a naive "always predict home win" baseline.

    Returns a dict: {model_name: {"accuracy": ..., "log_loss": ...}}
    """
    X_train = train[FEATURE_COLUMNS].values
    y_train = train["result"].astype(int).values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test["result"].astype(int).values

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)

    naive_pred = np.full_like(y_test, fill_value=2)  # always predict home win

    results = {}
    for name, pred, proba in [
        ("Naive (always home win)", naive_pred, None),
        ("Logistic Regression", logreg_pred, logreg_proba),
        ("XGBoost", xgb_pred, xgb_proba),
    ]:
        acc = accuracy_score(y_test, pred)
        ll = log_loss(y_test, proba, labels=[0, 1, 2]) if proba is not None else None
        results[name] = {"accuracy": acc, "log_loss": ll}

    return results


def train_final_model(played):
    """
    Train the model actually used for predicting the future.

    train_models() above deliberately holds out the 2022/2026 World Cup
    matches so we can measure accuracy on games the model never trained on.
    But once you're happy with that number and want to predict *upcoming*
    fixtures, you want the model to learn from every match played so far —
    including the 2026 games that already happened, since those are now
    real history, not a leak. This retrains XGBoost on all of `played`.
    """
    X = played[FEATURE_COLUMNS].values
    y = played["result"].astype(int).values

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


def swap_home_away(features):
    """
    Mirror a feature frame as if home and away teams were swapped: exchange
    the elo/form columns, negate elo_diff, and flip the h2h winrate
    (away's winrate against home = 1 - home's winrate against away).
    """
    swapped = features.copy()
    swapped["elo_home"], swapped["elo_away"] = features["elo_away"], features["elo_home"]
    swapped["elo_diff"] = -features["elo_diff"]
    for col in ("ppg", "gs", "gc"):
        swapped[f"home_form_{col}"] = features[f"away_form_{col}"]
        swapped[f"away_form_{col}"] = features[f"home_form_{col}"]
    swapped["h2h_home_winrate"] = 1 - features["h2h_home_winrate"]
    return swapped


def predict_upcoming(unplayed, model, test_tournament="FIFA World Cup"):
    """
    Predict outcomes for fixtures with no result yet (future matches found
    by temporal_split). Returns a DataFrame with a win/draw/win probability
    per fixture, or an empty DataFrame if there's nothing to predict.

    These are neutral-venue matches where "home" is just FIFA's nominal
    labeling, so each fixture is scored in both orientations and averaged —
    otherwise the model's learned bias toward the listed home team (hosts
    and higher seeds are over-represented as nominal home in the history)
    would leak into the probabilities.
    """
    upcoming = unplayed[unplayed["tournament"] == test_tournament].copy()
    # Fixtures whose opponents aren't decided yet (e.g. the final before the
    # semis are played) have NaN teams — no real features to predict from.
    upcoming = upcoming.dropna(subset=["home_team", "away_team"])
    if upcoming.empty:
        return upcoming

    features = upcoming[FEATURE_COLUMNS]
    proba = model.predict_proba(features.values)  # columns are [away_win, draw, home_win]
    proba_swapped = model.predict_proba(swap_home_away(features)[FEATURE_COLUMNS].values)
    upcoming["p_away_win"] = (proba[:, 0] + proba_swapped[:, 2]) / 2
    upcoming["p_draw"] = (proba[:, 1] + proba_swapped[:, 1]) / 2
    upcoming["p_home_win"] = (proba[:, 2] + proba_swapped[:, 0]) / 2

    return upcoming[["date", "home_team", "away_team", "city",
                      "p_home_win", "p_draw", "p_away_win"]].sort_values("date")


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} matches loaded, {df['date'].min().date()} to {df['date'].max().date()}")

    print("Computing ELO ratings + features...")
    df = build_dataset(df)

    print("Splitting train/test...")
    train, test, unplayed = temporal_split(df)
    print(f"  Train: {len(train):,} | Test: {len(test):,} | Unplayed/future: {len(unplayed):,}")

    print("Training models...")
    results = train_models(train, test)

    print("\nResults on held-out World Cup matches:")
    for name, metrics in results.items():
        ll = f"{metrics['log_loss']:.3f}" if metrics["log_loss"] is not None else "n/a"
        print(f"  {name:28s} accuracy={metrics['accuracy']:.3f}  log_loss={ll}")

    print("\nTraining final model on all played matches (for live predictions)...")
    played = df.dropna(subset=["result"])
    final_model = train_final_model(played)

    print("\nPredicting upcoming World Cup fixtures...")
    upcoming = predict_upcoming(unplayed, final_model)
    if upcoming.empty:
        print("  No upcoming World Cup fixtures found in the dataset.")
    else:
        print(upcoming.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
