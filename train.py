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


if __name__ == "__main__":
    main()
