"""
Feature engineering on top of the Elo-augmented match history.

Same rule as elo.py: every feature only uses information available BEFORE
the current match. Matches are processed in chronological order while
running state (per-team form, per-matchup head-to-head record) is built up
incrementally.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'result' column: 0 = away win, 1 = draw, 2 = home win.
    Matches with no score yet (future fixtures) get NaN — comparisons against
    NaN are always False, so np.select falls through to `default` for them.
    """
    df = df.copy()
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    choices = [2, 1, 0]
    df["result"] = np.select(conditions, choices, default=np.nan)
    return df


def add_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each team, compute (evaluated BEFORE the current match) their form
    over the last `window` matches: points-per-game (win=3, draw=1, loss=0),
    average goals scored, and average goals conceded.

    Per-team history is kept in a dict of fixed-size deques, so pushing a
    new result automatically drops the oldest one once `window` is reached.
    Teams with no history yet get a neutral prior (ppg=1.0, gs=1.0, gc=1.0)
    rather than being assumed great or terrible.

    Adds: home_form_ppg, home_form_gs, home_form_gc,
          away_form_ppg, away_form_gs, away_form_gc
    """
    df = df.sort_values("date").reset_index(drop=True)
    history = defaultdict(lambda: deque(maxlen=window))

    def summarize(hist):
        if len(hist) == 0:
            return 1.0, 1.0, 1.0  # neutral prior for teams with no history yet
        pts = sum(h[0] for h in hist) / len(hist)
        gs = sum(h[1] for h in hist) / len(hist)
        gc = sum(h[2] for h in hist) / len(hist)
        return pts, gs, gc

    home_ppg, home_gs, home_gc = [], [], []
    away_ppg, away_gs, away_gc = [], [], []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]

        h_pts, h_gs, h_gc = summarize(history[home])
        a_pts, a_gs, a_gc = summarize(history[away])
        home_ppg.append(h_pts); home_gs.append(h_gs); home_gc.append(h_gc)
        away_ppg.append(a_pts); away_gs.append(a_gs); away_gc.append(a_gc)

        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            hs, as_ = row["home_score"], row["away_score"]
            if hs > as_:
                h_points, a_points = 3, 0
            elif hs == as_:
                h_points = a_points = 1
            else:
                h_points, a_points = 0, 3
            history[home].append((h_points, hs, as_))
            history[away].append((a_points, as_, hs))

    df["home_form_ppg"] = home_ppg
    df["home_form_gs"] = home_gs
    df["home_form_gc"] = home_gc
    df["away_form_ppg"] = away_ppg
    df["away_form_gs"] = away_gs
    df["away_form_gc"] = away_gc
    return df


def add_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'h2h_home_winrate': the home team's historical win rate specifically
    against this opponent, evaluated BEFORE the current match. Teams with no
    prior meetings get a neutral 0.5 prior.

    Each matchup is keyed by tuple(sorted([home, away])) so "A vs B" and
    "B vs A" share the same history bucket, with [wins, draws, losses]
    tracked from the perspective of whichever team is first alphabetically.
    """
    df = df.sort_values("date").reset_index(drop=True)
    h2h = defaultdict(lambda: [0, 0, 0])  # [wins, draws, losses] for the alphabetically-first team

    winrates = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        key = tuple(sorted([home, away]))
        first_team = key[0]
        w, d, l = h2h[key]
        total = w + d + l

        if total == 0:
            winrates.append(0.5)
        elif home == first_team:
            winrates.append((w + 0.5 * d) / total)
        else:
            winrates.append((l + 0.5 * d) / total)

        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            hs, as_ = row["home_score"], row["away_score"]
            if hs == as_:
                h2h[key][1] += 1
            elif (hs > as_ and home == first_team) or (as_ > hs and away == first_team):
                h2h[key][0] += 1
            else:
                h2h[key][2] += 1

    df["h2h_home_winrate"] = winrates
    return df


FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_ppg", "home_form_gs", "home_form_gc",
    "away_form_ppg", "away_form_gs", "away_form_gc",
    "h2h_home_winrate",
    "is_neutral",
]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    This one's done for you — just combines a couple of columns you'll
    already have by this point.
    """
    df = df.copy()
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["is_neutral"] = df["neutral"].astype(int)
    return df
