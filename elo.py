"""
Elo rating engine for international football.

Adapted from the World Football Elo Ratings methodology (eloratings.net):
- expected result = logistic curve based on the rating gap
- actual result = 1.0 / 0.5 / 0.0 for win / draw / loss
- delta = K * margin-of-victory multiplier * (actual - expected)
- winner's rating += delta, loser's rating -= delta

Each team's rating is recorded BEFORE it is updated for the current match,
so every feature reflects team strength walking INTO that match — no
leakage from the match's own result.
"""

import pandas as pd
import numpy as np

BASE_RATING = 1500
HOME_ADVANTAGE = 60  # elo points added to home team when not neutral

# Tournament importance weight (bigger K = ratings move more after these matches)
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 55,
    "UEFA Euro qualification": 35,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "African Cup of Nations qualification": 35,
    "AFC Asian Cup": 50,
    "AFC Asian Cup qualification": 35,
    "Gold Cup": 45,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 35,
    "Friendly": 20,
}
DEFAULT_WEIGHT = 30


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Bigger blowouts move ratings more, but with diminishing returns
    (a 6-0 shouldn't count 6x as much as a 1-0). Matches the margin-of-victory
    adjustment used by the World Football Elo Ratings (eloratings.net).
    """
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8


def compute_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through `matches` in chronological order, computing each team's
    Elo rating match by match. For every row, elo_home/elo_away are recorded
    BEFORE the result is applied, so they always reflect each team's strength
    walking INTO that match (no data leakage from future results).

    Returns
    -------
    matches, with elo_home and elo_away columns added. The final rating for
    every team is also attached via matches.attrs["final_ratings"].
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    ratings = {}

    elo_home_col = np.zeros(len(matches))
    elo_away_col = np.zeros(len(matches))

    for i, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]

        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        elo_home_col[i] = r_home
        elo_away_col[i] = r_away

        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue  # future fixture: rating recorded above, nothing to update

        home_adj = r_home + (HOME_ADVANTAGE if not row["neutral"] else 0)
        expected_home = 1 / (1 + 10 ** ((r_away - home_adj) / 400))
        actual_home = (
            1.0 if row["home_score"] > row["away_score"]
            else 0.5 if row["home_score"] == row["away_score"]
            else 0.0
        )
        K = TOURNAMENT_WEIGHT.get(row["tournament"], DEFAULT_WEIGHT)
        mult = goal_diff_multiplier(abs(row["home_score"] - row["away_score"]))
        delta = K * mult * (actual_home - expected_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    matches["elo_home"] = elo_home_col
    matches["elo_away"] = elo_away_col
    matches.attrs["final_ratings"] = ratings
    return matches
