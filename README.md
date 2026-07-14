# Football Elo Predictor

Machine learning pipeline that predicts the outcome of international football matches (home win / draw / away win). Team strength is modeled with an Elo rating system computed over 150+ years of match history, combined with rolling-form and head-to-head features, then fed to Logistic Regression and XGBoost classifiers — evaluated on held-out FIFA World Cup matches.

## How it works

```
data/results.csv ──► Elo engine ──► feature engineering ──► temporal split ──► models
                     (src/elo.py)     (src/features.py)         (src/train.py)
```

**1. Elo rating engine** ([src/elo.py](src/elo.py))
Implements the [World Football Elo Ratings](https://www.eloratings.net/about) methodology: a logistic expected-score curve, K-factors weighted by tournament importance (World Cup matches move ratings 3x more than friendlies), a margin-of-victory multiplier with diminishing returns, and a home-advantage bonus for non-neutral venues.

**2. Feature engineering** ([src/features.py](src/features.py))
- Pre-match Elo ratings and Elo difference
- Rolling form over each team's last 5 matches: points per game, goals scored, goals conceded
- Head-to-head win rate for the specific matchup
- Neutral-venue flag

**3. Training & evaluation** ([src/train.py](src/train.py))
Trains multinomial Logistic Regression and XGBoost, and compares both against a naive "always predict home win" baseline on the held-out World Cup test set.

### No data leakage, by construction

Matches are processed in strict chronological order, and every rating and feature is recorded *before* the match result is applied — each row reflects only what was knowable walking into that match. The train/test split is temporal as well: models are evaluated on real World Cup matches (2022, 2026) that the training data never sees.

## Getting started

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Get the data**

Download the [International football results, 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) dataset from Kaggle and place `results.csv` in the `data/` directory.

**3. Run the pipeline**

```bash
python src/train.py
```

The script prints dataset stats, the train/test split sizes, and accuracy + log-loss for each model against the naive baseline.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost
