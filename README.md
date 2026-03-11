# LoL Rank Predictor

A web application that predicts a League of Legends player's rank using a trained ML model based on their recent ranked match history. The result is complemented by a full analytics dashboard — performance comparison against the predicted rank average, z-scores, percentiles, live game analysis, and champion recommendations.

**Live demo:** [machine-learning-lol-analysis-prediction.onrender.com](https://machine-learning-lol-analysis-prediction.onrender.com/)

---

## What the app does

1. The user enters their Riot ID in the format `Name#TAG` and selects a region.
2. The backend fetches the player's last 10 ranked games via the Riot API.
3. Aggregated metrics are computed from the games (means, medians, standard deviations of KDA, CS/min, damage/min, etc.).
4. A trained ML model predicts the player's rank (Iron–Diamond).
5. The frontend displays:
   - Predicted rank with a confidence score and comparison to the player's actual rank
   - Per-rank probabilities (bar chart)
   - Performance radar chart
   - Comparison against the predicted rank average (z-score, percentile, deviations)
   - Metrics breakdown by category: Combat, Farming, Vision, Survivability, Economy
   - Performance distance from every rank (Iron–Diamond)
   - Estimated number of games needed to reach the next tier
   - Champion recommendations suited to the player's playstyle
   - Last 10 match history
   - KDA and CS/min trend over matches
   - Live game analysis (if the player is currently in a game): predicted ranks for all players on both teams and estimated win probability

---

## Project structure

```
MLProjektLolPredikce/
├── backend/                    # FastAPI backend
│   ├── config.py               # Configuration (API key, regions, paths)
│   ├── main.py                 # FastAPI app, static files, root endpoint
│   ├── models/
│   │   └── schemas.py          # Pydantic input validation
│   ├── routers/
│   │   └── predict.py          # POST /api/predict and POST /api/live-game endpoints
│   └── services/
│       ├── riot_api.py         # Riot API client (account, match history, parsing)
│       ├── predictor.py        # Model loading, inference, model switching
│       ├── analytics.py        # Rank average comparison (z-score, percentile)
│       ├── live_game.py        # Live game analysis, team prediction, win probability
│       ├── champion_recommender.py  # Champion recommendations (cosine similarity)
│       └── rank_progression.py     # Next-tier progression estimation
│
├── Training/                   # Data and ML pipeline
│   ├── checkpoint/
│   │   └── checkpoint.json     # Data collection state
│   ├── collection/
│   │   └── lol_data_collection.ipynb  # Data collection notebook (Riot API)
│   ├── datasets/
│   │   └── lol_rank_dataset.csv       # Training dataset (~38,000 games, 2,065 players)
│   ├── models/                 # Generated training artifacts
│   │   ├── model.pkl           # Best model (selected automatically)
│   │   ├── model_<name>.pkl    # Individual model artifacts
│   │   ├── scaler.pkl
│   │   ├── rank_stats.pkl
│   │   ├── model_meta.json
│   │   └── models_comparison.json  # Model comparison results
│   ├── notebooks/              # Jupyter notebooks (one per model + comparison)
│   │   ├── 01_logistic_regression.ipynb
│   │   ├── 02_random_forest.ipynb
│   │   ├── 03_xgboost.ipynb
│   │   ├── 04_lightgbm.ipynb
│   │   ├── 05_gradient_boosting.ipynb
│   │   ├── 06_extra_trees.ipynb
│   │   ├── 07_knn.ipynb
│   │   ├── 08_svm.ipynb
│   │   ├── 09_mlp.ipynb
│   │   └── 10_model_comparison.ipynb
│   ├── preprocessing/
│   │   ├── cleaner.py          # Dataset loading and cleaning
│   │   └── feature_engineering.py  # Per-player aggregation, derived metrics
│   └── scripts/
│       ├── train_all_models.py # Train all models and select the best one
│       └── evaluate.py         # GroupKFold evaluation
│
├── frontend/                   # Static frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── requirements.txt
└── run.py                      # Server entry point
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, uvicorn |
| ML | scikit-learn, XGBoost, LightGBM |
| Data | pandas, numpy, scipy |
| API communication | httpx |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |
| Deployment | Render |

---

## ML models

`train_all_models.py` trains 9 models, compares their CV accuracy and F1 scores, and automatically saves the best one as `model.pkl`. Results are stored in `Training/models/models_comparison.json`.

| # | Model | CV Accuracy | F1 Score |
|---|-------|------------|----------|
| 1 | Logistic Regression | 39.8% | 39.8% |
| 2 | Neural Network (MLP) | 38.5% | 38.5% |
| 3 | **XGBoost** *(currently deployed)* | 37.9% | **39.6%** |
| 4 | Random Forest | 37.8% | 36.9% |
| 5 | LightGBM | 37.3% | 38.7% |
| 6 | Support Vector Machine | 37.2% | 39.2% |
| 7 | Extra Trees | 36.7% | 36.2% |
| 8 | Gradient Boosting | 35.6% | 36.2% |
| 9 | K-Nearest Neighbors | 29.7% | 34.1% |

> ~40% accuracy across 7 classes (Iron–Diamond) is significantly better than a random classifier (14%). Rank in LoL depends on many factors beyond measurable in-game metrics, which inherently limits the achievable accuracy.

---

## Installation and setup

### 1. Requirements

- Python 3.10+
- pip

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate on Windows:
```bash
.venv\Scripts\activate
```

Activate on macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

The script trains all models, compares them, and saves the best one as `model.pkl`. Artifacts are saved to `Training/models/`.

```bash
python Training/scripts/train_all_models.py
```

The output shows CV accuracy and F1 score for each model. Artifacts (`model.pkl`, `model_<name>.pkl`, `scaler.pkl`, `rank_stats.pkl`, `model_meta.json`, `models_comparison.json`) will appear in `Training/models/`.

### 5. Start the server

```bash
python run.py
```

The server runs at [http://localhost:8000](http://localhost:8000).

---

## Dataset

The dataset contains approximately **38,000 ranked games** from **2,065 players** on the EUW server, evenly distributed across ranks Iron through Diamond (~295 players per rank).

Each row represents a single game and contains metrics such as `kda`, `cs_per_min`, `damage_per_min`, `gold_per_min`, `vision_per_min`, `deaths_per_min`, `damage_taken_per_min`, and basic kills/deaths/assists stats.

For training, games are aggregated at the player level — means, medians, and standard deviations of key metrics are computed across all games. The model therefore predicts rank from a player's performance profile, not a single game.

A GroupKFold split ensures that games from the same player are never simultaneously in the training and test sets (no data leakage).

---

## API

### `POST /api/predict`

**Request body:**
```json
{
  "game_name": "Name",
  "tag_line": "EUW",
  "region": "euw"
}
```

**Response:**
```json
{
  "summoner": { "game_name": "...", "tag_line": "..." },
  "prediction": {
    "predicted_rank": "GOLD",
    "confidence": 0.42,
    "rank_probabilities": { "IRON": 0.02, "BRONZE": 0.08, "...": "..." }
  },
  "player_stats": { "kda_mean": 3.2, "cs_per_min_mean": 6.1, "...": "..." },
  "comparison": {
    "combat": [ { "metric": "kda_mean", "label": "KDA", "player_value": 3.2, "rank_average": 2.8, "diff_percent": 14.3, "z_score": 0.9, "percentile": 81.6, "above_average": true } ],
    "...": "..."
  },
  "all_ranks_distance": { "IRON": 0.12, "BRONZE": 0.31, "...": "..." },
  "matches": [ { "champion": "Jinx", "role": "BOTTOM", "kills": 8, "deaths": 2, "...": "..." } ],
  "record": { "wins": 7, "losses": 3, "total": 10 },
  "real_rank": { "tier": "GOLD", "rank": "II", "leaguePoints": 45 },
  "rank_progression": { "games_to_next_tier": 18, "current_tier": "GOLD", "next_tier": "PLATINUM" },
  "champion_recommendations": {
    "player_champions": [ { "champion": "Jinx", "games": 5, "winrate": 60.0 } ],
    "recommendations": [ { "champion": "Caitlyn", "similarity": 0.94, "avg_winrate": 0.52 } ]
  }
}
```

### `POST /api/live-game`

Returns live game analysis if the player is currently in a game.

**Request body:** same format as `/api/predict`

**Response:**
```json
{
  "in_game": true,
  "game_mode": "CLASSIC",
  "blue_team": [ { "summoner_name": "...", "champion_name": "Jinx", "predicted_rank": "GOLD", "confidence": 0.41, "winrate": 55.0 } ],
  "red_team": [ { "...": "..." } ],
  "searched_team": "blue",
  "blue_win_pct": 54.2,
  "red_win_pct": 45.8
}
```

Returns `{ "in_game": false }` if the player is not currently in a game.

Available regions: `euw`, `eune`, `na`, `kr`, `jp`, `br`, `lan`, `las`, `oce`, `tr`, `ru`

---

## Notes

- The Riot API key in `backend/config.py` is a development key with a limit of 20 requests per second. A production key is required for production use — obtain one at [developer.riotgames.com](https://developer.riotgames.com).
- The app filters exclusively for ranked games (queue ID 420). Normals, ARAM, and other modes are ignored.
- If a player has fewer than 3 ranked games in their history, the API returns an error displayed in the UI.
- Champion recommendations are based on cosine similarity between the player's aggregate performance profile and champion profiles built from the dataset (minimum 20 games per champion).
