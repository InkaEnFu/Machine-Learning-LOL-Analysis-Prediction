# LoL Rank Predictor — Project Documentation

| Field | Value |
|---|---|
| **Project Name** | LoL Rank Predictor |
| **Author** | Matyáš Prokop |
| **Date** | March 2026 |
| **School** | Střední průmyslová škola elektrotechnická, Praha 2, Ječná 30 |
| **Project Type** | School project — Machine Learning / Web Application |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Functional Requirements](#2-functional-requirements)
3. [Non-Functional Requirements & Third-Party Dependencies](#3-non-functional-requirements--third-party-dependencies)
4. [Application Architecture](#4-application-architecture)
5. [Application Behaviour — Activity Diagrams](#5-application-behaviour--activity-diagrams)
6. [Network Schema & API Communication](#6-network-schema--api-communication)
7. [Configuration](#7-configuration)
8. [Installation & Running](#8-installation--running)
9. [Error States & Error Handling](#9-error-states--error-handling)
10. [Testing & Validation](#10-testing--validation)
11. [Versions & Known Issues](#11-versions--known-issues)
12. [Legal & Licensing](#12-legal--licensing)

---

## 1. Project Overview

LoL Rank Predictor is a full-stack web application that uses machine learning to predict a League of Legends player's competitive rank (Iron through Diamond) based on their recent ranked match history.

The user enters their Riot ID (`GameName#TAG`) and selects a region. The backend fetches the player's last 10 ranked games via the official Riot Games API, computes aggregated performance metrics, passes them through a trained ML classifier, and returns a prediction alongside a rich analytics dashboard.

**Live demo:** [machine-learning-lol-analysis-prediction.onrender.com](https://machine-learning-lol-analysis-prediction.onrender.com/)

### What the application shows

- Predicted rank with confidence score and comparison to the player's actual rank
- Per-rank probability distribution (bar chart)
- Performance radar chart
- Per-metric comparison against the predicted rank average (z-score, percentile, deviation)
- Metrics breakdown by category: Combat, Farming, Vision, Survivability, Economy
- Performance distance from every rank (Iron–Diamond)
- Estimated games needed to reach the next tier
- Champion recommendations based on cosine similarity of playstyles
- Last 10 match history with KDA and CS/min trend charts
- Live game analysis (if in game): predicted ranks for all 10 players and estimated win probability

---

## 2. Functional Requirements

### Business Requirements

| ID | Requirement |
|---|---|
| BR-01 |The software shall be executable on school workstation PCs without the requirement of an Integrated Development Environment (IDE).|
| BR-02 | The system shall provide a meaningful real-world utility, supported by a user manual and technical documentation for all primary functions. |
| BR-03 | The system shall incorporate a machine learning model supported by a documented development process (e.g., Google Colab or Jupyter Notebook). |
| BR-04 | The model shall be trained exclusively on an original, non-preprocessed dataset of at least 1500 records and 5 attributes, manually collected by the author. |
| BR-05 | The documentation shall explicitly detail the data source and collection methodology (e.g., web crawling, video analysis, or hardware sensors). |
| BR-06 | The system shall demonstrate a complete data preprocessing pipeline, including data cleaning, scaling, and transformation of the raw collected data. |
| BR-07 | The model shall provide verifiable and useful outputs, such as accurate classifications, regressions, or other data-driven predictions. |
| BR-08 | The source code shall maintain a strict separation between original authorship and third-party libraries by utilizing dedicated directories (e.g., /lib or /vendor). |


### Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | The user can enter a Riot ID in the format `Name#TAG` and choose one of 11 supported regions. |
| FR-02 | The backend fetches the 10 most recent ranked (queue 420) matches from the Riot Match v5 API. |
| FR-03 | The system aggregates mean, standard deviation, and median of 10 per-game features per player. |
| FR-04 | The system runs the aggregated feature vector through a trained ML classifier and returns a rank label (IRON–DIAMOND). |
| FR-05 | The system computes z-scores and percentile ranks against the predicted tier's distribution. |
| FR-06 | The system fetches the player's actual rank via the Riot League v4 API and compares it to the prediction. |
| FR-07 | The system estimates games to next tier based on current LP, division, and recent win rate. |
| FR-08 | The system recommends up to 3 champions using cosine similarity against a champion profile database. |
| FR-09 | The POST `/api/live-game` endpoint analyses an active game if the player is currently in one; returns `in_game: false` otherwise. |
| FR-10 | The system exposes all trained models and allows the frontend to switch between them via the `model_name` request field. |



## 3. Non-Functional Requirements & Third-Party Dependencies

### Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | API responses must complete within 15 seconds under normal network conditions. |
| NFR-02 | The application must handle Riot API rate limiting (HTTP 429) with automatic back-off. |
| NFR-03 | The ML models must be loaded into memory on first request and cached for subsequent requests. |
| NFR-04 | The frontend must function in all modern browsers without additional plugins. |
| NFR-05 | A valid Riot Games API key must be provided via the `RIOT_API_KEY` environment variable. |

### Third-Party Libraries

| Library | Version | Purpose |
|---|---|---|
| **fastapi** | latest | Web framework and REST API server |
| **uvicorn[standard]** | latest | ASGI server for running FastAPI |
| **pandas** | latest | Data manipulation and CSV handling |
| **numpy** | latest | Numerical computation, array operations |
| **scikit-learn** | latest | ML algorithms, preprocessing, evaluation |
| **xgboost** | latest | XGBoost gradient-boosted classifier |
| **lightgbm** | latest | LightGBM gradient-boosted classifier |
| **httpx** | latest | Asynchronous/synchronous HTTP client for Riot API calls |
| **python-dotenv** | latest | Loading `.env` configuration files |
| **scipy** | latest | Statistical functions (CDF for percentile computation) |

Frontend (loaded from CDN, no installation required):

| Library | Purpose |
|---|---|
| **Chart.js** | Bar charts, radar charts, trend line charts |

### External Services

| Service | Purpose |
|---|---|
| **Riot Games API** (`api.riotgames.com`) | Player accounts, match history, live game, league standings |
| **Riot Data Dragon** (`ddragon.leagueoflegends.com`) | Champion names, champion images, game version |

---


## 4. Project Structure

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

### You can find all these diagrams in the Diagrams.Drawio:
 - Architecture
 - Use Case
 - Actiivte: Prediction Flow
 - Activity: Live Game Flow
 - Activity: ML Training Pipeline
 - Network Topology
 - Data Collcetion

### Key Design Patterns

| Pattern | Where used |
|---|---|
| **Service Layer** | All business logic is in `backend/services/`; routers only orchestrate calls. |
| **Repository / Lazy Load** | Models are loaded from disk only on first request and then cached in module-level globals. |
| **Strategy** | `predictor.py` supports switching between 10 trained models at runtime via `model_name`. |
| **DTO / Schema** | `PredictRequest` (Pydantic) validates all incoming API data before any processing. |
| **Pipeline** | Training follows a strict pipeline: collect → clean → engineer → train → evaluate → save. |

### REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `frontend/index.html` |
| `GET` | `/static/*` | Serves static CSS/JS assets |
| `POST` | `/api/predict` | Main prediction endpoint |
| `POST` | `/api/live-game` | Live game analysis endpoint |

#### POST `/api/predict` — Request Body

```json
{
  "game_name": "string  (1–24 chars)",
  "tag_line":  "string  (1–5 chars)",
  "region":    "string  (euw|eune|na|kr|jp|br|lan|las|oce|tr|ru)  default: euw"
}
```

Optional field (model switching):
```json
{
  "model_name": "string  (logistic_regression|random_forest|xgboost|lightgbm|gradient_boosting|extra_trees|knn|svm|mlp)"
}
```

### Riot API Endpoints Used

| Endpoint | Version | Purpose |
|---|---|---|
| `GET /riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}` | v1 | Resolve Riot ID to PUUID |
| `GET /riot/account/v1/accounts/by-puuid/{puuid}` | v1 | Resolve PUUID to display name |
| `GET /lol/match/v5/matches/by-puuid/{puuid}/ids` | v5 | Get ranked match ID list |
| `GET /lol/match/v5/matches/{matchId}` | v5 | Get full match details |
| `GET /lol/spectator/v5/active-games/by-summoner/{puuid}` | v5 | Get active game data |
| `GET /lol/league/v4/entries/by-summoner/{summonerId}` | v4 | Get real rank / LP |
| `GET /lol/summoner/v4/summoners/by-puuid/{puuid}` | v4 | Get summoner ID |

Base URLs are either `https://{regional}.api.riotgames.com` (regional) or `https://{platform}.api.riotgames.com` (platform-specific). Authentication is via the `X-Riot-Token` request header.

---

## 7. Configuration

### Environment Variables

The application is configured via a `.env` file in the project root (loaded by `python-dotenv`).

| Variable | Required | Description |
|---|---|---|
| `RIOT_API_KEY` | **Yes** | Riot Games API key. Obtain from [developer.riotgames.com](https://developer.riotgames.com). Development keys expire after 24 hours; production keys require an application. |

Create a `.env` file:
```
RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Region Mapping (`backend/config.py`)

| Region key (user input) | Platform code | Regional routing |
|---|---|---|
| `euw` | `euw1` | `europe` |
| `eune` | `eun1` | `europe` |
| `na` | `na1` | `americas` |
| `kr` | `kr` | `asia` |
| `jp` | `jp1` | `asia` |
| `br` | `br1` | `americas` |
| `lan` | `la1` | `americas` |
| `las` | `la2` | `americas` |
| `oce` | `oc1` | `americas` |
| `tr` | `tr1` | `europe` |
| `ru` | `ru` | `europe` |

### Model Directory

The `MODELS_DIR` variable in `config.py` resolves to `Training/models/` relative to the project root. This is where the training pipeline saves all `.pkl` and `.json` artifacts.

### Input Validation Rules

| Field | Constraint |
|---|---|
| `game_name` | 1–24 characters, stripped of leading/trailing whitespace |
| `tag_line` | 1–5 characters, stripped of leading/trailing whitespace |
| `region` | Must be one of the 11 keys in `REGION_TO_PLATFORM` |

### Server Configuration (`run.py`)

| Parameter | Value | Description |
|---|---|---|
| `host` | `0.0.0.0` | Listens on all interfaces |
| `port` | `8000` | TCP port |
| `reload` | `True` (dev mode) | Auto-reloads on file changes |

Change port or disable reload by editing `run.py` directly, or pass arguments to `uvicorn` manually.

---

## 8. Installation & Running

### Prerequisites

- Python 3.11 or newer
- A valid Riot Games API key

### Steps

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd MLProjektLolPredikce
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create the environment file**
   ```
   RIOT_API_KEY=RGAPI-your-key-here
   ```
   Save as `.env` in the project root.

4. **Train the models** (only required once; skip if `.pkl` files already exist in `Training/models/`)
   ```bash
   python Training/scripts/train_all_models.py
   ```
   This produces `model.pkl`, `scaler.pkl`, `rank_stats.pkl`, `model_meta.json`, and `models_comparison.json`.

5. **Start the server**
   ```bash
   python run.py
   ```
   The application is now available at `http://localhost:8000`.

6. **Open the frontend**
   Navigate to `http://localhost:8000` in a browser.

### Deployment (production)

The application is deployable on cloud platforms such as Render. Set the `RIOT_API_KEY` environment variable via the platform's secret management. The start command is:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 9. Error States & Error Handling

### HTTP Error Codes Returned by the API

| HTTP Status | Trigger | Message |
|---|---|---|
| `404 Not Found` | Player does not exist on Riot servers | `"Player not found"` |
| `404 Not Found` | Player has no ranked matches | `"No ranked matches found for this player"` |
| `404 Not Found` | Fewer than 3 ranked games collected | `"Not enough ranked matches found (found N, need at least 3)"` |
| `403 Forbidden` | Riot API key is expired or invalid | `"API key expired or invalid"` |
| `422 Unprocessable Entity` | Input validation failure (bad game_name, tag, or region) | FastAPI/Pydantic validation detail |
| `429 Too Many Requests` | Riot API rate limit reached after retries | `"Rate limit exceeded, try again later"` |
| `500 Internal Server Error` | Unexpected server-side error | Internal exception message |
| `503 Service Unavailable` | ML model `.pkl` file not found on disk | `"Model not found: <path>. Run: python Training/scripts/train.py"` |

### Internal Error Handling

| Situation | Behaviour |
|---|---|
| Individual match fetch fails during match collection loop | The match is silently skipped; processing continues with remaining matches. |
| Real rank fetch fails (League v4) | `real_rank` and `rank_progression` fields are returned as `null`; the prediction is still returned. |
| Live game participant has fewer than 2 matches | `predicted_rank` is `null`, `confidence` is `0`; the player entry is still included. |
| Model `predict_proba` not available | `probabilities` dict is empty; the predicted label is still returned. |
| Model feature importances not available | `feature_importance` dict is empty. |
| Riot API returns HTTP 429 | Automatic back-off using the `Retry-After` header value; retried up to 3 times. |
| Champion profiles CSV not found on disk | `recommend_champions` raises `FileNotFoundError` at module level; request fails with 503-equivalent. |

### Common User-Facing Error Messages

| Message | Cause | Resolution |
|---|---|---|
| *"Player not found"* | Typo in game name or tag, wrong region selected | Verify the Riot ID and selected region. |
| *"No ranked matches found"* | Player has never played ranked | The feature requires at least 3 recent ranked games. |
| *"API key expired or invalid"* | `.env` key has expired (development keys last 24 h) | Regenerate the key at developer.riotgames.com and update `.env`. |
| *"Rate limit exceeded"* | Too many concurrent requests to Riot API | Wait a few seconds and retry. |
| *"Model not found"* | Training was not run or models folder is missing | Run `python Training/scripts/train_all_models.py`. |

---

## 10. Testing & Validation

### Unit Tests

The `UnitTests/` directory contains 8 test modules using the standard `unittest` framework.

| File | What is tested |
|---|---|
| `test_01_aggregate_means.py` | Mean values computed correctly by `aggregate_player_games()` |
| `test_02_aggregate_winrate.py` | Win rate calculation from a match list |
| `test_03_aggregate_missing_columns.py` | Graceful handling of missing feature columns (defaults to 0) |
| `test_04_aggregate_unique_stats.py` | `unique_roles` and `unique_champions` counted correctly |
| `test_05_derived_vision_per_min.py` | `vision_per_min` derived feature computed from `visionScore` and `timePlayed` |
| `test_06_derived_damage_taken_per_min.py` | `damage_taken_per_min` derived feature computation |
| `test_07_lp_to_next_tier.py` | LP remaining to next tier calculated from current division and LP |
| `test_08_estimate_games.py` | Estimated games to next tier based on win rate and LP needed |

**Running all tests:**
```bash
python -m unittest discover -s UnitTests
```

### ML Model Evaluation

Training is evaluated with `GroupKFold(n_splits=5)` where groups are individual player PUUIDs. This prevents data leakage — all of a single player's games are always in the same fold.

#### Model Comparison Results

| Rank | Model | CV Accuracy | CV Std | F1 Score |
|---|---|---|---|---|
| 1 | Logistic Regression | 39.8% | ±1.65% | 0.398 |
| 2 | Neural Network (MLP) | 38.5% | ±2.48% | 0.385 |
| 3 | XGBoost | 37.9% | ±1.78% | 0.396 |
| 4 | Random Forest | 37.8% | ±2.63% | 0.369 |
| 5 | LightGBM | 37.3% | ±1.74% | 0.387 |
| 6 | SVM (RBF) | 37.2% | ±1.90% | 0.392 |
| 7 | Extra Trees | 36.7% | ±1.09% | 0.362 |
| 8 | Gradient Boosting | 35.6% | ±0.97% | 0.362 |
| 9 | K-Nearest Neighbors | 29.7% | ±3.24% | — |


**Default model at runtime:** XGBoost (best F1 score; `model_meta.json` references `xgboost`).

#### Evaluation Command

```bash
python Training/scripts/evaluate.py
```

Outputs: per-fold accuracy, overall CV accuracy, classification report with precision/recall/F1 per tier, and confusion matrix.

### Dataset

| Property | Value |
|---|---|
| Source | Riot Games API (ranked queue 420) |
| Size | ~38,000 games |
| Players | 2,065 unique players |
| Tiers | IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND |
| Collection date | Prior to March 2026 |
| Feature count (per player) | 30 (after aggregation) |

---

## 11. Versions & Known Issues

### Version History

| Version | Date | Notes |
|---|---|---|
| 1.0.0 | March 2026 | Initial release — single model (XGBoost), basic analytics |
| 1.1.0 | March 2026 | Multi-model support, model comparison, model switching via API |
| 1.2.0 | March 2026 | Champion recommender (cosine similarity), rank progression estimator |
| 1.3.0 | March 2026 | Live game analysis endpoint, team win probability |

### Known Issues & Limitations

| ID | Severity | Description |
|---|---|---|
| BUG-01 | Low | Development Riot API keys expire after 24 hours; the backend returns HTTP 403 until the key is renewed. |
| BUG-02 | Medium | Live game analysis is slow (~10–30 s) for players with little match history because the system must fetch and predict all 10 participants sequentially. |
| BUG-03 | Low | If a player has played fewer than 3 ranked games in the current split, the application cannot make a prediction. |
| BUG-04 | Low | Champion recommendation requires the `lol_rank_dataset.csv` to be present at runtime; the feature fails if the file is missing or the path is changed. |
| LIM-01 | Info | The model is trained only on ranks Iron–Diamond; Grandmaster, Master, and Challenger players are not predicted. |
| LIM-02 | Info | Model accuracy (~40%) is inherently limited because rank is influenced by factors not captured in per-game statistics (LP gains/losses, streak, duo queue, etc.). |
| LIM-03 | Info | The training dataset has unequal tier distribution; mid-tier performance (Silver–Gold) may be predicted more reliably than extremes. |

---

## 12. Legal & Licensing

### Project License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Matyáš Prokop

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Riot Games Legal Notice

> "LoL Rank Predictor is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc."

This application uses the **Riot Games API** in accordance with the [Riot Games Developer Policies](https://developer.riotgames.com/policies/general). By running this application, you agree to comply with those policies, including:

- The API key must not be shared publicly.
- Data collected via the API must not be used for commercial purposes without explicit approval from Riot Games.
- Rate limits imposed by Riot Games must be respected (the application handles HTTP 429 back-off automatically).
- Game data (match history, champion names, images) is the intellectual property of Riot Games, Inc.

### Champion Images

Champion images are served directly from the **Riot Data Dragon CDN** (`ddragon.leagueoflegends.com`). These assets are owned by Riot Games, Inc. and are used solely for non-commercial, informational display purposes.

### Third-Party Library Licenses

| Library | License |
|---|---|
| FastAPI | MIT |
| uvicorn | BSD |
| pandas | BSD |
| numpy | BSD |
| scikit-learn | BSD (New BSD) |
| XGBoost | Apache 2.0 |
| LightGBM | MIT |
| httpx | BSD |
| python-dotenv | BSD |
| scipy | BSD |
| Chart.js | MIT |

---


