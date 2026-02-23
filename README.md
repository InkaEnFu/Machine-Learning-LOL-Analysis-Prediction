# LoL Rank Predictor

Webová aplikace, která na základě posledních ranked her hráče v League of Legends odhadne jeho rank pomocí ML modelu. Výsledek je doplněn o analytický dashboard — číselné srovnání výkonu hráče s průměrem predikovaného ranku, z-score, percentily a grafy.

---

## Co aplikace dělá

1. Uživatel zadá své Riot ID ve formátu `Jméno#TAG` a vybere region.
2. Backend přes Riot API stáhne posledních 10 ranked her hráče.
3. Z her se spočítají agregované metriky (průměry, mediány, rozptyly KDA, CS/min, damage/min atd.).
4. Natrénovaný ML model (LogisticRegression) predikuje rank (Iron–Diamond).
5. Frontend zobrazí:
   - predikovaný rank s confidence score
   - pravděpodobnosti pro každý rank (bar chart)
   - radar chart výkonu
   - porovnání s průměrem predikovaného ranku (z-score, percentil, odchylky)
   - přehled metrik rozdělený do kategorií: Combat, Farming, Vision, Survivability, Economy
   - historii posledních 10 her
   - trend KDA a CS/min přes zápasy

---

## Struktura projektu

```
MLProjekt/
├── backend/                    # FastAPI backend
│   ├── config.py               # Konfigurace (API klíč, regiony, cesty)
│   ├── main.py                 # FastAPI app, statické soubory, root endpoint
│   ├── models/
│   │   └── schemas.py          # Pydantic validace vstupu
│   ├── routers/
│   │   └── predict.py          # POST /api/predict endpoint
│   └── services/
│       ├── riot_api.py         # Riot API klient (account, match history, parsing)
│       ├── predictor.py        # Načtení modelu a inference
│       └── analytics.py        # Výpočet srovnání s rank průměrem
│
├── Training/                   # Vše kolem dat a ML
│   ├── checkpoint/
│   │   └── checkpoint.json     # Stav sběru dat
│   ├── collection/
│   │   └── lol_data_collection.ipynb  # Notebook pro sběr dat
│   ├── datasets/
│   │   └── lol_rank_dataset.csv       # Trénovací dataset (~38 000 her, 2065 hráčů)
│   ├── models/                 # Generované artefakty po trénování
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   ├── rank_stats.pkl
│   │   └── model_meta.json
│   ├── preprocessing/
│   │   ├── cleaner.py          # Načtení a čištění datasetu
│   │   └── feature_engineering.py  # Agregace per-hráč, odvozené metriky
│   └── scripts/
│       ├── train.py            # Trénování modelu
│       └── evaluate.py         # Evaluace přes GroupKFold
│
├── frontend/                   # Statický frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── requirements.txt
└── run.py                      # Spuštění serveru
```

---

## Technologie

| Vrstva | Technologie |
|---|---|
| Backend | Python 3.13, FastAPI, uvicorn |
| ML | scikit-learn (LogisticRegression, StandardScaler, GroupKFold) |
| Data | pandas, numpy, scipy |
| API komunikace | httpx |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |

---

## Instalace a spuštění

### 1. Požadavky

- Python 3.10+
- pip

### 2. Vytvoření virtuálního prostředí (doporučeno)

```bash
python -m venv .venv
```

Aktivace na Windows:
```bash
.venv\Scripts\activate
```

Aktivace na macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 4. Trénování modelu

Model musí být natrénovaný před spuštěním serveru. Artefakty se uloží do `Training/models/`.

```bash
python Training/scripts/train.py
```

Výstup ukáže CV accuracy a classification report. Artefakty (`model.pkl`, `scaler.pkl`, `rank_stats.pkl`, `model_meta.json`) se objeví v `Training/models/`.

### 5. Spuštění serveru

```bash
python run.py
```

Server běží na [http://localhost:8000](http://localhost:8000).

---

## Dataset

Dataset obsahuje přibližně **38 000 ranked her** od **2 065 hráčů** na serveru EUW, rovnoměrně rozdělených přes ranky Iron až Diamond (cca 295 hráčů na rank).

Každý řádek odpovídá jedné hře a obsahuje metriky jako `kda`, `cs_per_min`, `damage_per_min`, `gold_per_min`, `vision_per_min`, `deaths_per_min`, `damage_taken_per_min` a základní statistiky kills/deaths/assists.

Pro trénování modelu jsou hry agregovány na úroveň hráče — z 20 her se spočítají průměry, mediány a směrodatné odchylky klíčových metrik. Model tak predikuje rank z profilu hráče, ne z jediné hry.

GroupKFold split zajišťuje, že hry stejného hráče nikdy nejsou zároveň v trénovací i testovací množině (žádný data leakage).

---

## API

### `POST /api/predict`

**Request body:**
```json
{
  "game_name": "Jméno",
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
    "rank_probabilities": { "IRON": 0.02, "BRONZE": 0.08, ... },
    "feature_importance": { "kda_mean": 0.18, ... }
  },
  "player_stats": { "kda_mean": 3.2, "cs_per_min_mean": 6.1, ... },
  "comparison": {
    "combat": [ { "metric": "kda_mean", "label": "KDA", "player_value": 3.2, "rank_average": 2.8, "diff_percent": 14.3, "z_score": 0.9, "percentile": 81.6, "above_average": true } ],
    ...
  },
  "matches": [ { "champion": "Jinx", "role": "BOTTOM", "kills": 8, "deaths": 2, ... } ],
  "record": { "wins": 7, "losses": 3, "total": 10 }
}
```

Dostupné regiony: `euw`, `eune`, `na`, `kr`, `jp`, `br`, `lan`, `las`, `oce`, `tr`, `ru`

---

## Poznámky

- Riot API klíč v `backend/config.py` je vývojový klíč s limitem 20 requestů za sekundu. Pro produkci je potřeba produkční klíč přes [developer.riotgames.com](https://developer.riotgames.com).
- Aplikace filtruje výhradně ranked hry (queue ID 420). Normály, ARAM a ostatní režimy jsou ignorovány.
- Pokud má hráč méně než 3 ranked hry v historii, API vrátí chybu zobrazenou v UI.
