# Predicting Goal Contributions in Europe's Top Leagues Using Explainable Machine Learning

**Author:** Chukwunonso Okpala
**Supervisor:** Mr Asad Hussain
**Degree programme:** BSc Computer Science, University of Reading
**Academic year:** 2025-2026

---

## Summary

This project forecasts next-season goal contributions (goals plus assists per 90 minutes) for attacking players across the five largest European football leagues, and uses SHAP to explain each prediction at the individual-player level. The pipeline scrapes eight seasons of public match and player data from FBref and Understat, merges and filters it, engineers 23 per-90 and team-context features, and trains both a tertile classifier (low / mid / high next-season contribution) and a regressor for the actual numeric value. Random Forest is used as the main model, with logistic and linear regression as interpretable baselines. Evaluation uses a temporal train/test split so the model is tested on a season strictly later than any season it has seen in training.

The contribution of this project is not raw predictive accuracy — the linear baselines match the Random Forest on aggregate metrics — but **interpretable per-prediction attribution**, which only the tree-based model supports via SHAP and which opens the door to archetype-aware analysis in future work.

---

## Key results

On the held-out 2023-24 test season (predicting 2024-25 outcomes):

| Task | Model | Metric | Value |
|---|---|---|---|
| Classification | Random Forest | Accuracy | 67.6% |
| Classification | Random Forest | Macro-F1 | 0.67 |
| Classification | Random Forest | Per-class F1 (low / mid / high) | 0.69 / 0.52 / 0.81 |
| Regression | Random Forest | R² | 0.666 |
| Regression | Random Forest | RMSE | 0.143 |
| Regression | Random Forest | MAE | 0.093 |

Training set: 4,574 player-seasons from 2017-18 through 2022-23.
Test set: 738 player-seasons from 2023-24 with next-season targets in 2024-25.

---

## Folder structure

```
.
├── README.md              This file
├── LICENSE
├── requirements.txt       Python dependencies
├── .gitignore
│
├── src/                   Pipeline scripts, run in order
│   ├── stage1_scrape_fbref.py
│   ├── stage1b_scrape_understat.py
│   ├── stage2_merge.py
│   ├── stage3_model.py
│   └── stage4_figures.py
│
├── data/
│   ├── raw/               Scraped CSVs (git-ignored due to size)
│   └── processed/         Merged, filtered, feature-engineered tables
│
├── models/                Trained models, metrics, SHAP values
├── figures/               Poster and dissertation figures
└── report/                Dissertation writeup (work in progress)
```

---

## How to reproduce

### Requirements

- Python 3.11 or later
- Roughly 2 GB of free disk space (mostly for raw scraped data)
- An internet connection for the scraping stages

### Setup

```bash
# Clone this repository
git clone <repo-url>
cd predictive-analysis-of-football-statistics

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the pipeline

All scripts are run from the **project root**, not from inside `src/`. Run them in order:

```bash
# Stage 1a: Scrape FBref data (~15-25 min; writes data/raw/players_*.csv + teams_*.csv)
python src/stage1_scrape_fbref.py

# Stage 1b: Scrape Understat data (~5-10 min; writes data/raw/understat_players.csv)
python src/stage1b_scrape_understat.py

# Stage 2: Merge + filter + feature-engineer (< 1 min; writes data/processed/*.csv)
python src/stage2_merge.py

# Stage 3: Train classifier + regressor, compute SHAP (~2 min; writes models/*)
python src/stage3_model.py

# Stage 4: Generate the four poster figures (< 1 min; writes figures/*.png)
python src/stage4_figures.py
```

Each stage is idempotent — re-running on cached data is fast. The scraper stages skip files that already exist, so if a scrape is interrupted you can restart and it picks up where it left off.

### Skipping the scrape

The scraping stages take over 25 minutes combined and hit third-party servers. If you only want to verify the modelling pipeline, the `data/processed/` tables produced by Stage 2 are included in the repository, so you can jump straight to Stage 3.

---

## Data sources and ethics

All data was obtained from publicly accessible statistics pages via the open-source [`soccerdata`](https://github.com/probberechts/soccerdata) Python library, which wraps:

- **FBref.com** (Sports Reference LLC) — match and player counting statistics, team-level context.
- **Understat.com** — expected-goals metrics (xG, xA, npxG, key passes, xG chain/buildup).

No proprietary feeds (Opta, StatsBomb) or personally identifiable information beyond what is already publicly disclosed by these providers were used. No injury, medical, or contract information is collected. Scraping is rate-limited via `soccerdata`'s built-in delays to respect the source sites.

---

## Features

23 per-player-season features fed to the models:

**Expected stats (per 90, from Understat):** `npxG_p90`, `xA_p90`, `npxG_plus_xA_p90`, `xG_p90`, `key_passes_p90`, `xGChain_p90`, `xGBuildup_p90`.

**Shot volume and quality:** `shots_p90`, `shots_on_target_p90`, `xG_per_shot`.

**Current-season output (autoregressive):** `GA_p90`, `np_goals_p90`.

**Defensive and involvement (from FBref):** `crosses_p90`, `interceptions_p90`, `tackles_won_p90`, `fouls_drawn_p90`, `offsides_p90`.

**Team context:** `team_goals_p90` (attacking strength proxy), `team_possession`, `share_team_goals`, `share_team_shots`, `minutes_pct`.

**Demographic:** `age`.

---

## Targets

- **Classification:** tertile label (`low` / `mid` / `high`) of next-season G+A/90, computed *within each league-season* so scoring inflation across eras does not bias the labels.
- **Regression:** raw next-season G+A/90.

Cross-season linking uses Understat's stable `player_id` rather than name matching to avoid transliteration issues across sources.

---

## Known limitations

The limitations are documented here because they are methodologically important and some affect interpretation of the results.

1. **FBref ⇄ Understat match rate is 72.8%.** Losses are concentrated in team-name spelling differences (e.g., "Manchester United" vs "Manchester Utd"). A manual alias map would recover most of these. Addressed as future work.
2. **Player-level possession and passing features (touches in attacking penalty area, progressive carries, key passes from FBref) are unavailable through `soccerdata`'s current API.** These were in the original methodology plan and would strengthen the feature set. Reachable via the R package `worldfootballR` or direct FBref scraping; flagged as next-stage work.
3. **Team-level expected-goals totals are not exposed** by `soccerdata.read_team_season_stats(stat_type="standard")`. The project uses `team_goals_p90` as a proxy for team attacking strength; team xG would be a less noisy alternative.
4. **Mid-season transfers where neither stint reaches 900 minutes are excluded.** Aggregating these would preserve more rows but complicate the team-context features (which team's context applies?). Documented limitation.
5. **Aggregate metrics are equivalent between the Random Forest and the linear baselines** (≈ 67-68% accuracy, R² ≈ 0.67). This likely reflects a ceiling on the signal available in the current feature set rather than a model inadequacy. The dissertation contribution is therefore framed as interpretability rather than accuracy gain.
6. **The tertile target uses `pd.qcut` with `duplicates='drop'`**, which produces NaN labels for a small number of tie-boundary players. Acceptable loss at the 900-minute threshold.

---

## Dependencies

See `requirements.txt`. The main ones:

- `soccerdata` (1.9.0) — scraping FBref and Understat
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — Random Forest, logistic regression, linear regression
- `shap` — model explainability
- `matplotlib` — figures
- `unidecode` — name normalization during merges
- `joblib` — saving trained models

---

## Acknowledgements

Thanks to Mr Asad Hussain for project supervision and to the maintainers of the open-source `soccerdata`, `scikit-learn`, and `shap` libraries that made this work possible.

---

## Licence

See `LICENSE` file. Data obtained from FBref and Understat remains subject to those providers' terms of use.