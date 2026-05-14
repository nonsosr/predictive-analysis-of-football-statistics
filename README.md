# Predicting Goal Contributions in Europe's Top Leagues Using Explainable Machine Learning

**Author:** Chukwunonso Okpala (jn806962)
**Supervisor:** Mr Asad Hussain
**Degree programme:** BSc Computer Science, University of Reading
**Academic year:** 2025-2026

---

## Summary

This project forecasts next-season goal contributions (goals plus assists per 90 minutes) for attacking players across the five largest European football leagues, explains each prediction at the individual-player level using SHAP, and recovers data-driven player archetypes through clustering. The pipeline scrapes eight seasons (2017-18 to 2024-25) of public match and player data from FBref and Understat, merges and filters it into 12,848 eligible player-seasons (5,290 with next-season targets), engineers 23 per-90 and team-context features, and trains both a tertile classifier (low / mid / high next-season contribution) and a regressor for the actual numeric value. Random Forest and XGBoost ensembles are tuned via TimeSeriesSplit cross-validation, with logistic and linear regression as interpretable baselines. Clustering uses K-Means and Gaussian mixtures on a 14-feature style subset.

The headline methodological finding is that three substantially different model families (linear baselines, tuned Random Forest, tuned XGBoost) converge on classification accuracy within a window of approximately two percentage points, indicating a feature-set ceiling rather than a model-selection limit. The contribution of this work is therefore positioned as **interpretable forecasting and archetype-aware analysis** rather than raw predictive performance.

---

## Key results

Held-out test season 2023-24 (n = 738), trained on 2017-18 through 2022-23 (n = 4,574), tuned via 5-fold TimeSeriesSplit CV.

### Classification (tertile of next-season G+A/90)

| Model | Accuracy | Macro-F1 | F1 low / mid / high |
|---|---|---|---|
| **RF (tuned)** | **0.686** | **0.683** | 0.70 / 0.54 / 0.81 |
| XGB (tuned) | 0.667 | 0.663 | 0.67 / 0.51 / 0.81 |

### Regression (next-season G+A/90)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| RF (tuned) | 0.670 | 0.142 | 0.093 |
| **XGB (tuned)** | **0.673** | **0.142** | **0.092** |

Best classifier: tuned Random Forest. Best regressor: tuned XGBoost. Per-class performance is consistently strongest on the `high` tier (F1 ≈ 0.81), reflecting that elite contributors are statistically separable, while the `mid` tier sits closest to the decision boundaries.

### Clustering

Five interpretable archetypes recovered in each of two runs, validated against named exemplar players:

- **All-positions run** (n ≈ 8,634, k = 5): Defensive Anchors, Box-to-Box, Secondary Forwards, Playmakers, Elite Finishers.
- **Attacker-only run** (n ≈ 2,278, k = 5): Penalty-Box Strikers, Support Forwards, Wide Creators, Elite Wide Attackers, Elite Finishers.

Silhouette peaks at k = 2 algorithmically; k = 5 is retained on interpretive grounds, following the precedent set by Decroos and Davis (2020).

---

## Folder structure

```
.
├── README.md                       This file
├── LICENSE
├── requirements.txt                Python dependencies
├── .gitignore
│
├── src/                            Pipeline scripts, run in order
│   ├── stage1_scrape_fbref.py
│   ├── stage1b_scrape_understat.py
│   ├── stage2_merge.py
│   ├── stage3_model.py             Baseline RF/logistic/linear + SHAP
│   ├── stage4_figures.py           Poster and dissertation figures (300 DPI)
│   ├── stage5_tuning.py            Grid search under TimeSeriesSplit CV
│   └── stage6_clustering.py        K-Means + GMM + UMAP/PCA + SHAP-per-cluster
│
├── data/
│   ├── raw/                        Scraped CSVs (git-ignored due to size)
│   └── processed/                  Merged, filtered, feature-engineered tables
│
├── models/                         Trained models, tuning_results.json, shap_values_tuned.npz
├── figures/                        Poster and dissertation figures
└── report/                         Dissertation writeup
```

---

## How to reproduce

### Requirements

- Python 3.11 or later
- Roughly 2 GB of free disk space (mostly for raw scraped data)
- An internet connection for the scraping stages

### Setup

```bash
git clone https://csgitlab.reading.ac.uk/jn806962/predictive-analysis-of-football-statistics
cd predictive-analysis-of-football-statistics

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Running the pipeline

All scripts are run from the **project root**, in order:

```bash
# Stage 1a: Scrape FBref (~15-25 min)
python src/stage1_scrape_fbref.py

# Stage 1b: Scrape Understat (~5-10 min)
python src/stage1b_scrape_understat.py

# Stage 2: Merge, filter, feature-engineer (< 1 min)
python src/stage2_merge.py

# Stage 3: Train baseline classifier + regressor, compute SHAP (~2 min)
python src/stage3_model.py

# Stage 4: Generate figures at 300 DPI (< 1 min)
python src/stage4_figures.py

# Stage 5: Hyperparameter tuning under TimeSeriesSplit CV (~10-15 min)
python src/stage5_tuning.py

# Stage 6: Clustering analysis + SHAP-per-cluster (~3-5 min)
python src/stage6_clustering.py
```

Each stage is idempotent. Stage 1 skips files that already exist on disk, so an interrupted scrape can be resumed without loss.

Stage 6 contains an `ATTACKING_ONLY` toggle and a `SUFFIX` system so the all-positions and attacker-only runs coexist without overwriting each other's outputs.

### Skipping the scrape

The scraping stages take over 25 minutes combined. If you only want to verify the modelling pipeline, the `data/processed/` tables produced by Stage 2 are included in the repository, so you can jump straight to Stage 3.

---

## Data sources and ethics

All data was obtained from publicly accessible statistics pages via the open-source [`soccerdata`](https://github.com/probberechts/soccerdata) Python library (v1.9.0), which wraps:

- **FBref.com** (Sports Reference LLC) — match and player counting statistics, team-level context.
- **Understat.com** — expected-goals metrics (xG, xA, npxG, key passes, xG chain/buildup).

No proprietary feeds (Opta, StatsBomb) or personally identifiable information beyond what is already publicly disclosed by these providers were used. No injury, medical, or contract information is collected. Scraping is rate-limited via `soccerdata`'s built-in delays to respect the source sites.

---

## Features

**Predictive feature set (23 features, fed to classifier and regressor):**

- *Expected stats (per 90, Understat):* `npxG_p90`, `xA_p90`, `npxG_plus_xA_p90`, `xG_p90`, `key_passes_p90`, `xGChain_p90`, `xGBuildup_p90`
- *Shot volume and quality:* `shots_p90`, `shots_on_target_p90`, `xG_per_shot`
- *Current-season output (autoregressive):* `GA_p90`, `np_goals_p90`
- *Defensive and involvement:* `crosses_p90`, `interceptions_p90`, `tackles_won_p90`, `fouls_drawn_p90`, `offsides_p90`
- *Team context:* `team_goals_p90`, `team_possession`, `share_team_goals`, `share_team_shots`, `minutes_pct`
- *Demographic:* `age`

**Clustering feature set (14-feature style subset):** the predictive set minus the five team-context features and `age`, plus the two demographic-adjacent items. Team-context and age are excluded so clusters reflect *how* a player plays rather than *where* they play, following the methodological argument of Decroos and Davis (2020).

---

## Targets

- **Classification:** tertile label (`low` / `mid` / `high`) of next-season G+A/90, computed *within each league-season* so scoring inflation across eras does not bias the labels.
- **Regression:** raw next-season G+A/90.

Cross-season linking uses Understat's stable `player_id` rather than name matching to avoid transliteration issues across sources.

---

## Known limitations

1. **FBref ↔ Understat match rate is 72.8%.** Losses are concentrated in team-name spelling differences (e.g., "Manchester United" vs "Manchester Utd"). A manual alias map would recover most of these. Documented as future work.
2. **Player-level possession and passing features** (touches in attacking penalty area, progressive carries, FBref key passes) are not exposed by `soccerdata`'s current API. These were in the original methodology plan and would strengthen the feature set. Reachable via the R package `worldfootballR` or direct FBref scraping.
3. **Team-level expected-goals totals are not exposed** by `soccerdata.read_team_season_stats(stat_type="standard")`. The project uses `team_goals_p90` as a proxy for team attacking strength; team xG would be a less noisy alternative.
4. **Mid-season transfers where neither stint reaches 900 minutes are excluded.** Aggregating these would preserve more rows but complicate the team-context features (which team's context applies?).
5. **Three model families converge to within ~2 percentage points** on classification accuracy (tuned RF 68.6%, tuned XGB 66.7%, linear baselines comparable). This is interpreted as a ceiling on the signal available in the current feature set rather than a model inadequacy. The dissertation's contribution is correspondingly framed as interpretability and archetype analysis, not aggregate accuracy.
6. **Silhouette score is algorithmically optimal at k = 2** but k = 5 is retained on interpretive grounds. This is a deliberate methodological choice defended in the dissertation, following Decroos and Davis (2020).
7. **The tertile target uses `pd.qcut` with `duplicates='drop'`**, which produces NaN labels for a small number of tie-boundary players. Acceptable loss at the 900-minute threshold.

---

## Dependencies

See `requirements.txt`. Main libraries:

- `soccerdata` (1.9.0) — scraping FBref and Understat
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — Random Forest, logistic regression, linear regression, K-Means, GMM, TimeSeriesSplit
- `xgboost` — gradient-boosted trees (tuned regressor leads)
- `shap` — model explainability
- `umap-learn` — clustering visualisation (PCA and t-SNE as fallbacks)
- `matplotlib` — figures at 300 DPI
- `unidecode` — name normalisation during merges
- `joblib` — saving trained models

---

## Repository

- GitLab (primary): https://csgitlab.reading.ac.uk/jn806962/predictive-analysis-of-football-statistics
- GitHub (mirror): https://github.com/nonsosr/predictive-analysis-of-football-statistics

---

## Acknowledgements

Thanks to Mr Asad Hussain for project supervision, and to the maintainers of the open-source `soccerdata`, `scikit-learn`, `xgboost`, `shap`, and `umap-learn` libraries that made this work possible.

---

## Licence

See `LICENSE`. Data obtained from FBref and Understat remains subject to those providers' terms of use.
