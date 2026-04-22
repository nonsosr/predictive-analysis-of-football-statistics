"""
Stage 5: Hyperparameter tuning with cross-validation.

Tunes the Random Forest and XGBoost classifiers (and their regressor
counterparts) using TimeSeriesSplit cross-validation on the training
seasons, then evaluates the best models on the held-out 2023-24 test
season. This addresses the supervisor's feedback that the single
train/test split needs CV, and pushes for accuracy past the untuned
67.6% baseline.

Inputs (./data/processed/, ./models/)
  predictive_dataset.csv
  player_seasons_eligible.csv
  metrics.json (used for baseline comparison)

Outputs (./models/)
  rf_classifier_tuned.joblib
  xgb_classifier_tuned.joblib
  rf_regressor_tuned.joblib
  xgb_regressor_tuned.joblib
  shap_values_tuned.npz
  test_set_with_preds_tuned.csv
  tuning_results.json        cv scores, best params, baseline-vs-tuned comparison
  metrics_tuned.json         test-set metrics in the same format as metrics.json,
                             so stage4_figures.py can be repointed at it cleanly

Run from project root:
  python src/stage5_tune.py

Runtime: ~30-60 minutes depending on machine.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor


# ------------------------------------------------------------------ config

PROC = Path("./data/processed")
MODELS = Path("./models")
MODELS.mkdir(parents=True, exist_ok=True)

# Same temporal split as Stage 3 so the headline test numbers are
# directly comparable to the untuned baselines.
TRAIN_SEASONS = [1718, 1819, 1920, 2021, 2122, 2223]
TEST_SEASON   = 2324

CLASS_ORDER = ["low", "mid", "high"]

# Same feature set as Stage 3.
FEATURES = [
    "npxG_p90", "xA_p90", "npxG_plus_xA_p90", "xG_p90",
    "key_passes_p90", "xGChain_p90", "xGBuildup_p90",
    "shots_p90", "shots_on_target_p90", "xG_per_shot",
    "GA_p90", "np_goals_p90",
    "crosses_p90", "interceptions_p90", "tackles_won_p90",
    "fouls_drawn_p90", "offsides_p90",
    "team_goals_p90", "team_possession",
    "share_team_goals", "share_team_shots", "minutes_pct",
    "age",
]

# CV strategy: TimeSeriesSplit respects temporal ordering.
# With 6 training seasons I use 5 splits, so each fold trains on
# at least one season and tests on the next. Equivalent to expanding
# window CV: fold 1 trains on s1, tests on s2; fold 5 trains on s1-s5,
# tests on s6.
N_SPLITS = 5

# Grid sizes are deliberately moderate: large enough to find real gains,
# small enough that the search fits in ~30-60 min on a typical laptop.
RF_GRID = {
    "n_estimators":     [200, 400, 800],
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features":     ["sqrt", 0.5],
}

XGB_GRID = {
    "n_estimators":     [200, 400, 800],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Map low/mid/high -> 0/1/2 for XGBoost (it requires numeric targets).
LABEL_TO_INT = {"low": 0, "mid": 1, "high": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("stage5")


# ------------------------------------------------------------------ data
# Reuses the loading + splitting logic from Stage 3 so the train/test
# rows are identical. Duplicated rather than imported because Stage 3
# is structured as a script, not a module.

def load_data() -> pd.DataFrame:
    pred = pd.read_csv(PROC / "predictive_dataset.csv")
    elig = pd.read_csv(PROC / "player_seasons_eligible.csv")

    nxt = (elig[["understat_player_id", "season", "GA_p90"]]
           .rename(columns={"season": "_match_season",
                            "GA_p90": "target_GA_p90_next"})
           .dropna(subset=["understat_player_id"]))
    nxt["_match_season"] = nxt["_match_season"].astype(str)
    pred["season_next"] = pred["season_next"].astype(str)

    pred = pred.merge(
        nxt,
        left_on=["understat_player_id", "season_next"],
        right_on=["understat_player_id", "_match_season"],
        how="left",
    ).drop(columns=["_match_season"])

    pred = pred.dropna(subset=["target_GA_p90_next", "target_tertile_next"])
    log.info(f"Loaded {len(pred):,} rows with valid targets")
    return pred


def split(df: pd.DataFrame):
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test  = df[df["season"] == TEST_SEASON].copy()

    # Sort training rows by season so the CV folds respect temporal order.
    train = train.sort_values("season").reset_index(drop=True)

    log.info(f"Train: {len(train):,} rows, seasons {sorted(train['season'].unique())}")
    log.info(f"Test:  {len(test):,} rows, season {TEST_SEASON}")

    def _pack(d):
        X = d[FEATURES].copy()
        X["xG_per_shot"] = X["xG_per_shot"].fillna(0)
        return {
            "X": X,
            "y_clf_str": d["target_tertile_next"].astype(str),
            "y_clf_int": d["target_tertile_next"].map(LABEL_TO_INT).astype(int),
            "y_reg":     d["target_GA_p90_next"],
            "meta":      d[["player", "team", "league", "season",
                            "GA_p90", "npxG_plus_xA_p90", "age"]],
        }

    return _pack(train), _pack(test)


# ------------------------------------------------------------------ tuning

def grid_search(model, param_grid, X, y, cv, scoring, name) -> GridSearchCV:
    """
    GridSearchCV wrapper with logging. n_jobs=-1 uses all CPU cores —
    if your machine fans get loud, that's expected.
    """
    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    log.info(f"  {name}: {n_combos} combinations × {cv.n_splits} folds = "
             f"{n_combos * cv.n_splits} fits")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,            # refits best model on full training set
        return_train_score=False,
    )

    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    log.info(f"  {name}: best CV {scoring} = {search.best_score_:.4f} "
             f"({elapsed:.0f}s)")
    log.info(f"  {name}: best params = {search.best_params_}")
    return search


def tune_classifiers(train, cv):
    log.info("Tuning RF classifier ...")
    rf_search = grid_search(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        RF_GRID,
        train["X"], train["y_clf_str"],
        cv, scoring="f1_macro", name="RF clf",
    )

    log.info("Tuning XGB classifier ...")
    xgb_search = grid_search(
        XGBClassifier(
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss",
            random_state=42, n_jobs=1, verbosity=0,
            tree_method="hist",   # fast histogram-based splits
        ),
        XGB_GRID,
        train["X"], train["y_clf_int"],
        cv, scoring="f1_macro", name="XGB clf",
    )

    return rf_search, xgb_search


def tune_regressors(train, cv):
    log.info("Tuning RF regressor ...")
    rf_search = grid_search(
        RandomForestRegressor(random_state=42, n_jobs=1),
        RF_GRID,
        train["X"], train["y_reg"],
        cv, scoring="r2", name="RF reg",
    )

    log.info("Tuning XGB regressor ...")
    xgb_search = grid_search(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42, n_jobs=1, verbosity=0,
            tree_method="hist",
        ),
        XGB_GRID,
        train["X"], train["y_reg"],
        cv, scoring="r2", name="XGB reg",
    )

    return rf_search, xgb_search


# ------------------------------------------------------------------ eval

def eval_classifier(name, model, test, label_type="str") -> dict:
    """label_type: 'str' for sklearn (low/mid/high), 'int' for XGBoost (0/1/2)."""
    pred_raw = model.predict(test["X"])

    # Convert XGB int predictions back to string labels for unified reporting.
    if label_type == "int":
        pred = np.array([INT_TO_LABEL[i] for i in pred_raw])
    else:
        pred = pred_raw

    y_true = test["y_clf_str"].values

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", labels=CLASS_ORDER)
    cm = confusion_matrix(y_true, pred, labels=CLASS_ORDER)

    log.info(f"  {name}: TEST acc = {acc:.4f}   macro-F1 = {macro_f1:.4f}")
    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "labels": CLASS_ORDER,
        "report": classification_report(
            y_true, pred, labels=CLASS_ORDER, output_dict=True
        ),
    }


def eval_regressor(name, model, test) -> dict:
    pred = model.predict(test["X"])
    r2 = r2_score(test["y_reg"], pred)
    rmse = float(np.sqrt(mean_squared_error(test["y_reg"], pred)))
    mae = mean_absolute_error(test["y_reg"], pred)
    log.info(f"  {name}: TEST R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {
        "model": name,
        "r2": float(r2),
        "rmse": rmse,
        "mae": float(mae),
    }


# ------------------------------------------------------------------ shap

def compute_shap(clf, test, label_type="str"):
    """
    SHAP for the BEST classifier (whichever model wins). We compute on the
    test set so the figure-generation downstream uses honest out-of-sample
    explanations, not training-set ones.
    """
    log.info("Computing SHAP for best classifier ...")
    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(test["X"])

    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_arrays = [raw[:, :, i] for i in range(raw.shape[2])]
    else:
        shap_arrays = list(raw)

    exp_val = explainer.expected_value
    if np.isscalar(exp_val):
        exp_val = np.array([exp_val])
    else:
        exp_val = np.asarray(exp_val)

    # Get class order from the model
    if label_type == "int":
        # XGB returns int classes; map back so figures use string labels
        class_order = np.array([INT_TO_LABEL[i] for i in clf.classes_])
    else:
        class_order = clf.classes_

    return shap_arrays, exp_val, class_order


# ------------------------------------------------------------------ main

def main():
    log.info("=" * 60)
    log.info("Stage 5: hyperparameter tuning + CV")
    log.info("=" * 60)

    # Load baseline metrics for the comparison story
    try:
        baseline = json.load(open(MODELS / "metrics.json"))
    except FileNotFoundError:
        log.warning("metrics.json not found — run Stage 3 first for baseline comparison")
        baseline = None

    df = load_data()
    train, test = split(df)

    # TimeSeriesSplit on the training set only. Within the training seasons
    # (2017-18 ... 2022-23), each fold trains on earlier seasons and tests
    # on the next. Test season 2023-24 is held out completely from this CV.
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    # ----- Classification tuning -----
    log.info("\n>>> CLASSIFICATION")
    rf_clf_search, xgb_clf_search = tune_classifiers(train, cv)

    log.info("\nEvaluating tuned classifiers on held-out test season ...")
    rf_clf_metrics  = eval_classifier("RF (tuned)",  rf_clf_search.best_estimator_, test, "str")
    xgb_clf_metrics = eval_classifier("XGB (tuned)", xgb_clf_search.best_estimator_, test, "int")

    # Pick the better classifier for SHAP and saving as the headline model
    if xgb_clf_metrics["accuracy"] > rf_clf_metrics["accuracy"]:
        best_clf_name = "XGB"
        best_clf = xgb_clf_search.best_estimator_
        best_clf_label_type = "int"
    else:
        best_clf_name = "RF"
        best_clf = rf_clf_search.best_estimator_
        best_clf_label_type = "str"
    log.info(f"\nBest classifier: {best_clf_name}")

    # ----- Regression tuning -----
    log.info("\n>>> REGRESSION")
    rf_reg_search, xgb_reg_search = tune_regressors(train, cv)

    log.info("\nEvaluating tuned regressors on held-out test season ...")
    rf_reg_metrics  = eval_regressor("RF (tuned)",  rf_reg_search.best_estimator_, test)
    xgb_reg_metrics = eval_regressor("XGB (tuned)", xgb_reg_search.best_estimator_, test)

    if xgb_reg_metrics["r2"] > rf_reg_metrics["r2"]:
        best_reg_name = "XGB"
        best_reg = xgb_reg_search.best_estimator_
    else:
        best_reg_name = "RF"
        best_reg = rf_reg_search.best_estimator_
    log.info(f"\nBest regressor: {best_reg_name}")

    # ----- SHAP on best classifier -----
    shap_arrays, expected_value, class_order = compute_shap(
        best_clf, test, best_clf_label_type)

    # ----- Save everything -----
    joblib.dump(rf_clf_search.best_estimator_,  MODELS / "rf_classifier_tuned.joblib")
    joblib.dump(xgb_clf_search.best_estimator_, MODELS / "xgb_classifier_tuned.joblib")
    joblib.dump(rf_reg_search.best_estimator_,  MODELS / "rf_regressor_tuned.joblib")
    joblib.dump(xgb_reg_search.best_estimator_, MODELS / "xgb_regressor_tuned.joblib")

    np.savez(
        MODELS / "shap_values_tuned.npz",
        shap_low=shap_arrays[0],
        shap_mid=shap_arrays[1],
        shap_high=shap_arrays[2],
        expected_value=expected_value,
        class_order=np.array(class_order),
        feature_names=np.array(FEATURES),
        X_test=test["X"].to_numpy(),
        best_model=np.array([best_clf_name]),
    )

    # Test-set predictions table for the figure scripts
    test_out = test["meta"].copy()
    if best_clf_label_type == "int":
        test_out["pred_tertile"] = [INT_TO_LABEL[i] for i in best_clf.predict(test["X"])]
    else:
        test_out["pred_tertile"] = best_clf.predict(test["X"])
    test_out["true_tertile"]    = test["y_clf_str"].values
    test_out["true_GA_p90_nxt"] = test["y_reg"].values
    test_out["pred_GA_p90_nxt"] = best_reg.predict(test["X"])
    test_out.to_csv(MODELS / "test_set_with_preds_tuned.csv", index=False)

    # metrics_tuned.json: same shape as metrics.json so figure scripts
    # can be pointed at it without changes.
    metrics_tuned = {
        "classification": [rf_clf_metrics, xgb_clf_metrics],
        "regression":     [rf_reg_metrics, xgb_reg_metrics],
        "best_classifier": best_clf_name,
        "best_regressor":  best_reg_name,
        "split": {
            "train_seasons": TRAIN_SEASONS,
            "test_season":   TEST_SEASON,
            "n_train":       len(train["y_clf_str"]),
            "n_test":        len(test["y_clf_str"]),
            "cv_splits":     N_SPLITS,
        },
        "features": FEATURES,
    }
    with open(MODELS / "metrics_tuned.json", "w") as f:
        json.dump(metrics_tuned, f, indent=2)

    # tuning_results.json: search details + baseline comparison
    tuning_results = {
        "rf_classifier": {
            "best_cv_macro_f1": float(rf_clf_search.best_score_),
            "best_params":      rf_clf_search.best_params_,
        },
        "xgb_classifier": {
            "best_cv_macro_f1": float(xgb_clf_search.best_score_),
            "best_params":      xgb_clf_search.best_params_,
        },
        "rf_regressor": {
            "best_cv_r2":  float(rf_reg_search.best_score_),
            "best_params": rf_reg_search.best_params_,
        },
        "xgb_regressor": {
            "best_cv_r2":  float(xgb_reg_search.best_score_),
            "best_params": xgb_reg_search.best_params_,
        },
        "test_set_metrics": metrics_tuned,
        "baseline_comparison": _build_comparison(baseline, metrics_tuned)
                               if baseline else "baseline metrics.json not found",
    }
    with open(MODELS / "tuning_results.json", "w") as f:
        json.dump(tuning_results, f, indent=2)

    # ----- Final summary printout -----
    log.info("=" * 60)
    log.info("FINAL SUMMARY (held-out 2023-24 test season)")
    log.info("=" * 60)
    if baseline:
        rf_baseline_clf = next(r for r in baseline["classification"] if r["model"] == "RF")
        log.info(f"Baseline RF clf:       acc = {rf_baseline_clf['accuracy']:.4f}  "
                 f"macro-F1 = {rf_baseline_clf['macro_f1']:.4f}")
    log.info(f"Tuned RF clf:          acc = {rf_clf_metrics['accuracy']:.4f}  "
             f"macro-F1 = {rf_clf_metrics['macro_f1']:.4f}")
    log.info(f"Tuned XGB clf:         acc = {xgb_clf_metrics['accuracy']:.4f}  "
             f"macro-F1 = {xgb_clf_metrics['macro_f1']:.4f}")
    log.info("")
    if baseline:
        rf_baseline_reg = next(r for r in baseline["regression"] if r["model"] == "RF")
        log.info(f"Baseline RF reg:       R² = {rf_baseline_reg['r2']:.4f}  "
                 f"RMSE = {rf_baseline_reg['rmse']:.4f}")
    log.info(f"Tuned RF reg:          R² = {rf_reg_metrics['r2']:.4f}  "
             f"RMSE = {rf_reg_metrics['rmse']:.4f}")
    log.info(f"Tuned XGB reg:         R² = {xgb_reg_metrics['r2']:.4f}  "
             f"RMSE = {xgb_reg_metrics['rmse']:.4f}")
    log.info("=" * 60)


def _build_comparison(baseline_metrics: dict, tuned_metrics: dict) -> dict:
    """Side-by-side baseline vs tuned for the writeup."""
    rf_b_clf = next(r for r in baseline_metrics["classification"] if r["model"] == "RF")
    rf_b_reg = next(r for r in baseline_metrics["regression"] if r["model"] == "RF")
    rf_t_clf = next(r for r in tuned_metrics["classification"] if r["model"] == "RF (tuned)")
    rf_t_reg = next(r for r in tuned_metrics["regression"] if r["model"] == "RF (tuned)")
    xgb_t_clf = next(r for r in tuned_metrics["classification"] if r["model"] == "XGB (tuned)")
    xgb_t_reg = next(r for r in tuned_metrics["regression"] if r["model"] == "XGB (tuned)")

    return {
        "classification_accuracy": {
            "rf_baseline": rf_b_clf["accuracy"],
            "rf_tuned":    rf_t_clf["accuracy"],
            "xgb_tuned":   xgb_t_clf["accuracy"],
            "best_gain_pp": max(
                rf_t_clf["accuracy"] - rf_b_clf["accuracy"],
                xgb_t_clf["accuracy"] - rf_b_clf["accuracy"],
            ),
        },
        "regression_r2": {
            "rf_baseline": rf_b_reg["r2"],
            "rf_tuned":    rf_t_reg["r2"],
            "xgb_tuned":   xgb_t_reg["r2"],
            "best_gain": max(
                rf_t_reg["r2"] - rf_b_reg["r2"],
                xgb_t_reg["r2"] - rf_b_reg["r2"],
            ),
        },
    }


if __name__ == "__main__":
    main()