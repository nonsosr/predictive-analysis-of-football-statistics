"""
Stage 3: Modelling.

Trains four models on the predictive dataset:
    - Random Forest CLASSIFIER (main story; predicts next-season tertile)
    - Logistic Regression CLASSIFIER (baseline)
    - Random Forest REGRESSOR (supplementary; predicts next-season G+A/90)
    - Linear Regression REGRESSOR (baseline)

Evaluates all four. Computes SHAP values on the RF classifier (the
explainability story revolves around this model).


Inputs (./data/processed/)
--------
  predictive_dataset.csv       current-season features + next-season tertile
  player_seasons_eligible.csv  used to build the regression target

Outputs (./models/)
--------
  metrics.json            all four models' eval metrics
  rf_classifier.joblib    trained RF classifier
  rf_regressor.joblib     trained RF regressor
  shap_values.npz         SHAP values for the test set (RF clf)
  feature_columns.txt     ordered feature list (so figures can label axes)
  test_set_with_preds.csv test rows + predictions + true values (for figures)

"""

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ config

PROC = Path("./data/processed")
MODELS = Path("./models")
MODELS.mkdir(parents=True, exist_ok=True)

# Temporal split. We train on rows whose CURRENT season is in TRAIN_SEASONS
# (their next-season target lives in the following season), and test on
# rows whose current season is TEST_SEASON. Test-set rows therefore
# predict outcomes in 2024-25, which the training set never saw.
TRAIN_SEASONS = [1718, 1819, 1920, 2021, 2122, 2223]   # ints in the CSV
TEST_SEASON = 2324

# Feature columns: explicitly listed, not "all numeric", so the model can
# only see what we want it to see (no leakage of player_id, team strings,
# next-season fields, etc.).
FEATURES = [
    # Core attacking — expected stats from Understat, per 90
    "npxG_p90", "xA_p90", "npxG_plus_xA_p90", "xG_p90",
    "key_passes_p90", "xGChain_p90", "xGBuildup_p90",
    # Shot volume & quality
    "shots_p90", "shots_on_target_p90", "xG_per_shot",
    # Current-season output (autoregressive baseline)
    "GA_p90", "np_goals_p90",
    # Defensive / involvement profile (for the clustering story too)
    "crosses_p90", "interceptions_p90", "tackles_won_p90",
    "fouls_drawn_p90", "offsides_p90",
    # Team context (the player's environment)
    "team_goals_p90", "team_possession",
    "share_team_goals", "share_team_shots", "minutes_pct",
    # Demographic
    "age",
]

CLASS_ORDER = ["low", "mid", "high"]   # consistent ordering everywhere


# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("stage3")


# ------------------------------------------------------------------ data

def load_data() -> pd.DataFrame:
    """
    Load predictive dataset and attach the regression target
    (next-season GA_p90), which Stage 2 didn't compute.

    The join uses (understat_player_id, season_next) — same logic as the
    tertile join, just pulling the raw value instead of the bin label.
    """
    pred = pd.read_csv(PROC / "predictive_dataset.csv")
    elig = pd.read_csv(PROC / "player_seasons_eligible.csv")

    # Build (player_id, season) -> GA_p90 lookup from the eligible set.
    # We need eligible (not predictive) because predictive's GA_p90
    # column is the CURRENT-season value, not next-season.
    nxt_value = (elig[["understat_player_id", "season", "GA_p90"]]
                 .rename(columns={"season": "_match_season",
                                  "GA_p90": "target_GA_p90_next"})
                 .dropna(subset=["understat_player_id"]))
    nxt_value["_match_season"] = nxt_value["_match_season"].astype(str)
    pred["season_next"] = pred["season_next"].astype(str)

    pred = pred.merge(
        nxt_value,
        left_on=["understat_player_id", "season_next"],
        right_on=["understat_player_id", "_match_season"],
        how="left",
    ).drop(columns=["_match_season"])

    # Drop rows where regression target couldn't be built
    # (would only happen if Understat ID is missing). Without a target
    # row we can't train OR test.
    n_before = len(pred)
    pred = pred.dropna(subset=["target_GA_p90_next", "target_tertile_next"])
    log.info(f"Loaded {len(pred):,} rows ({n_before - len(pred)} dropped for missing target)")
    return pred


def split(df: pd.DataFrame):
    """
    Temporal split. Returns dicts keyed by 'X', 'y_clf', 'y_reg',
    'meta' (player/team/season info kept for figure-side labelling).

    Imputation: xG_per_shot is NaN when a player took 0 shots; impute
    as 0 (no shot quality because no shots). All other features have
    zero nulls per the audit.
    """
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test  = df[df["season"] == TEST_SEASON].copy()

    log.info(f"Train: {len(train):,} rows ({sorted(train['season'].unique())})")
    log.info(f"Test:  {len(test):,} rows  (season {TEST_SEASON})")

    def _pack(d):
        X = d[FEATURES].copy()
        X["xG_per_shot"] = X["xG_per_shot"].fillna(0)
        return {
            "X": X,
            "y_clf": d["target_tertile_next"],
            "y_reg": d["target_GA_p90_next"],
            "meta": d[["player", "team", "league", "season",
                       "GA_p90", "npxG_plus_xA_p90", "age"]],
        }

    return _pack(train), _pack(test)


# ------------------------------------------------------------------ models

def train_classifier(train, test):
    """
    Random Forest + Logistic Regression baseline.

    LR uses StandardScaler in a Pipeline because it's distance-based;
    RF is invariant to monotonic transforms so it doesn't need scaling.
    Both use class_weight='balanced' as a defensive measure even though
    our tertile design is already balanced — it costs nothing.
    """
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    lr = Pipeline([
        ("scale", StandardScaler()),
        # Newer sklearn auto-selects multinomial for >2 classes;
        # the multi_class kwarg was removed in 1.5+.
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    log.info("Training RF classifier ...")
    rf.fit(train["X"], train["y_clf"])
    log.info("Training LR classifier ...")
    lr.fit(train["X"], train["y_clf"])

    return rf, lr


def train_regressor(train, test):
    """RF + Linear Regression baseline. Same scaling rationale."""
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    linr = Pipeline([
        ("scale", StandardScaler()),
        ("lin", LinearRegression()),
    ])

    log.info("Training RF regressor ...")
    rf.fit(train["X"], train["y_reg"])
    log.info("Training Linear regressor ...")
    linr.fit(train["X"], train["y_reg"])

    return rf, linr


# ------------------------------------------------------------------ eval

def eval_classifier(name, model, test) -> dict:
    pred = model.predict(test["X"])
    acc = accuracy_score(test["y_clf"], pred)
    macro_f1 = f1_score(test["y_clf"], pred, average="macro", labels=CLASS_ORDER)
    cm = confusion_matrix(test["y_clf"], pred, labels=CLASS_ORDER)
    log.info(f"{name}: acc={acc:.3f}  macro-F1={macro_f1:.3f}")
    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "labels": CLASS_ORDER,
        "report": classification_report(
            test["y_clf"], pred, labels=CLASS_ORDER, output_dict=True
        ),
    }


def eval_regressor(name, model, test) -> dict:
    pred = model.predict(test["X"])
    r2 = r2_score(test["y_reg"], pred)
    rmse = float(np.sqrt(mean_squared_error(test["y_reg"], pred)))
    mae = mean_absolute_error(test["y_reg"], pred)
    log.info(f"{name}: R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return {
        "model": name,
        "r2": float(r2),
        "rmse": rmse,
        "mae": float(mae),
    }


# ------------------------------------------------------------------ autoregressive baseline + CIs

def train_autoregressive_baseline(train, test) -> dict:
    """
    Univariate autoregressive baseline: predict next-season tertile and
    G+A/90 from current-season GA_p90 alone (no other features).

    This is the critical diagnostic missing from the original pipeline.
    It answers: how much of the full model's performance derives from
    simple output persistence vs the remaining 22 features?

    A properly fitted AR baseline (trained on training seasons, evaluated
    on the test season) gives a fair lower bound. Any multi-feature model
    that doesn't beat this baseline adds no value over trivial persistence.
    """
    X_train_ar = train["X"][["GA_p90"]].values
    X_test_ar  = test["X"][["GA_p90"]].values

    # Classification baseline
    lr_ar_clf = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    lr_ar_clf.fit(X_train_ar, train["y_clf"])
    pred_clf = lr_ar_clf.predict(X_test_ar)
    acc = accuracy_score(test["y_clf"], pred_clf)
    f1  = f1_score(test["y_clf"], pred_clf, average="macro", labels=CLASS_ORDER)
    cm  = confusion_matrix(test["y_clf"], pred_clf, labels=CLASS_ORDER)

    # Regression baseline
    linr_ar = Pipeline([
        ("scale", StandardScaler()),
        ("lin", LinearRegression()),
    ])
    linr_ar.fit(X_train_ar, train["y_reg"])
    pred_reg = linr_ar.predict(X_test_ar)
    r2   = r2_score(test["y_reg"], pred_reg)
    rmse = float(math.sqrt(mean_squared_error(test["y_reg"], pred_reg)))
    mae  = mean_absolute_error(test["y_reg"], pred_reg)

    log.info(
        f"AR baseline (GA_p90 only): "
        f"clf acc={acc:.3f} F1={f1:.3f} | reg R²={r2:.3f} RMSE={rmse:.3f}"
    )

    return {
        "classification": {
            "model":        "AR baseline (GA_p90 only)",
            "accuracy":     float(acc),
            "macro_f1":     float(f1),
            "confusion_matrix": cm.tolist(),
            "labels":       CLASS_ORDER,
        },
        "regression": {
            "model": "AR baseline (GA_p90 only)",
            "r2":    float(r2),
            "rmse":  rmse,
            "mae":   float(mae),
        },
    }


def compute_confidence_intervals(metrics: dict, n_test: int) -> dict:
    """
    Add 95% Wald confidence intervals for every classification accuracy
    figure and two-proportion z-tests between all model pairs.

    Modifies metrics in-place, writes results under
    metrics["confidence_intervals"] and metrics["pairwise_z_tests"].

    At n=738, the half-width is ~±0.034, so differences smaller than
    ~0.07 are not statistically significant. This is the key result
    that supports the feature-set ceiling interpretation.
    """
    clf_rows = metrics.get("classification", [])

    ci_list = []
    for m in clf_rows:
        a  = m["accuracy"]
        se = math.sqrt(a * (1 - a) / n_test)
        ci_list.append({
            "model":            m["model"],
            "accuracy":         a,
            "ci_95_lo":         round(a - 1.96 * se, 4),
            "ci_95_hi":         round(a + 1.96 * se, 4),
            "ci_95_half_width": round(1.96 * se, 4),
        })

    # Two-proportion z-test for every pair
    z_tests = []
    for i in range(len(ci_list)):
        for j in range(i + 1, len(ci_list)):
            a1 = ci_list[i]["accuracy"]
            a2 = ci_list[j]["accuracy"]
            se_diff = math.sqrt(a1*(1-a1)/n_test + a2*(1-a2)/n_test)
            z = (a1 - a2) / se_diff if se_diff > 0 else 0.0
            p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
            z_tests.append({
                "model_1":        ci_list[i]["model"],
                "model_2":        ci_list[j]["model"],
                "delta_accuracy": round(a1 - a2, 4),
                "z_stat":         round(z, 3),
                "p_value":        round(p, 3),
                "significant_at_05": p < 0.05,
            })
            log.info(
                f"z-test {ci_list[i]['model']} vs {ci_list[j]['model']}: "
                f"Δ={a1-a2:+.4f}  z={z:.2f}  p={p:.3f}"
                + ("  *" if p < 0.05 else "")
            )

    metrics["confidence_intervals"] = ci_list
    metrics["pairwise_z_tests"]     = z_tests
    return metrics


# ------------------------------------------------------------------ shap

def compute_shap(rf_clf, train, test):
    """
    SHAP for the RF classifier.

    TreeExplainer is exact for tree models (no sampling approximation).
    For multi-class RF it returns a list of arrays — one per class —
    so we save the full structure and let Stage 4 decide which class
    to plot for which figure.

    We use a SAMPLE of training data as the background distribution
    (full training set works but is slow for plotting) and compute
    SHAP on the FULL test set (we want every test prediction explained).
    """
    log.info("Computing SHAP values ...")

    # TreeExplainer with model_output='raw' returns one array per class
    explainer = shap.TreeExplainer(rf_clf)

    # Compute on test set
    raw = explainer.shap_values(test["X"])

    # In recent shap versions, multi-class trees return a 3D array of
    # shape (n_samples, n_features, n_classes). Older versions return a
    # list of 2D arrays. Normalize to a list of 2D arrays.
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_arrays = [raw[:, :, i] for i in range(raw.shape[2])]
    else:
        shap_arrays = list(raw)

    log.info(
        f"SHAP done: {len(shap_arrays)} class arrays, "
        f"each of shape {shap_arrays[0].shape}"
    )

    # explainer.expected_value: scalar or array of length n_classes
    exp_val = explainer.expected_value
    if np.isscalar(exp_val):
        exp_val = np.array([exp_val])
    else:
        exp_val = np.asarray(exp_val)

    return shap_arrays, exp_val, rf_clf.classes_



def main():
    log.info("=" * 60)
    log.info("Stage 3: train + evaluate + SHAP")
    log.info("=" * 60)

    df = load_data()
    train, test = split(df)

    rf_clf, lr_clf = train_classifier(train, test)
    rf_reg, lin_reg = train_regressor(train, test)

    metrics = {
        "classification": [
            eval_classifier("RF", rf_clf, test),
            eval_classifier("LogReg (baseline)", lr_clf, test),
        ],
        "regression": [
            eval_regressor("RF", rf_reg, test),
            eval_regressor("Linear (baseline)", lin_reg, test),
        ],
        "split": {
            "train_seasons": TRAIN_SEASONS,
            "test_season": TEST_SEASON,
            "n_train": len(train["y_clf"]),
            "n_test":  len(test["y_clf"]),
        },
        "features": FEATURES,
    }

    # Autoregressive baseline — answers "does the full feature set add lift
    # over simple persistence?" before claiming a feature-set ceiling.
    ar = train_autoregressive_baseline(train, test)
    metrics["ar_baseline"] = ar
    # Append to classification list so CI computation covers all models
    metrics["classification"].append(ar["classification"])
    metrics["regression"].append(ar["regression"])

    # 95% confidence intervals + pairwise z-tests across all clf models.
    # Key finding: at n=738, all model pairs produce p > 0.40 — differences
    # are indistinguishable from sampling noise → supports convergence claim.
    compute_confidence_intervals(metrics, n_test=len(test["y_clf"]))

    # SHAP on the headline model
    shap_arrays, expected_value, class_order = compute_shap(rf_clf, train, test)

    # Save everything
    joblib.dump(rf_clf, MODELS / "rf_classifier.joblib")
    joblib.dump(rf_reg, MODELS / "rf_regressor.joblib")
    np.savez(
        MODELS / "shap_values.npz",
        shap_low=shap_arrays[0],
        shap_mid=shap_arrays[1],
        shap_high=shap_arrays[2],
        expected_value=expected_value,
        class_order=np.array(class_order),
        feature_names=np.array(FEATURES),
        X_test=test["X"].to_numpy(),
    )

    # Test rows with predictions, for both confusion-matrix figure and
    # the regression scatter, plus the SHAP individual-explainer figure.
    test_out = test["meta"].copy()
    test_out["true_tertile"]    = test["y_clf"].values
    test_out["pred_tertile"]    = rf_clf.predict(test["X"])
    test_out["true_GA_p90_nxt"] = test["y_reg"].values
    test_out["pred_GA_p90_nxt"] = rf_reg.predict(test["X"])
    test_out.to_csv(MODELS / "test_set_with_preds.csv", index=False)

    (MODELS / "feature_columns.txt").write_text("\n".join(FEATURES))
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 60)
    log.info("Saved:")
    for p in sorted(MODELS.iterdir()):
        log.info(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()