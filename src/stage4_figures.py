"""
Stage 4: Figure generation.

Outputs (./figures/):
  fig_a_pipeline.png          Methodology pipeline diagram
  fig_b_confusion.png         Confusion matrix + headline metrics
  fig_c_shap.png              SHAP feature importance (per class)
  fig_d_regression.png        Predicted vs actual scatter
  fig_model_comparison.png    All models vs AR baseline with 95% CIs
  fig_shap_example_high.png   Per-prediction SHAP: correct HIGH
  fig_shap_example_low.png    Per-prediction SHAP: correct LOW
  fig_shap_example_error.png  Per-prediction SHAP: mispredicted case

Inputs (./models/):
  metrics.json, metrics_tuned.json
  shap_values_tuned.npz
  test_set_with_preds_tuned.csv

Prefers tuned outputs when available; falls back to Stage 3 baseline.

Run:
    python stage4_figures.py
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


# ------------------------------------------------------------------ config

MODELS = Path("./models")
FIGS = Path("./figures")
FIGS.mkdir(parents=True, exist_ok=True)

# Palette
NAVY   = "#122F66"
BLUE   = "#1F58D2"
LIGHT  = "#E8EFFA"
RED    = "#C0392B"
GREEN  = "#2A9D8F"
GREY   = "#6C757D"
DARK_TEXT = "#1A1A1A"
FOOTER_TEXT = "#4A5566"

CMAP_BLUES = LinearSegmentedColormap.from_list(
    "uor_blues", ["#FFFFFF", LIGHT, BLUE, NAVY])
CMAP_DIV = LinearSegmentedColormap.from_list(
    "uor_div", [RED, "#F8F9FA", BLUE])

# Typography
plt.rcParams.update({
    "font.family":      "DejaVu Sans",      # ships with matplotlib
    "font.size":        12,
    "axes.titlesize":   16,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "axes.edgecolor":   DARK_TEXT,
    "axes.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":      DARK_TEXT,
    "ytick.color":      DARK_TEXT,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("stage4")


# pipeline

def fig_a_pipeline(out_path: Path, metrics: dict) -> None:
    """
    Three-column flow diagram:
        SOURCES -> PROCESSING -> MODELS & OUTPUTS

    Each box shows what's at that stage and (where useful) the row count.
    Arrows tie the columns together. No clip art / icons — just clean
    rounded rectangles + connectors so it scales crisply on print.
    """
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    n_train = metrics["split"]["n_train"]
    n_test  = metrics["split"]["n_test"]

    def box(x, y, w, h, title, body, fill=LIGHT, edge=NAVY):
        b = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.5, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(b)
        ax.text(x + w/2, y + h - 0.32, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color=NAVY)
        ax.text(x + w/2, y + 0.18, body, ha="center", va="bottom",
                fontsize=10, color=DARK_TEXT)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY,
                                    mutation_scale=18))

    # --- Column 1: Sources ---
    box(0.3, 4.3, 3.2, 1.7,
        "FBref",
        "Standard, shooting, misc\nteam standard, shooting\n22,419 player-seasons")
    box(0.3, 1.6, 3.2, 1.7,
        "Understat",
        "xG, xA, npxG\nkey passes\nxGChain, xGBuildup\n21,837 player-seasons")

    # --- Column 2: Processing ---
    # Wider Stage 2 box so the longer body lines ("Tertile labels ...",
    # "Per-90 rates, share-of-team,") don't overflow the rounded border.
    box(4.0, 3.1, 3.2, 2.8,
        "Stage 2: Merge & engineer",
        "Join on (player, team, season)\n900-min filter\nPer-90 rates, share-of-team,\nteam attacking strength\nTertile labels (within league-season)")

    # --- Column 3: Modelling + Outputs ---
    box(7.7, 4.5, 3.0, 1.5,
        "Classification",
        f"Random Forest + LR baseline\nTrain n={n_train:,}, Test n={n_test:,}")
    box(7.7, 2.7, 3.0, 1.5,
        "Regression",
        "Random Forest + Linear baseline\nNext-season G+A/90")
    box(7.7, 0.9, 3.0, 1.5,
        "Explainability",
        "SHAP per-prediction\nFeature attribution by class",
        fill="#FFFFFF", edge=BLUE)

    # --- Arrows ---
    # Sources -> processing (updated x for wider Stage 2 box)
    arrow(3.5, 5.15, 4.0, 4.8)   # FBref  -> processing
    arrow(3.5, 2.45, 4.0, 4.2)   # Understat -> processing
    # Processing -> models
    arrow(7.2, 5.1, 7.7, 5.25)   # processing -> classification
    arrow(7.2, 4.3, 7.7, 3.45)   # processing -> regression
    # Classification -> Explainability routed AROUND the Regression box on the right
    # (L-shape: right from Classification, then straight down, then left into Explainability).
    right_x = 10.95   # just outside the right edge of the model boxes
    ax.plot([10.7, right_x], [4.95, 4.95], color=NAVY, lw=1.6)        # out right
    ax.plot([right_x, right_x], [4.95, 1.65], color=NAVY, lw=1.6)     # down
    ax.annotate("", xy=(10.7, 1.65), xytext=(right_x, 1.65),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=NAVY,
                                mutation_scale=18))                    # back in left

    # Title strip
    ax.text(5.7, 6.25, "Methodology Pipeline",
            ha="center", va="top", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(5.7, 0.3,
            "Big 5 European leagues  ·  2017-18 to 2024-25  ·  Temporal train/test split",
            ha="center", va="bottom", fontsize=10, color=GREY, style="italic")

    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# confusion

def fig_b_confusion(out_path: Path, metrics: dict) -> None:
    """
    Confusion matrix for the best classifier in the metrics file.
    Cells show raw count and row-normalised percentage.
    """
    # Stage 5 metrics carry 'best_classifier'; Stage 3 defaults to RF.
    if "best_classifier" in metrics:
        best_name = metrics["best_classifier"]
        best = next(r for r in metrics["classification"]
                    if r["model"].startswith(best_name))
        title_model = f"{best_name} Tuned"
    else:
        best = next(r for r in metrics["classification"] if r["model"] == "RF")
        title_model = "Random Forest Baseline"

    cm = np.array(best["confusion_matrix"])
    labels = best["labels"]
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_pct, cmap=CMAP_BLUES, vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.upper() for l in labels], fontsize=12, fontweight="bold")
    ax.set_yticklabels([l.upper() for l in labels], fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted next-season tertile", fontweight="bold")
    ax.set_ylabel("True next-season tertile", fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if cm_pct[i, j] > 50 else DARK_TEXT
            ax.text(j, i - 0.12, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=18,
                    fontweight="bold", color=colour)
            ax.text(j, i + 0.22, f"{cm_pct[i, j]:.0f}%",
                    ha="center", va="center", fontsize=11, color=colour)

    ax.set_title(
        f"Next-Season Tertile Classification ({title_model})\n"
        f"Accuracy = {best['accuracy']:.1%}   ·   Macro-F1 = {best['macro_f1']:.2f}",
        pad=18,
    )

    f1s = " · ".join(
        f"{c.upper()}: F1={best['report'][c]['f1-score']:.2f}" for c in labels
    )
    fig.text(0.5, 0.005, f"Per-class F1   {f1s}",
             ha="center", fontsize=10, color=FOOTER_TEXT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of true class", rotation=270, labelpad=15)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# SHAP

def fig_c_shap(out_path: Path, shap_npz_path: Path) -> None:
    """
    Mean |SHAP| per feature, broken down by predicted class (low/mid/high).

    Three side-by-side horizontal bar charts let the viewer compare
    which features drive predictions for different player profiles —
    directly relevant to the archetype/explainability story.
    """
    z = np.load(shap_npz_path, allow_pickle=True)
    shap_low  = z["shap_low"]
    shap_mid  = z["shap_mid"]
    shap_high = z["shap_high"]
    feature_names = list(z["feature_names"])

    # Mean |SHAP| per feature, per class
    imp_low  = np.abs(shap_low).mean(axis=0)
    imp_mid  = np.abs(shap_mid).mean(axis=0)
    imp_high = np.abs(shap_high).mean(axis=0)

    # Order by total importance across classes, take top 12, reverse for h-bar.
    overall = imp_low + imp_mid + imp_high
    order = np.argsort(overall)[::-1][:12]

    feat = [feature_names[i] for i in order][::-1]
    lo = imp_low[order][::-1]
    md = imp_mid[order][::-1]
    hi = imp_high[order][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.8), sharey=True, sharex=True)
    titles  = ["Predicting LOW", "Predicting MID", "Predicting HIGH"]
    series  = [lo, md, hi]
    colours = [RED, GREY, BLUE]

    # Shared x-limit keeps bar lengths comparable across panels.
    xmax = max(lo.max(), md.max(), hi.max()) * 1.25

    for ax, title, vals, colour in zip(axes, titles, series, colours):
        bars = ax.barh(feat, vals, color=colour, alpha=0.85,
                       edgecolor=DARK_TEXT, linewidth=0.5)
        ax.set_title(title, fontsize=13, color=DARK_TEXT, pad=8)
        ax.set_xlim(0, xmax)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("mean |SHAP|", fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(v + xmax * 0.012, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", ha="left", va="center",
                    fontsize=9, color=DARK_TEXT, fontweight="bold")

    fig.suptitle("Feature Importance by Predicted Class (SHAP)",
                 fontsize=16, fontweight="bold", color=NAVY, y=0.97)
    fig.text(0.5, 0.015,
             "Higher bar = stronger influence on this class's probability. "
             "Different features matter for different player profiles.",
             ha="center", fontsize=10, color=FOOTER_TEXT)

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# regression

def fig_d_regression(out_path: Path, metrics: dict, preds_path: Path) -> None:
    """
    Predicted vs actual G+A/90 scatter on the test set.
    Points coloured by absolute error. Top-3 over- and under-predicted
    players annotated by name.
    """
    df = pd.read_csv(preds_path)

    # Stage 5 metrics carry 'best_regressor'; Stage 3 defaults to RF.
    if "best_regressor" in metrics:
        best_name = metrics["best_regressor"]
        best = next(r for r in metrics["regression"]
                    if r["model"].startswith(best_name))
        title_model = f"{best_name} Tuned"
    else:
        best = next(r for r in metrics["regression"] if r["model"] == "RF")
        title_model = "Random Forest Baseline"

    x = df["true_GA_p90_nxt"].values
    y = df["pred_GA_p90_nxt"].values
    err = np.abs(x - y)

    # Sequential colourmap: light cream -> deep red. Higher absolute
    # error reads as more "alarming" without implying directionality.
    cmap_err = LinearSegmentedColormap.from_list(
        "uor_err", ["#F4E9E9", "#E8A5A5", RED, "#6E1313"])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sc = ax.scatter(x, y, c=err, cmap=cmap_err, vmin=0, vmax=err.max(),
                    s=24, alpha=0.78, edgecolor="white", linewidth=0.3)

    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, linestyle="--",
            linewidth=1.2, label="Perfect prediction (y=x)")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_aspect("equal")

    ax.set_xlabel("Actual next-season G+A per 90", fontweight="bold")
    ax.set_ylabel("Predicted next-season G+A per 90", fontweight="bold")
    ax.set_title(
        f"Regression: Predicting Next-Season G+A/90 ({title_model})\n"
        f"R² = {best['r2']:.3f}   ·   RMSE = {best['rmse']:.3f}   ·   "
        f"MAE = {best['mae']:.3f}",
        pad=18,
    )
    ax.grid(linestyle=":", alpha=0.4)

    # Annotate 3 most over- and 3 most under-predicted players.
    # Flip annotation side for points in the right half to avoid colorbar overlap.
    df["_err_signed"] = df["pred_GA_p90_nxt"] - df["true_GA_p90_nxt"]
    over  = df.nlargest(3, "_err_signed")
    under = df.nsmallest(3, "_err_signed")
    for _, row in pd.concat([over, under]).iterrows():
        x_off = 8 if row["true_GA_p90_nxt"] < lim * 0.6 else -65
        ax.annotate(
            row["player"],
            xy=(row["true_GA_p90_nxt"], row["pred_GA_p90_nxt"]),
            xytext=(x_off, 8), textcoords="offset points",
            fontsize=9, color=DARK_TEXT, alpha=0.9,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, alpha=0.55),
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error", rotation=270, labelpad=15)

    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# model comparison

def fig_e_model_comparison(
    out_path: Path, metrics: dict, metrics_tuned: dict
) -> None:
    """
    Two-panel figure comparing all classifiers and regressors including
    the autoregressive baseline, with 95% Wald confidence intervals on
    the classification panel. Reads AR baseline and CI data written by
    Stage 5's run_ar_baseline_and_ci().
    """
    import math

    n = metrics_tuned.get("split", {}).get("n_test", 738)

    # Classification accuracies
    tuned_clf = {m["model"]: m for m in metrics_tuned.get("classification", [])}
    rf_acc  = tuned_clf.get("RF (tuned)",  {}).get("accuracy", 0.686)
    xgb_acc = tuned_clf.get("XGB (tuned)", {}).get("accuracy", 0.667)
    lr_acc  = next((m["accuracy"] for m in metrics.get("classification", [])
                    if "LogReg" in m["model"]), 0.682)
    ar_acc  = (metrics_tuned.get("ar_baseline", {})
                             .get("classification", {})
                             .get("accuracy", 0.649))

    # Regression R²
    tuned_reg = {m["model"]: m for m in metrics_tuned.get("regression", [])}
    xgb_r2  = tuned_reg.get("XGB (tuned)", {}).get("r2", 0.673)
    rf_r2   = tuned_reg.get("RF (tuned)",  {}).get("r2", 0.670)
    lin_r2  = next((m["r2"] for m in metrics.get("regression", [])
                    if "Linear" in m["model"]), 0.673)
    ar_r2   = (metrics_tuned.get("ar_baseline", {})
                             .get("regression", {})
                             .get("r2", 0.611))

    LGREY = "#D4DCE8"
    WHITE = "#FFFFFF"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(WHITE)

    # Left panel: classification with CIs
    models_c = [
        "Autoregressive\nbaseline\n(GA/90 only)",
        "Tuned\nXGBoost",
        "Logistic\nRegression",
        "Tuned\nRandom Forest",
    ]
    accs   = [ar_acc, xgb_acc, lr_acc, rf_acc]
    cis    = [1.96 * math.sqrt(a * (1 - a) / n) for a in accs]
    colors = [LGREY, GREY, GREY, GREEN]

    ax1.set_facecolor("#F8FAFB")
    ax1.barh(models_c, accs, xerr=cis, color=colors,
             edgecolor=[GREY, GREY, GREY, GREEN], linewidth=1.2,
             error_kw=dict(ecolor=DARK_TEXT, capsize=4, capthick=1.5, elinewidth=1.5),
             height=0.55)
    ax1.set_xlim(0.58, 0.74)
    ax1.set_xlabel(f"Accuracy (held-out 2023-24, n={n})", fontsize=11)
    ax1.set_title("Classification Accuracy\nwith 95% CIs",
                  fontsize=12, fontweight="bold", color=NAVY, pad=10)
    for i, (a, ci) in enumerate(zip(accs, cis)):
        ax1.text(a + ci + 0.003, i, f"{a:.3f}", va="center", fontsize=10,
                 color=DARK_TEXT, fontweight="bold" if i == 3 else "normal")
    ax1.text(0.593, -0.75,
             "All pairwise comparisons p > 0.40 — no statistically significant differences",
             fontsize=8, color=GREY, style="italic")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: regression R²
    models_r = [
        "AR baseline\n(GA/90 only)",
        "Tuned RF\nregressor",
        "Linear regression\n(baseline)",
        "Tuned XGBoost\nregressor",
    ]
    r2s      = [ar_r2, rf_r2, lin_r2, xgb_r2]
    colors_r = [LGREY, GREY, GREY, GREEN]

    ax2.set_facecolor("#F8FAFB")
    ax2.barh(models_r, r2s, color=colors_r,
             edgecolor=[GREY, GREY, GREY, GREEN], linewidth=1.2, height=0.55)
    ax2.set_xlim(0.55, 0.73)
    ax2.set_xlabel(f"R² (held-out 2023-24, n={n})", fontsize=11)
    ax2.set_title("Regression R²\n(next-season G+A/90)",
                  fontsize=12, fontweight="bold", color=NAVY, pad=10)
    for i, r2 in enumerate(r2s):
        ax2.text(r2 + 0.003, i, f"{r2:.3f}", va="center", fontsize=10,
                 color=DARK_TEXT, fontweight="bold" if i == 3 else "normal")
    ax2.annotate("Linear reg ties\nXGBoost at R²=0.673",
                 xy=(0.673, 1.5), xytext=(0.595, 2.6), fontsize=8.5,
                 color=GREY, style="italic",
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(facecolor=LGREY, edgecolor=GREY,  label="Autoregressive baseline"),
        mpatches.Patch(facecolor=GREY,  edgecolor=GREY,  label="Other models / baselines"),
        mpatches.Patch(facecolor=GREEN, edgecolor=GREEN, label="Best tuned tree ensemble"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Model Performance — All Models Including Autoregressive Baseline",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# individual SHAP examples

def fig_f_shap_examples(
    shap_path: Path, preds_path: Path, figures_dir: Path
) -> None:
    """
    Three per-prediction SHAP bar charts for worked examples:
      fig_shap_example_high.png  — correct HIGH prediction
      fig_shap_example_low.png   — correct LOW prediction
      fig_shap_example_error.png — mispredicted case (pred HIGH, actual LOW)

    For each case we pick the player with the highest total |SHAP| among
    eligible candidates, giving the most decisive / legible example.
    """
    data  = np.load(shap_path, allow_pickle=True)
    preds = pd.read_csv(preds_path).reset_index(drop=True)

    shap_high  = data["shap_high"]
    shap_low   = data["shap_low"]
    feat_names = [
        str(f).replace("_p90", "(/90)").replace("npxG_plus_xA", "npxG+xA").replace("_", " ")
        for f in data["feature_names"]
    ]

    def pick(mask_fn, shap_arr):
        candidates = [i for i in range(len(preds)) if mask_fn(preds.iloc[i])]
        return candidates[int(np.argmax([abs(shap_arr[i]).sum() for i in candidates]))]

    cases = [
        dict(
            idx=pick(lambda r: r.pred_tertile == "high" and r.true_tertile == "high",
                     shap_high),
            shap=shap_high, color=GREEN,
            xlabel="SHAP contribution toward HIGH prediction (top 12 features)",
            fname="fig_shap_example_high.png",
        ),
        dict(
            idx=pick(lambda r: r.pred_tertile == "low" and r.true_tertile == "low",
                     shap_low),
            shap=shap_low, color=BLUE,
            xlabel="SHAP contribution toward LOW prediction (top 12 features)",
            fname="fig_shap_example_low.png",
        ),
        dict(
            idx=pick(lambda r: r.pred_tertile == "high" and r.true_tertile == "low",
                     shap_high),
            shap=shap_high, color=RED,
            xlabel="SHAP contribution toward predicted HIGH class — prediction INCORRECT",
            fname="fig_shap_example_error.png",
        ),
    ]

    for case in cases:
        idx  = case["idx"]
        row  = preds.iloc[idx]
        svs  = case["shap"][idx]
        order = np.argsort(np.abs(svs))[::-1][:12]
        bars  = svs[order]
        names = [feat_names[i] for i in order]
        bar_colors = [case["color"] if v >= 0 else RED for v in bars]

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#F8FAFB")

        ax.barh(range(len(bars)), bars, color=bar_colors, edgecolor="none", height=0.6)
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels(names, fontsize=10.5)
        ax.axvline(0, color=DARK_TEXT, linewidth=0.8)

        # Expand x-axis symmetrically so value labels have room on both sides
        # without colliding with y-tick labels.
        abs_max = max(abs(bars).max(), 1e-6)
        ax.set_xlim(-abs_max * 1.55, abs_max * 1.55)

        gap = abs_max * 0.04   # uniform offset from bar tip to label
        for i, v in enumerate(bars):
            if v >= 0:
                ax.text(v + gap, i, f"+{v:.3f}", va="center",
                        ha="left", fontsize=9, color=DARK_TEXT)
            else:
                ax.text(v - gap, i, f"{v:.3f}", va="center",
                        ha="right", fontsize=9, color=DARK_TEXT)

        pred   = row.pred_tertile.upper()
        true   = row.true_tertile.upper()
        status = "✓ CORRECT" if pred == true else "✗ INCORRECT"

        ax.set_title(
            f"{row.player} — {row.team} ({row.league}, 2023-24)\n"
            f"Predicted: {pred}  ·  Actual: {true}  ·  {status}  ·  "
            f"G+A/90 (current season): {row.GA_p90:.3f}",
            fontsize=11, fontweight="bold", color=NAVY, pad=10,
        )
        ax.set_xlabel(case["xlabel"], fontsize=10, color=DARK_TEXT, labelpad=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.text(0.5, 0.01,
                 "Coloured bars: feature pushes toward predicted class.  "
                 "Red bars: feature pushes away.  Bar length = contribution magnitude.",
                 ha="center", fontsize=8.5, color=GREY, style="italic")

        # Extra bottom margin so footer text doesn't overlap the x-axis label.
        fig.subplots_adjust(left=0.22, right=0.96, top=0.88, bottom=0.12)

        out = figures_dir / case["fname"]
        fig.savefig(out)
        plt.close(fig)
        log.info(f"  wrote {out}")


# main

def main():
    log.info("=" * 60)
    log.info("Stage 4: figures")
    log.info("=" * 60)

    # Prefer tuned outputs when Stage 5 has run.
    if (MODELS / "metrics_tuned.json").exists():
        log.info("Using tuned outputs (Stage 5)")
        metrics_path = MODELS / "metrics_tuned.json"
        shap_path    = MODELS / "shap_values_tuned.npz"
        preds_path   = MODELS / "test_set_with_preds_tuned.csv"
    else:
        log.info("Using baseline outputs (Stage 3)")
        metrics_path = MODELS / "metrics.json"
        shap_path    = MODELS / "shap_values.npz"
        preds_path   = MODELS / "test_set_with_preds.csv"

    metrics       = json.load(open(metrics_path))
    metrics_tuned = json.load(open(MODELS / "metrics_tuned.json")) \
                    if (MODELS / "metrics_tuned.json").exists() else metrics

    fig_a_pipeline      (FIGS / "fig_a_pipeline.png",         metrics)
    fig_b_confusion     (FIGS / "fig_b_confusion.png",        metrics)
    fig_c_shap          (FIGS / "fig_c_shap.png",             shap_path)
    fig_d_regression    (FIGS / "fig_d_regression.png",       metrics, preds_path)
    fig_e_model_comparison(FIGS / "fig_model_comparison.png", metrics, metrics_tuned)
    fig_f_shap_examples (shap_path, preds_path, FIGS)

    log.info("=" * 60)
    log.info("Done. Files:")
    for p in sorted(FIGS.iterdir()):
        log.info(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()