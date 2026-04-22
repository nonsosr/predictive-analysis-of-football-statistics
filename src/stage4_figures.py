"""
Stage 4: Generate the four poster figures.

Produces four 300-DPI PNG files sized for an A2 poster's chart slots:
  fig_a_pipeline.png        Methodology pipeline diagram
  fig_b_confusion.png       Confusion matrix + headline metrics
  fig_c_shap.png            SHAP feature importance (per class)
  fig_d_regression.png      Predicted vs actual scatter

Inputs (./models/)
  metrics.json
  rf_classifier.joblib
  rf_regressor.joblib
  shap_values.npz
  test_set_with_preds.csv
  feature_columns.txt

Outputs (./figures/)
  one PNG per figure above

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

# University of Reading inspired palette. Primary navy is UoR brand;
# accent reds/greens used sparingly.
NAVY   = "#122F66"   # primary
BLUE   = "#1F58D2"   # secondary (header banner)
LIGHT  = "#E8EFFA"   # very light blue background tint
RED    = "#C0392B"   # negative / errors
GREEN  = "#2A9D8F"   # positive / correct
GREY   = "#6C757D"
DARK_TEXT = "#1A1A1A"

# Sequential and diverging colourmaps tied to the palette above
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
    Confusion matrix for the headline classifier (the best one in
    metrics — works for both baseline RF and tuned RF/XGB outputs).
    Both raw counts and row-normalized percentages in each cell.
    Headline metrics annotated below.
    """
    # Pick the best classifier in this metrics file. For Stage 5
    # (tuned), the metrics dict has a 'best_classifier' key that
    # tells us which row to highlight. For Stage 3 (baseline), we
    # default to the RF row.
    if "best_classifier" in metrics:
        best_name = metrics["best_classifier"]
        # The matching row will be e.g. "RF (tuned)" or "XGB (tuned)"
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

    # Tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.upper() for l in labels], fontsize=12, fontweight="bold")
    ax.set_yticklabels([l.upper() for l in labels], fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted next-season tertile", fontweight="bold")
    ax.set_ylabel("True next-season tertile", fontweight="bold")

    # Cell annotations: count on top, percent below
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if cm_pct[i, j] > 50 else DARK_TEXT
            ax.text(j, i - 0.12, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=18,
                    fontweight="bold", color=colour)
            ax.text(j, i + 0.22, f"{cm_pct[i, j]:.0f}%",
                    ha="center", va="center", fontsize=11, color=colour)

    # Title with headline metrics
    ax.set_title(
        f"Next-Season Tertile Classification ({title_model})\n"
        f"Accuracy = {best['accuracy']:.1%}   ·   Macro-F1 = {best['macro_f1']:.2f}",
        pad=18,
    )

    # Per-class F1 footer. Darker colour (was GREY which rendered too
    # pale on print) and lower y-position so it's clearly separated
    # from the 'Predicted next-season tertile' axis label above it.
    FOOTER = "#4A5566"   # darker than GREY, still clearly secondary
    f1s = " · ".join(
        f"{c.upper()}: F1={best['report'][c]['f1-score']:.2f}" for c in labels
    )
    # Move closer to the bottom edge and make the text non-italic so
    # it reads more legibly at poster scale.
    fig.text(0.5, 0.005, f"Per-class F1   {f1s}",
             ha="center", fontsize=10, color=FOOTER)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of true class", rotation=270, labelpad=15)

    # Reserve a little bottom padding so the footer sits clearly below
    # the x-axis label (rather than touching it).
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

    # Order features by overall importance (sum across classes)
    overall = imp_low + imp_mid + imp_high
    order = np.argsort(overall)[::-1]
    top_n = 12
    order = order[:top_n]

    feat = [feature_names[i] for i in order][::-1]   # reverse for h-bar
    lo = imp_low[order][::-1]
    md = imp_mid[order][::-1]
    hi = imp_high[order][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.8), sharey=True, sharex=True)
    titles = [f"Predicting LOW", f"Predicting MID", f"Predicting HIGH"]
    series = [lo, md, hi]
    colours = [RED, GREY, BLUE]

    # Shared x-limit so panels are visually comparable. Without sharing,
    # the LOW panel (big SHAP values) and HIGH panel (small ones) would
    # show same-length bars for very different magnitudes. Extra 25%
    # margin (rather than 15%) because labels now sit OUTSIDE the bars.
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
        # Value labels on EVERY bar, uniformly styled: all placed just
        # outside the bar end in dark text at the same size. This avoids
        # the two-style inconsistency (white-inside for long bars,
        # dark-outside for short bars) which reads as two different
        # labelling systems on one chart.
        for bar, v in zip(bars, vals):
            ax.text(v + xmax * 0.012, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", ha="left", va="center",
                    fontsize=9, color=DARK_TEXT, fontweight="bold")

    fig.suptitle("Feature Importance by Predicted Class (SHAP)",
                 fontsize=16, fontweight="bold", color=NAVY, y=0.97)
    # Darker footer colour for readability at poster scale; no italic so
    # it doesn't get lost visually.
    FOOTER = "#4A5566"
    fig.text(0.5, 0.015,
             "Higher bar = stronger influence on this class's probability. "
             "Different features matter for different player profiles.",
             ha="center", fontsize=10, color=FOOTER)

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# regression

def fig_d_regression(out_path: Path, metrics: dict, preds_path: Path) -> None:
    """
    Predicted vs actual G+A/90 on the test set.

    Diagonal y=x reference line. Points coloured by absolute error using
    a SEQUENTIAL light-to-red palette (error is unidirectional, so a
    diverging palette would be misleading). Top-3 over- and under-
    predicted players annotated by name — grounds the abstract scatter
    in actual football.
    """
    df = pd.read_csv(preds_path)

    # Pick the best regressor (works for both baseline and tuned metrics).
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

    # Annotate the most extreme over/under predictions (3 each). Push
    # right-side annotations to the LEFT so they don't collide with
    # the colorbar.
    df["_err_signed"] = df["pred_GA_p90_nxt"] - df["true_GA_p90_nxt"]
    over  = df.nlargest(3, "_err_signed")    # over-predicted
    under = df.nsmallest(3, "_err_signed")   # under-predicted (e.g. breakouts)
    for _, row in pd.concat([over, under]).iterrows():
        # If point is in the right half, annotate to the LEFT instead
        x_off = 8 if row["true_GA_p90_nxt"] < lim * 0.6 else -65
        y_off = 8
        ax.annotate(
            row["player"],
            xy=(row["true_GA_p90_nxt"], row["pred_GA_p90_nxt"]),
            xytext=(x_off, y_off), textcoords="offset points",
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


# main

def main():
    log.info("=" * 60)
    log.info("Stage 4: poster figures")
    log.info("=" * 60)

    # Prefer tuned outputs if Stage 5 has run; fall back to baseline
    # if not. This way the same script regenerates the right figures
    # whether you've tuned or not.
    if (MODELS / "metrics_tuned.json").exists():
        log.info("Using TUNED model outputs from Stage 5")
        metrics_path = MODELS / "metrics_tuned.json"
        shap_path    = MODELS / "shap_values_tuned.npz"
        preds_path   = MODELS / "test_set_with_preds_tuned.csv"
    else:
        log.info("Using BASELINE model outputs from Stage 3")
        metrics_path = MODELS / "metrics.json"
        shap_path    = MODELS / "shap_values.npz"
        preds_path   = MODELS / "test_set_with_preds.csv"

    metrics = json.load(open(metrics_path))

    fig_a_pipeline (FIGS / "fig_a_pipeline.png",  metrics)
    fig_b_confusion(FIGS / "fig_b_confusion.png", metrics)
    fig_c_shap     (FIGS / "fig_c_shap.png",      shap_path)
    fig_d_regression(FIGS / "fig_d_regression.png", metrics, preds_path)

    log.info("=" * 60)
    log.info("Done. Files:")
    for p in sorted(FIGS.iterdir()):
        log.info(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()