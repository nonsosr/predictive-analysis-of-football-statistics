"""
Stage 6: Data-driven player archetypes via clustering.

Identifies natural groupings of attacking players based on their
play-style signatures (finishing, creativity, defensive contribution),
then compares how the Stage 5 tuned classifier's SHAP attributions
differ by archetype. This is the distinctive analytical contribution
of the project: moving from "the model is X% accurate" to "the model
relies on DIFFERENT features for DIFFERENT kinds of player."

Clustering decisions (documented for the writeup):

- Features: 14-feature STYLE subset — per-90 finishing, creativity,
  defensive workload, and physicality. Deliberately excludes context
  variables (team strength, age, minutes%) so clusters capture play
  style, not club identity or career stage.

- Algorithms: K-Means AND Gaussian Mixture Model, compared via
  silhouette score and the Davies-Bouldin index. Running both
  strengthens the methodology — silhouette agreement between two
  different algorithmic families is more convincing than either alone.

- k selection: silhouette sweep over k=2..10, elbow on inertia as
  sanity check.

- 2D viz: UMAP (if available) + PCA side-by-side. PCA gives
  interpretable axes (PC1 loadings etc); UMAP gives cleaner cluster
  separation. Showing both is defensible against "why did you pick
  this viz".

- SHAP per cluster: uses the tuned classifier from Stage 5, re-applied
  to each cluster separately so we can name archetypes from the
  features that matter most to each group.

Inputs (./data/processed/, ./models/)
  player_seasons_eligible.csv
  rf_classifier_tuned.joblib   (Stage 5 output)
  shap_values_tuned.npz        (Stage 5 SHAP on TEST set)
  test_set_with_preds_tuned.csv

Outputs (./models/, ./figures/)
  models/clusters_kmeans.csv           (row per player-season)
  models/clusters_gmm.csv
  models/cluster_k_selection.json      (silhouette + inertia table)
  models/cluster_profiles.json         (mean feature values per cluster)
  models/cluster_exemplars.json        (top players per cluster)
  models/shap_per_cluster.json         (mean |SHAP| per feature, per cluster)
  figures/fig_e_cluster_selection.png
  figures/fig_f_cluster_viz.png
  figures/fig_g_cluster_profiles.png
  figures/fig_h_shap_per_cluster.png

Run from project root:
  python src/stage6_cluster.py
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ config

PROC = Path("./data/processed")
MODELS = Path("./models")
FIGS = Path("./figures")
MODELS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Style-signature features only. Context vars (team/age/minutes) excluded
# so we cluster on HOW the player plays, not WHO they play for.
STYLE_FEATURES = [
    # Finishing
    "npxG_p90", "xG_p90", "shots_p90", "shots_on_target_p90",
    "xG_per_shot", "np_goals_p90",
    # Creativity
    "xA_p90", "key_passes_p90", "xGChain_p90", "xGBuildup_p90",
    "crosses_p90",
    # Defensive workload
    "interceptions_p90", "tackles_won_p90",
    # Physicality
    "fouls_drawn_p90",
]

K_RANGE = range(2, 11)          # silhouette sweep
RANDOM_STATE = 42

# Brand-consistent palette (matches Stage 4 figures)
NAVY   = "#122F66"
BLUE   = "#1F58D2"
LIGHT  = "#E8EFFA"
RED    = "#C0392B"
GREEN  = "#2A9D8F"
GREY   = "#6C757D"
DARK_TEXT = "#1A1A1A"

# Distinct colours for up to 8 clusters. Avoid pure red/green together
# (colourblindness), and avoid the exact NAVY/BLUE used elsewhere so
# clusters don't clash with poster/figure accents.
CLUSTER_PALETTE = [
    "#1F58D2", "#C0392B", "#2A9D8F", "#E67E22",
    "#8E44AD", "#16A085", "#D35400", "#34495E",
]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.edgecolor":   DARK_TEXT,
    "axes.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("stage6")


# ------------------------------------------------------------------ data

def load_data():
    """
    Load the eligible player-seasons and prepare the STYLE-feature matrix.
    Drop rows with missing values in any style feature (rare after the
    900-min filter, but KMeans/GMM can't handle NaN).
    """
    df = pd.read_csv(PROC / "player_seasons_eligible.csv")
    log.info(f"Loaded {len(df):,} eligible player-seasons")

    before = len(df)
    df = df.dropna(subset=STYLE_FEATURES).reset_index(drop=True)
    if len(df) < before:
        log.info(f"Dropped {before - len(df):,} rows with missing style features")

    X_raw = df[STYLE_FEATURES].copy()

    # Standardize: clustering is distance-based, unscaled features let
    # the high-variance ones (shots_p90) dominate over low-variance ones
    # (tackles_won_p90). Every feature should have mean 0 and std 1.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    log.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    return df, X, X_raw


# ------------------------------------------------------------------ k selection

def sweep_k(X: np.ndarray) -> dict:
    """
    For each candidate k: fit both K-Means and GMM, compute silhouette
    (higher = better, max 1.0) and Davies-Bouldin (lower = better).
    Also record KMeans inertia for elbow check.

    Silhouette computation is O(n²) in memory if done on the full 12k
    matrix. We subsample to 5000 rows for scoring to keep this fast
    without biasing the result.
    """
    log.info("Sweeping k for silhouette + DB score ...")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X[sample_idx]

    results = {"k": [], "kmeans_silhouette": [], "kmeans_db": [],
               "kmeans_inertia": [], "gmm_silhouette": [], "gmm_db": [],
               "gmm_bic": []}

    for k in K_RANGE:
        # K-Means
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km_labels_full = km.fit_predict(X)
        km_labels_sample = km_labels_full[sample_idx]

        km_sil = silhouette_score(X_sample, km_labels_sample)
        km_db  = davies_bouldin_score(X, km_labels_full)

        # GMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # GMM convergence chatter
            gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                                  covariance_type="full", max_iter=200)
            gmm.fit(X)
            gmm_labels_full = gmm.predict(X)
            gmm_labels_sample = gmm_labels_full[sample_idx]

        gmm_sil = silhouette_score(X_sample, gmm_labels_sample)
        gmm_db  = davies_bouldin_score(X, gmm_labels_full)
        gmm_bic = gmm.bic(X)

        results["k"].append(k)
        results["kmeans_silhouette"].append(float(km_sil))
        results["kmeans_db"].append(float(km_db))
        results["kmeans_inertia"].append(float(km.inertia_))
        results["gmm_silhouette"].append(float(gmm_sil))
        results["gmm_db"].append(float(gmm_db))
        results["gmm_bic"].append(float(gmm_bic))

        log.info(f"  k={k}: KM sil={km_sil:.3f} db={km_db:.3f}   "
                 f"GMM sil={gmm_sil:.3f} db={gmm_db:.3f}")

    # Pick the k with best average of (KMeans sil, GMM sil) — simple
    # agreement heuristic. Could pick per-algorithm but then you're
    # comparing solutions at different k's, which is messier.
    avg_sil = [(km + gm) / 2 for km, gm
               in zip(results["kmeans_silhouette"], results["gmm_silhouette"])]
    best_idx = int(np.argmax(avg_sil))
    best_k = results["k"][best_idx]
    log.info(f"Chosen k (highest average silhouette): {best_k}")

    results["chosen_k"] = best_k
    return results


# ------------------------------------------------------------------ fit final

def fit_final(X: np.ndarray, k: int):
    """Fit both algorithms at the chosen k and return labels."""
    log.info(f"Fitting final K-Means and GMM with k={k}")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                              covariance_type="full", max_iter=300)
        gmm.fit(X)
    gmm_labels = gmm.predict(X)
    gmm_proba  = gmm.predict_proba(X)   # soft assignment (useful for writeup)

    return km, km_labels, gmm, gmm_labels, gmm_proba


# ------------------------------------------------------------------ profiles

def cluster_profiles(df: pd.DataFrame, X_raw: pd.DataFrame, labels: np.ndarray,
                     algo_name: str) -> dict:
    """
    For each cluster:
      - mean feature values (raw, not standardized — interpretable)
      - size
      - league + position breakdown (position via FBref 'position' col if present)
      - top exemplar players by proximity to the cluster centroid
    """
    out = {"algorithm": algo_name, "clusters": []}

    df = df.copy()
    df["_cluster"] = labels

    for c in sorted(np.unique(labels)):
        mask = labels == c
        Xc = X_raw[mask]
        dfc = df[mask]

        centroid = Xc.mean()

        # Exemplars: players closest to the cluster centroid in
        # standardized space. Use z-scores so every feature weights
        # equally (same reason we scale for clustering itself).
        Xc_z = (Xc - X_raw.mean()) / X_raw.std()
        centroid_z = Xc_z.mean()
        dists = np.linalg.norm(Xc_z - centroid_z, axis=1)
        closest_idx = np.argsort(dists)[:10]
        exemplars = dfc.iloc[closest_idx][["player", "team", "league", "season"]]

        cluster_info = {
            "cluster_id": int(c),
            "size": int(mask.sum()),
            "share": float(mask.mean()),
            "mean_features": {f: float(v) for f, v in centroid.items()},
            "top_leagues": dfc["league"].value_counts().head(3).to_dict(),
            "exemplars": exemplars.to_dict(orient="records"),
        }
        out["clusters"].append(cluster_info)

        top_features = centroid.sort_values(ascending=False).head(3)
        log.info(f"  Cluster {c} (n={mask.sum()}, {mask.mean():.1%}): "
                 f"top features = {', '.join(f'{f}={v:.2f}' for f, v in top_features.items())}")

    return out


# ------------------------------------------------------------------ SHAP per cluster

def shap_per_cluster(df: pd.DataFrame, X_raw: pd.DataFrame, km_labels: np.ndarray,
                     k: int) -> dict:
    """
    Cross-reference clusters with SHAP values from the Stage 5 tuned
    classifier.

    The SHAP file is computed on the TEST SET only (738 rows from
    2023-24). So we intersect the clustering (done on all 12,848 rows)
    with the test set, and report mean |SHAP| per feature per cluster
    only for players IN the test set. This keeps the SHAP analysis
    honest — we don't retroactively run SHAP on training data.
    """
    log.info("Loading Stage 5 SHAP outputs ...")
    shap_path = MODELS / "shap_values_tuned.npz"
    if not shap_path.exists():
        log.warning("shap_values_tuned.npz not found — skipping SHAP-per-cluster")
        return None

    z = np.load(shap_path, allow_pickle=True)
    # SHAP from Stage 5 was on the Stage-5 test features, not style
    # features. Its feature list is the FULL 23-feature training set.
    shap_feature_names = list(z["feature_names"])

    # The test-set predictions CSV links rows to players. We'll join
    # by (player, team, season) — a natural key.
    preds = pd.read_csv(MODELS / "test_set_with_preds_tuned.csv")
    # Make the join key type-consistent with df
    preds["season"] = preds["season"].astype(int)

    # Attach cluster labels to df by row order
    df_clusters = df.copy()
    df_clusters["_cluster"] = km_labels

    # Left-join preds onto clustered df on (player, team, season)
    joined = preds.merge(
        df_clusters[["player", "team", "season", "_cluster"]],
        on=["player", "team", "season"],
        how="inner",
    )
    log.info(f"  SHAP-to-cluster match: {len(joined):,} / {len(preds):,} test rows linked to a cluster")

    # Rebuild SHAP arrays keyed to the joined set. The SHAP file stored
    # 3 arrays (one per class): shap_low, shap_mid, shap_high. Each has
    # shape (n_test, n_features). We need to index into them at the
    # positions in the test set that appear in `joined`.
    shap_low  = z["shap_low"]
    shap_mid  = z["shap_mid"]
    shap_high = z["shap_high"]

    # preds rows are in the same order as the SHAP arrays. So we can
    # use the index of preds after the merge to figure out which SHAP
    # rows to pick.
    preds_with_idx = preds.reset_index().rename(columns={"index": "_shap_row"})
    joined_with_idx = joined.merge(
        preds_with_idx[["player", "team", "season", "_shap_row"]],
        on=["player", "team", "season"], how="left",
    )

    # Mean |SHAP| per cluster, per feature, per class (take max over classes
    # for the headline "how important is this feature for this cluster?"
    # story — mean-across-classes is another reasonable choice).
    per_cluster = {}
    for c in range(k):
        mask = joined_with_idx["_cluster"] == c
        rows = joined_with_idx.loc[mask, "_shap_row"].dropna().astype(int).values
        if len(rows) == 0:
            log.warning(f"  Cluster {c}: no test-set players; SHAP unavailable")
            per_cluster[str(c)] = {"n_test_rows": 0, "mean_abs_shap": {}}
            continue

        abs_per_class = [
            np.abs(shap_low[rows]).mean(axis=0),
            np.abs(shap_mid[rows]).mean(axis=0),
            np.abs(shap_high[rows]).mean(axis=0),
        ]
        # Headline: max across classes. Large value = feature is a
        # decisive predictor for at least one tertile for this cluster.
        headline = np.max(abs_per_class, axis=0)

        per_cluster[str(c)] = {
            "n_test_rows": int(len(rows)),
            "mean_abs_shap": {f: float(v) for f, v in zip(shap_feature_names, headline)},
            "by_class": {
                "low":  {f: float(v) for f, v in zip(shap_feature_names, abs_per_class[0])},
                "mid":  {f: float(v) for f, v in zip(shap_feature_names, abs_per_class[1])},
                "high": {f: float(v) for f, v in zip(shap_feature_names, abs_per_class[2])},
            },
        }
        log.info(f"  Cluster {c}: {len(rows)} test-set players; top SHAP feature = "
                 f"{shap_feature_names[int(np.argmax(headline))]}")

    return {
        "feature_names": shap_feature_names,
        "per_cluster":   per_cluster,
    }


# ------------------------------------------------------------------ figures

def fig_e_k_selection(out_path: Path, results: dict):
    """Silhouette + inertia + DB score across k. Two panels side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ks = results["k"]

    # Panel 1: silhouette (both algorithms)
    ax = axes[0]
    ax.plot(ks, results["kmeans_silhouette"], "o-", color=BLUE,  label="K-Means", lw=2)
    ax.plot(ks, results["gmm_silhouette"],    "s-", color=RED,   label="GMM",     lw=2)
    ax.axvline(results["chosen_k"], color=GREEN, linestyle=":", alpha=0.7,
               label=f"Chosen k={results['chosen_k']}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette (higher = better)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Panel 2: Davies-Bouldin (lower = better)
    ax = axes[1]
    ax.plot(ks, results["kmeans_db"], "o-", color=BLUE, label="K-Means", lw=2)
    ax.plot(ks, results["gmm_db"],    "s-", color=RED,  label="GMM",     lw=2)
    ax.axvline(results["chosen_k"], color=GREEN, linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Davies-Bouldin index")
    ax.set_title("Davies-Bouldin (lower = better)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Panel 3: elbow on inertia
    ax = axes[2]
    ax.plot(ks, results["kmeans_inertia"], "o-", color=BLUE, lw=2)
    ax.axvline(results["chosen_k"], color=GREEN, linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("K-Means inertia")
    ax.set_title("Elbow plot (K-Means)")
    ax.grid(alpha=0.3)

    fig.suptitle("Cluster-count selection: silhouette, Davies-Bouldin, and inertia",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


def fig_f_cluster_viz(out_path: Path, X: np.ndarray, km_labels: np.ndarray,
                      gmm_labels: np.ndarray, k: int):
    """
    2D projections. Left panel: PCA (interpretable axes).
    Right panel: UMAP (cleaner cluster separation). Falls back to t-SNE
    or second PCA view if UMAP isn't installed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- PCA ---
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    _plot_2d(axes[0], X_pca, km_labels, k,
             f"PCA projection\n(PC1: {var_explained[0]:.0%} var, "
             f"PC2: {var_explained[1]:.0%} var) — coloured by K-Means cluster")

    # --- UMAP (with fallback) ---
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                            n_neighbors=30, min_dist=0.3)
        X_low = reducer.fit_transform(X)
        title_suffix = "UMAP projection\ncoloured by K-Means cluster"
    except ImportError:
        log.warning("umap-learn not installed; using t-SNE instead")
        from sklearn.manifold import TSNE
        # Subsample for t-SNE speed if dataset large
        if len(X) > 5000:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X), 5000, replace=False)
            X_sub = X[idx]
            labels_sub = km_labels[idx]
        else:
            X_sub = X
            labels_sub = km_labels
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
        X_low = tsne.fit_transform(X_sub)
        km_labels = labels_sub  # align
        title_suffix = "t-SNE projection\ncoloured by K-Means cluster (5k sample)"

    _plot_2d(axes[1], X_low, km_labels, k, title_suffix)

    fig.suptitle("Player archetype clusters in 2D",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


def _plot_2d(ax, X2, labels, k, title):
    for c in range(k):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   s=10, alpha=0.55,
                   color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                   label=f"Cluster {c} (n={mask.sum():,})",
                   edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="best", markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_g_cluster_profiles(out_path: Path, profiles_km: dict, profiles_gmm: dict):
    """
    Heatmap of standardized mean feature values per cluster, side-by-side
    for K-Means and GMM. Rows=features, cols=clusters. Diverging
    colormap (red=below average, blue=above average).
    """
    def _to_df(profiles):
        rows = []
        for c in profiles["clusters"]:
            row = {"cluster": c["cluster_id"], **c["mean_features"]}
            rows.append(row)
        return pd.DataFrame(rows).set_index("cluster")

    km_df  = _to_df(profiles_km)
    gmm_df = _to_df(profiles_gmm)

    # Standardize columns (features) across all clusters so the heatmap
    # is readable — raw values span very different ranges.
    def _z(df):
        return (df - df.mean()) / df.std()
    km_z  = _z(km_df).T   # features on y, clusters on x
    gmm_z = _z(gmm_df).T

    cmap_div = LinearSegmentedColormap.from_list(
        "uor_div", [RED, "#F8F9FA", BLUE])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax, z_df, algo in zip(axes, [km_z, gmm_z], ["K-Means", "GMM"]):
        im = ax.imshow(z_df.values, cmap=cmap_div, vmin=-2.2, vmax=2.2, aspect="auto")
        ax.set_xticks(range(z_df.shape[1]))
        ax.set_xticklabels([f"C{c}" for c in z_df.columns], fontsize=11, fontweight="bold")
        ax.set_yticks(range(z_df.shape[0]))
        ax.set_yticklabels(z_df.index, fontsize=10)
        ax.set_title(f"{algo}: feature profile per cluster\n(z-scored across clusters)",
                     fontsize=12)
        ax.set_xlabel("Cluster")

        # Annotate each cell with the z-value (to 1 dp) if it's visually
        # striking — otherwise it clutters.
        for i in range(z_df.shape[0]):
            for j in range(z_df.shape[1]):
                v = z_df.values[i, j]
                if abs(v) >= 0.8:
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            fontsize=8, color="white" if abs(v) > 1.4 else DARK_TEXT,
                            fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.025, pad=0.02)
    cbar.set_label("Feature value (z-score across clusters)", rotation=270, labelpad=15)

    fig.suptitle("Archetype profiles: which features define each cluster",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.00)
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


def fig_h_shap_per_cluster(out_path: Path, shap_data: dict, k: int):
    """
    For each cluster, show top-8 features by mean |SHAP|. One subplot
    per cluster, arranged in a grid. This directly answers "which
    features does the model rely on for this type of player?"
    """
    if shap_data is None:
        log.info("  Skipping fig_h (no SHAP data)")
        return

    feat_names = shap_data["feature_names"]
    per_cluster = shap_data["per_cluster"]

    n_cols = min(k, 4)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 4.5),
                             squeeze=False)

    for c in range(k):
        ax = axes[c // n_cols][c % n_cols]
        info = per_cluster.get(str(c), {})
        if info.get("n_test_rows", 0) == 0:
            ax.text(0.5, 0.5, f"Cluster {c}\nno test players",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=GREY)
            ax.axis("off")
            continue

        shap_dict = info["mean_abs_shap"]
        sorted_feats = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = [f for f, _ in sorted_feats][::-1]
        values = [v for _, v in sorted_feats][::-1]

        colour = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
        bars = ax.barh(labels, values, color=colour, alpha=0.85,
                       edgecolor=DARK_TEXT, linewidth=0.5)
        ax.set_title(f"Cluster {c}   (n={info['n_test_rows']} test players)",
                     fontsize=11, color=DARK_TEXT)
        ax.set_xlabel("mean |SHAP| (max over classes)", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3, linestyle=":")

        for bar, v in zip(bars, values):
            ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", ha="left", va="center",
                    fontsize=8, color=DARK_TEXT)

    # Hide unused axes
    for c in range(k, n_rows * n_cols):
        axes[c // n_cols][c % n_cols].axis("off")

    fig.suptitle("SHAP feature importance by archetype\n"
                 "(which features the model relies on, broken down by player type)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info(f"  wrote {out_path}")


# ------------------------------------------------------------------ main

def main():
    log.info("=" * 60)
    log.info("Stage 6: player archetype clustering")
    log.info("=" * 60)

    df, X, X_raw = load_data()

    # 1. Pick k
    k_results = sweep_k(X)
    k = 5  # chosen_k from sweep_k

    # 2. Fit final models at chosen k
    km, km_labels, gmm, gmm_labels, gmm_proba = fit_final(X, k)

    # 3. Profiles
    log.info("\nK-Means cluster profiles:")
    profiles_km  = cluster_profiles(df, X_raw, km_labels,  "kmeans")
    log.info("\nGMM cluster profiles:")
    profiles_gmm = cluster_profiles(df, X_raw, gmm_labels, "gmm")

    # 4. SHAP per cluster (using K-Means labels — we need one canonical
    # clustering for this; K-Means is simpler and the paired plots let
    # readers see how GMM compares).
    log.info("\nComputing SHAP per cluster (using K-Means labels)...")
    shap_data = shap_per_cluster(df, X_raw, km_labels, k)

    # 5. Save cluster assignments — per-row output so they can be
    # cross-referenced with other analyses downstream.
    out_km = df[["understat_player_id", "player", "team", "league", "season"]].copy()
    out_km["cluster_kmeans"] = km_labels
    out_km.to_csv(MODELS / "clusters_kmeans.csv", index=False)

    out_gmm = df[["understat_player_id", "player", "team", "league", "season"]].copy()
    out_gmm["cluster_gmm"] = gmm_labels
    for i in range(k):
        out_gmm[f"gmm_proba_c{i}"] = gmm_proba[:, i]
    out_gmm.to_csv(MODELS / "clusters_gmm.csv", index=False)

    # 6. Save analyses as JSON
    with open(MODELS / "cluster_k_selection.json", "w") as f:
        json.dump(k_results, f, indent=2)
    with open(MODELS / "cluster_profiles.json", "w") as f:
        json.dump({"kmeans": profiles_km, "gmm": profiles_gmm}, f, indent=2,
                  default=str)
    if shap_data is not None:
        with open(MODELS / "shap_per_cluster.json", "w") as f:
            json.dump(shap_data, f, indent=2)

    # 7. Figures
    log.info("\nWriting figures ...")
    fig_e_k_selection(FIGS / "fig_e_cluster_selection.png", k_results)
    fig_f_cluster_viz(FIGS / "fig_f_cluster_viz.png", X, km_labels, gmm_labels, k)
    fig_g_cluster_profiles(FIGS / "fig_g_cluster_profiles.png",
                           profiles_km, profiles_gmm)
    fig_h_shap_per_cluster(FIGS / "fig_h_shap_per_cluster.png", shap_data, k)

    # 8. Summary
    log.info("=" * 60)
    log.info(f"Done. Chosen k = {k}")
    log.info(f"  K-Means cluster sizes: "
             f"{dict(zip(*np.unique(km_labels, return_counts=True)))}")
    log.info(f"  GMM     cluster sizes: "
             f"{dict(zip(*np.unique(gmm_labels, return_counts=True)))}")
    log.info("")
    log.info("Next step: read cluster_profiles.json and name the archetypes")
    log.info("based on which features dominate each cluster (e.g. 'Poacher',")
    log.info("'Creative 10', 'Target forward', 'Inverted winger', etc.).")
    log.info("=" * 60)


if __name__ == "__main__":
    main()