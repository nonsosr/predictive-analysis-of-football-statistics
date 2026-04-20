import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap
import warnings

warnings.filterwarnings('ignore')

# --- 1. POSTER STYLE CONFIGURATION ---
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT    = "#00c896"       # Teal
ACCENT2   = "#f7b731"       # Gold
ACCENT3   = "#e05c5c"       # Red
TEXT      = "#e6edf3"
MUTED     = "#8b949e"

ARCHETYPE_PALETTE = {
    "Elite Goal Threats":   ACCENT2,  # Gold
    "Regular Contributors": ACCENT,   # Teal
    "Fringe / Rotation":    ACCENT3   # Red
}

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    MUTED,
    "axes.labelcolor":   TEXT,
    "text.color":        TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        "#21262d",
    "grid.alpha":        0.7,
    "font.size":         12,
    "axes.titlesize":    16,
    "axes.labelsize":    14
})

# Feature name mapping for human-readable charts
FEATURE_MAP = {
    'current_g_plus_a': 'Current G+A',
    'shots_on_target_90': 'Shots on Target / 90',
    'shots_90': 'Shots / 90',
    'minutes': 'Minutes Played',
    'subs': 'Sub Appearances',
    'starts': 'Starts',
    'club_position': 'Club Position',
    'age': 'Age'
}

# --- 2. FILE PATHS (NUCLEAR OPTION) ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
FIGURES_DIR = os.path.join(desktop_path, "Dissertation_Charts")
os.makedirs(FIGURES_DIR, exist_ok=True)

# MAKE SURE THIS MATCHES YOUR RAW DATA FOLDER
RAW_DATA_DIR = r"C:\Users\nonso\Documents\Dissertation\data\raw"

# --- 3. DATA LOADING & CLEANING ---
def load_and_standardize(filename, season_start_year):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath, encoding='latin1')
    
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_').str.replace('+', '_plus_')
    df = df.rename(columns={'name': 'player_name', 'matches_played': 'appearances', 'club_position_at_the_end_of_the_league': 'club_position'})
    df['season'] = season_start_year
    
    if 'club_position' in df.columns:
        df['club_position'] = df['club_position'].astype(str).str.extract(r'(\d+)').astype(float)
    return df

print("Loading Data...")
df20 = load_and_standardize('premier_league_2020_21.csv', 2020)
df21 = load_and_standardize('premier_league_2021_22.csv', 2021)
df22 = load_and_standardize('premier_league_2022_23.csv', 2022)
df23 = load_and_standardize('premier_league_2023_24.csv', 2023)
df24 = load_and_standardize('premier_league_2024_25.csv', 2024)

def create_pairs(df_current, df_next):
    df_merged = pd.merge(df_current, df_next[['player_name', 'goals', 'assists']], on='player_name', suffixes=('', '_next'))
    df_merged['next_season_g_plus_a'] = df_merged['goals_next'] + df_merged['assists_next']
    return df_merged

df_all = pd.concat([create_pairs(df20, df21), create_pairs(df21, df22), create_pairs(df22, df23), create_pairs(df23, df24)], ignore_index=True)
df_all['current_g_plus_a'] = df_all['goals'] + df_all['assists']
df_all = df_all[df_all['position'].isin(['AM', 'LW', 'RW', 'ST', 'CF', 'FW'])]

features = list(FEATURE_MAP.keys())
df_model = df_all.dropna(subset=features + ['next_season_g_plus_a']).copy()

# --- 4. K-MEANS CLUSTERING ---
print("Running K-Means Clustering...")
cluster_features = ['shots_90', 'shots_on_target_90', 'current_g_plus_a', 'minutes']
X_scaled = StandardScaler().fit_transform(df_model[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42)
df_model['archetype'] = kmeans.fit_predict(X_scaled)

cluster_means = df_model.groupby('archetype')[cluster_features].mean()
sorted_idx = cluster_means.sort_values('current_g_plus_a').index
names = {sorted_idx[0]: "Fringe / Rotation", sorted_idx[1]: "Regular Contributors", sorted_idx[2]: "Elite Goal Threats"}
df_model['archetype_name'] = df_model['archetype'].map(names)

# --- CHART 1: ARCHETYPE BOXPLOT WITH JITTER & STATS ---
print("Generating Chart 1: Archetype Boxplot...")
plt.figure(figsize=(9, 6))
order = ["Elite Goal Threats", "Regular Contributors", "Fringe / Rotation"]

ax = sns.boxplot(x='archetype_name', y='next_season_g_plus_a', data=df_model, 
                 order=order, palette=ARCHETYPE_PALETTE, fliersize=0, boxprops=dict(alpha=0.7))
sns.stripplot(x='archetype_name', y='next_season_g_plus_a', data=df_model, 
              order=order, color='white', alpha=0.3, jitter=True, size=5)

for i, category in enumerate(order):
    subset = df_model[df_model['archetype_name'] == category]['next_season_g_plus_a']
    median_val = subset.median()
    n_count = subset.count()
    ax.text(i, ax.get_ylim()[1] * 0.95, f"Median: {median_val:.0f}\n(n={n_count})", 
            ha='center', va='top', color=TEXT, fontweight='bold', fontsize=11, 
            bbox=dict(facecolor=DARK_BG, edgecolor=MUTED, boxstyle='round,pad=0.3', alpha=0.8))

plt.title('Next Season Output Distribution by Archetype', pad=15)
plt.ylabel('Next Season Goal Contributions (G+A)')
plt.xlabel('')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, '01_archetype_boxplot.png'), bbox_inches='tight', dpi=300)
plt.close()

# --- 5. RANDOM FOREST REGRESSION ---
print("Training Random Forest...")
X = df_model[features]
y = df_model['next_season_g_plus_a']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Retrieve archetypes for test set to color code
test_archetypes = df_model.loc[X_test.index, 'archetype_name']

# --- CHART 2: ACTUAL VS PREDICTED WITH SHADED BAND ---
print("Generating Chart 2: Actual vs Predicted...")
plt.figure(figsize=(8, 8))
ax = sns.scatterplot(x=y_test, y=y_pred, hue=test_archetypes, palette=ARCHETYPE_PALETTE, 
                     s=80, alpha=0.8, edgecolor='w', linewidth=0.5)

# Shaded Tolerance Band (+/- 5)
max_val = max(y_test.max(), y_pred.max()) + 5
x_vals = np.linspace(0, max_val, 100)
plt.plot(x_vals, x_vals, color=TEXT, linestyle='--', alpha=0.5, label='Perfect Prediction')
plt.fill_between(x_vals, x_vals - 5, x_vals + 5, color=TEXT, alpha=0.05, label='±5 G+A Tolerance')

# Metrics Text Box
textstr = f"R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}"
props = dict(boxstyle='round,pad=0.5', facecolor=CARD_BG, edgecolor=MUTED, alpha=0.9)
plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='top', bbox=props, color=TEXT, fontweight='bold')

plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('Actual Next Season G+A', fontweight='bold')
plt.ylabel('Predicted Next Season G+A', fontweight='bold')
plt.title('Model Performance: Predicted vs Actual Outputs', pad=15)
plt.legend(loc='lower right', facecolor=CARD_BG, edgecolor=MUTED, labelcolor=TEXT)
plt.grid(True, linestyle='--', alpha=0.2)
plt.savefig(os.path.join(FIGURES_DIR, '02_actual_vs_predicted.png'), bbox_inches='tight', dpi=300)
plt.close()

# --- CHART 3: SHAP SUMMARY PLOT ---
print("Generating Chart 3: SHAP Summary Plot...")
X_test_readable = X_test.rename(columns=FEATURE_MAP)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
ax = plt.gca()
ax.set_facecolor(CARD_BG)

shap.summary_plot(shap_values, X_test_readable, show=False, color_bar=False)

ax.tick_params(colors=TEXT)
ax.xaxis.label.set_color(TEXT)
plt.title('SHAP Values: Impact of Metrics on Future Output', color=TEXT, fontsize=16, pad=20)
plt.savefig(os.path.join(FIGURES_DIR, '03_shap_summary.png'), bbox_inches='tight', dpi=300, facecolor=DARK_BG)
plt.close()

# --- CHART 4: FEATURE IMPORTANCE BAR CHART (NEW) ---
print("Generating Chart 4: Feature Importances...")
importances = rf.feature_importances_
indices = np.argsort(importances)
readable_features = [FEATURE_MAP[features[i]] for i in indices]

plt.figure(figsize=(10, 6))
ax = plt.gca()

# Color the top feature Teal, the rest Muted
colors = [ACCENT if i == len(indices)-1 else MUTED for i in range(len(indices))]

bars = plt.barh(range(len(indices)), importances[indices], color=colors, align='center', alpha=0.9)
plt.yticks(range(len(indices)), readable_features, color=TEXT, fontweight='bold')
plt.xlabel('Relative Importance (Gini)', fontweight='bold')
plt.title('Global Feature Importance Drivers', pad=15)

# Add data labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
             ha='left', va='center', color=TEXT, fontsize=10)

plt.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(FIGURES_DIR, '04_feature_importance.png'), bbox_inches='tight', dpi=300)
plt.close()

print("\n" + "="*50)
print(f"SUCCESS! All 4 styled charts have been saved to:")
print(FIGURES_DIR)
print("Drop these right into your poster.")
print("="*50 + "\n")