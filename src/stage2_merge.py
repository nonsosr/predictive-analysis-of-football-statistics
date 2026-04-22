"""
Stage 2: Clean, merge, feature-engineer.

Consumes six raw CSVs and emits three processed tables.

Inputs  (./data/raw/)
--------
  players_standard.csv    FBref: minutes, goals, assists, G+A/90 (already computed)
  players_shooting.csv    FBref: shots, shots on target, G/Sh
  players_misc.csv        FBref: crosses, fouls drawn, offsides, int, tkl won
  teams_standard.csv      FBref: team goals, possession %, team minutes
  teams_shooting.csv      FBref: team shots total
  understat_players.csv   Understat: xG, xA, npxG, key_passes, xGChain/Buildup

Outputs (./data/processed/)
---------
  player_seasons_all.csv        merged raw (no minutes filter) — for diagnostics
  player_seasons_eligible.csv   after 900-min filter, feature-engineered, tertile labelled
  predictive_dataset.csv        rows where we have BOTH season-t features AND season-(t+1) target


"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from unidecode import unidecode


# ------------------------------------------------------------------ config

RAW = Path("./data/raw")
OUT = Path("./data/processed")
OUT.mkdir(parents=True, exist_ok=True)

MIN_MINUTES = 900
TERTILE_LABELS = ["low", "mid", "high"]


# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("stage2")


# ------------------------------------------------------------------ helpers

def norm(s) -> str:
    """
    Normalize a name: strip accents, lowercase, collapse whitespace.
    Used for matching players across FBref and Understat, and teams too
    (Understat has 'Manchester United', FBref has 'Manchester Utd' — this
    won't fix every mismatch but handles accents cleanly).
    """
    if pd.isna(s):
        return ""
    return unidecode(str(s)).lower().strip()


def drop_index_col(df: pd.DataFrame) -> pd.DataFrame:
    """Older raw files saved with pandas index; remove if present."""
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


# ------------------------------------------------------------------ load & merge

def load_fbref_players() -> pd.DataFrame:
    """
    Merge the 3 FBref player tables into one. We rename the relevant
    columns to short, snake_case names and drop everything we won't use,
    so the downstream table is legible.

    Why inner-joining on (league, season, team, player): these four
    fields together uniquely identify a player's stint at a club in a
    season. They match exactly across FBref's own tables, so no fuzzy
    matching is needed here — only when we join Understat next.
    """
    std = drop_index_col(pd.read_csv(RAW / "players_standard.csv"))
    shoot = drop_index_col(pd.read_csv(RAW / "players_shooting.csv"))
    misc = drop_index_col(pd.read_csv(RAW / "players_misc.csv"))

    # Columns we want from STANDARD (minutes and goal contributions)
    std = std.rename(columns={
        "Playing Time_MP": "matches",
        "Playing Time_Starts": "starts",
        "Playing Time_Min": "minutes",
        "Playing Time_90s": "nineties",
        "Performance_Gls": "goals",
        "Performance_Ast": "assists",
        "Performance_G-PK": "np_goals",
        "Performance_PK": "pens_scored",
        "Performance_PKatt": "pens_taken",
        "Per 90 Minutes_G+A": "GA_p90",       # FBref precomputes this
        "Per 90 Minutes_G-PK": "np_goals_p90",
    })
    std_keep = [
        "league", "season", "team", "player", "nation", "pos", "age",
        "matches", "starts", "minutes", "nineties",
        "goals", "assists", "np_goals", "pens_scored", "pens_taken",
        "GA_p90", "np_goals_p90",
    ]
    std = std[std_keep]

    # Columns we want from SHOOTING (shot volume & conversion)
    shoot = shoot.rename(columns={
        "Standard_Sh": "shots",
        "Standard_SoT": "shots_on_target",
        "Standard_Sh/90": "shots_p90",
        "Standard_SoT/90": "shots_on_target_p90",
        "Standard_G/Sh": "goals_per_shot",
    })[[
        "league", "season", "team", "player",
        "shots", "shots_on_target", "shots_p90", "shots_on_target_p90",
        "goals_per_shot",
    ]]

    # Columns we want from MISC (physical / duels / creation proxies)
    misc = misc.rename(columns={
        "Performance_Fls": "fouls",
        "Performance_Fld": "fouls_drawn",
        "Performance_Off": "offsides",
        "Performance_Crs": "crosses",
        "Performance_Int": "interceptions",
        "Performance_TklW": "tackles_won",
    })[[
        "league", "season", "team", "player",
        "fouls", "fouls_drawn", "offsides", "crosses",
        "interceptions", "tackles_won",
    ]]

    df = (std
          .merge(shoot, on=["league", "season", "team", "player"], how="left")
          .merge(misc,  on=["league", "season", "team", "player"], how="left"))

    log.info(f"FBref merged: {len(df):,} rows, {df.shape[1]} cols")
    return df


def load_understat() -> pd.DataFrame:
    """
    Load Understat, keep only the columns that ADD information on top of
    FBref. We deliberately don't re-import Understat's goals/assists/shots
    (FBref is the source of truth for those) — but we keep them under
    prefixed names so we can validate the merge downstream.
    """
    df = pd.read_csv(RAW / "understat_players.csv")
    df = df.rename(columns={
        "xg": "xG",
        "xa": "xA",
        "np_xg": "npxG",
        "xg_chain": "xGChain",
        "xg_buildup": "xGBuildup",
        "player_id": "understat_player_id",
        # Keep these for post-merge sanity checks only
        "goals": "_ust_goals",
        "assists": "_ust_assists",
        "shots": "_ust_shots",
        "minutes": "_ust_minutes",
        "np_goals": "_ust_np_goals",
    })[[
        "league", "season", "team", "player", "understat_player_id",
        "xG", "xA", "npxG", "key_passes", "xGChain", "xGBuildup",
        "_ust_goals", "_ust_assists", "_ust_shots", "_ust_minutes",
    ]]
    log.info(f"Understat loaded: {len(df):,} rows, {df.shape[1]} cols")
    return df


def merge_fbref_understat(fb: pd.DataFrame, ust: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Understat onto FBref on (season, team_norm, player_norm).

    We use NORMALIZED (accents stripped, lowercased) versions of team and
    player because the two sources disagree on spelling:
        Understat "Müller"           -> FBref "Müller"       (match after unidecode)
        Understat "Manchester United" -> FBref "Manchester Utd" (still won't match)

    Team-name mismatches are the main source of merge failures; we log
    the rate and inspect in diagnostics. We do NOT merge on 'league'
    because (season + team + player) is already unique without it, and
    Understat's league naming is occasionally inconsistent.

    A player who transferred mid-season within the same league produces
    two rows on both sides, and each row matches to its own stint.
    """
    fb = fb.copy()
    ust = ust.copy()

    fb["_team_n"]   = fb["team"].apply(norm)
    fb["_player_n"] = fb["player"].apply(norm)
    ust["_team_n"]   = ust["team"].apply(norm)
    ust["_player_n"] = ust["player"].apply(norm)

    merged = fb.merge(
        ust.drop(columns=["league", "team", "player"]),
        on=["season", "_team_n", "_player_n"],
        how="left",
    )

    matched = merged["understat_player_id"].notna().sum()
    rate = matched / len(merged)
    log.info(
        f"FBref<>Understat merge: {matched:,}/{len(merged):,} matched "
        f"({rate:.1%})"
    )
    if rate < 0.80:
        log.warning(
            "Match rate <80% — likely team-name mismatches. "
            "Check diagnostics in the unmatched subset."
        )

    return merged.drop(columns=["_team_n", "_player_n"])


# ------------------------------------------------------------------ filter + features

def apply_min_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only player-stints with >= MIN_MINUTES. This is applied at the
    (player, team, season) level, so a player with 500+500 across two
    clubs gets filtered out. Aggregating mid-season transfers is a
    known limitation flagged for future work.
    """
    before = len(df)
    out = df[df["minutes"] >= MIN_MINUTES].copy()
    log.info(
        f"900-min filter: {before:,} -> {len(out):,} "
        f"({len(out)/before:.1%} retained)"
    )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build three groups of features:
      1. Per-90 rates for the stats that aren't already per-90.
      2. 'Holy grail' composite: npxG + xA per 90.
      3. Team-context: team attacking strength, share of team output,
         minutes percentage.
    """
    df = df.copy()
    # A per-90 conversion factor. A player with 1800 min has factor 20
    # (they played 20 "90-minute equivalents"), so dividing totals by it
    # gives per-90 rates.
    factor = df["minutes"] / 90

    # --- 1. Per-90 rates for counting stats Understat/FBref leave raw.
    for raw_col, p90_col in [
        ("xG",            "xG_p90"),
        ("npxG",          "npxG_p90"),
        ("xA",            "xA_p90"),
        ("key_passes",    "key_passes_p90"),
        ("xGChain",       "xGChain_p90"),
        ("xGBuildup",     "xGBuildup_p90"),
        ("crosses",       "crosses_p90"),
        ("fouls_drawn",   "fouls_drawn_p90"),
        ("offsides",      "offsides_p90"),
        ("interceptions", "interceptions_p90"),
        ("tackles_won",   "tackles_won_p90"),
    ]:
        df[p90_col] = df[raw_col] / factor

    # --- 2. npxG + xA per 90 — the core predictor in the methodology.
    df["npxG_plus_xA_p90"] = df["npxG_p90"] + df["xA_p90"]

    # Shot quality: expected goals per shot taken. High npxG/shot = plays
    # high-value chances; low = volume shooter from outside the box.
    # Uses Understat shots and all xG (penalty noise is minor at season
    # level); replace 0-shot divisor with NaN to avoid inf.
    df["xG_per_shot"] = df["xG"] / df["shots"].replace(0, np.nan)

    # --- 3. Team context.
    # Team standard has goals, possession %, and total team minutes.
    teams_std = drop_index_col(pd.read_csv(RAW / "teams_standard.csv"))
    teams_std = teams_std.rename(columns={
        "Performance_Gls": "team_goals",
        "Performance_G-PK": "team_np_goals",
        "Playing Time_Min": "team_minutes",
        "Poss": "team_possession",
    })[[
        "league", "season", "team",
        "team_goals", "team_np_goals", "team_minutes", "team_possession",
    ]]

    # Team shooting gives us total team shots.
    teams_shoot = drop_index_col(pd.read_csv(RAW / "teams_shooting.csv"))
    teams_shoot = teams_shoot.rename(columns={"Standard_Sh": "team_shots"})[
        ["league", "season", "team", "team_shots"]
    ]

    df = (df
          .merge(teams_std,  on=["league", "season", "team"], how="left")
          .merge(teams_shoot, on=["league", "season", "team"], how="left"))

    # Share of team output: what fraction of the team's attacking output
    # was this player responsible for. High values = talismanic; low =
    # system player. This directly supports the archetype-clustering
    # stage of the dissertation.
    df["share_team_goals"] = df["goals"] / df["team_goals"].replace(0, np.nan)
    df["share_team_shots"] = df["shots"] / df["team_shots"].replace(0, np.nan)

    # Minutes percentage: the share of total available match-minutes
    # this player was on the pitch for. Proxy for manager trust +
    # availability. team_minutes is ~3420 for a 38-game league.
    df["minutes_pct"] = df["minutes"] / df["team_minutes"]

    # Team attacking strength: team goals per 90 is a simple and
    # interpretable proxy. We DON'T have team xG (FBref team standard
    # didn't return it — same library limitation). This is the one
    # place the FBref xG gap actually hurts us; goals/90 is correlated
    # with team xG but noisier.
    df["team_goals_p90"] = df["team_goals"] / (df["team_minutes"] / 90)

    return df


# ------------------------------------------------------------------ target

def build_tertile_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each player-season a low/mid/high label by G+A/90,
    CALCULATED WITHIN their own league-season.

    Within-season tertiles handle the fact that different leagues and
    eras produce different scoring rates: a 0.55 G+A/90 might be
    upper-mid in 2017-18 Premier League but bottom-mid in 2022-23
    Serie A. Ranking inside the comparison group removes that bias.

    pd.qcut with duplicates='drop' handles the edge case where many
    players share the same G+A/90 (e.g. many zeros in defensive
    clusters). In that case fewer than 3 tertiles are produced and
    the label is NaN for players on the tie boundary — rare at 900-min
    filter, but we accept a small loss.
    """
    def _label(group):
        try:
            return pd.qcut(group, q=3, labels=TERTILE_LABELS, duplicates="drop")
        except ValueError:
            return pd.Series([np.nan] * len(group), index=group.index)

    df = df.copy()
    df["GA_p90_tertile"] = (
        df.groupby(["league", "season"], group_keys=False)["GA_p90"]
          .apply(_label)
    )
    return df


def build_predictive_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row (player, season_t) attach the tertile they had in
    season_(t+1). Rows without a next-season observation (retirement,
    transfer out of Big 5, injury, etc.) get NaN and are dropped.

    Matching across seasons uses Understat's stable player_id, which
    is far more reliable than name matching.
    """
    df = df.copy()

    # The season strings ("1718", "2425") happen to be lexicographically
    # ordered == chronologically ordered, so sorted() gives the correct
    # sequence. We build a t -> t+1 mapping.
    ordered = sorted(df["season"].astype(str).unique())
    nxt = {s: ordered[i + 1] for i, s in enumerate(ordered[:-1])}
    df["season_next"] = df["season"].astype(str).map(nxt)

    # Build a lookup: (player_id, season) -> their tertile that season.
    lookup = (df[["understat_player_id", "season", "GA_p90_tertile"]]
              .rename(columns={
                  "season": "_match_season",
                  "GA_p90_tertile": "target_tertile_next"})
              .dropna(subset=["understat_player_id", "target_tertile_next"]))
    # Cast _match_season to str so it matches season_next's dtype. The raw
    # 'season' column loaded from CSV is int64 (values are all-numeric like
    # "1718"), but season_next was built via a dict mapping and is str, so
    # pandas refuses to merge them without this cast.
    lookup["_match_season"] = lookup["_match_season"].astype(str)

    # Attach via merge rather than row-wise apply — much faster.
    out = df.merge(
        lookup,
        left_on=["understat_player_id", "season_next"],
        right_on=["understat_player_id", "_match_season"],
        how="left",
    ).drop(columns=["_match_season"])

    return out


# ------------------------------------------------------------------ main

def main() -> None:
    log.info("=" * 60)
    log.info("Stage 2: merge + clean + feature + target")
    log.info("=" * 60)

    # Load FBref + Understat, merge them
    fb = load_fbref_players()
    ust = load_understat()
    merged = merge_fbref_understat(fb, ust)
    merged.to_csv(OUT / "player_seasons_all.csv", index=False)
    log.info(f"Saved {OUT/'player_seasons_all.csv'} ({len(merged):,} rows)")

    # Filter, feature-engineer, label tertile
    eligible = apply_min_minutes(merged)
    eligible = engineer_features(eligible)
    eligible = build_tertile_target(eligible)
    eligible.to_csv(OUT / "player_seasons_eligible.csv", index=False)
    log.info(f"Saved {OUT/'player_seasons_eligible.csv'} ({len(eligible):,} rows)")

    # Predictive: current-season features + next-season target
    predictive = build_predictive_dataset(eligible)
    predictive_final = predictive[predictive["target_tertile_next"].notna()].copy()
    predictive_final.to_csv(OUT / "predictive_dataset.csv", index=False)
    log.info(
        f"Saved {OUT/'predictive_dataset.csv'} ({len(predictive_final):,} rows "
        f"with valid next-season target)"
    )

    # Summary
    log.info("=" * 60)
    log.info("ELIGIBLE SET tertile distribution:")
    for k, v in eligible["GA_p90_tertile"].value_counts().items():
        log.info(f"  {k}: {v:,}")
    log.info("")
    log.info("PREDICTIVE SET next-season target distribution:")
    for k, v in predictive_final["target_tertile_next"].value_counts().items():
        log.info(f"  {k}: {v:,}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()