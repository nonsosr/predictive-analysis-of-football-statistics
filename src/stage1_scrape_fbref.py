"""
Stage 1: Raw data scraper for Big Five football leagues.

Pulls player-season and team-season stats from FBref via the `soccerdata`
library, for seasons 2017-18 through 2024-25. Saves each stat table as a
separate parquet file so downstream stages can merge them cleanly.

Run from your project root:
    python stage1_scrape.py

Re-running is safe: already-scraped files are skipped. soccerdata also
caches raw HTML in ~/soccerdata/, so even the scraped-but-not-saved case
(e.g. if you kill the process) won't re-fetch from FBref.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import soccerdata as sd


# ------------------------------------------------------------------ config

# Using FBref's combined Big-Five page is ~5x faster than looping over
# leagues individually: one HTTP request per stat type returns all five
# leagues at once. The returned DataFrame still has a 'league' column,
# so downstream filtering/grouping is unchanged.
LEAGUES = "Big 5 European Leagues Combined"

# soccerdata season format: start year + end year, both 2 digits.
# "1718" = 2017-18, "2425" = 2024-25.
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]

# Player-level stat types. soccerdata's read_player_season_stats accepts
# only these five values (confirmed from the library's validator):
#   standard, keeper, shooting, playing_time, misc
# We skip 'keeper' (irrelevant for outfield goal contribution) and
# 'playing_time' (its info is a subset of 'standard' for our purposes).
#
# NOTE for the dissertation writeup: player-level possession (box touches,
# progressive carries) and passing (key passes, progressive passes) tables
# exist on FBref but are not exposed by soccerdata. They're reachable via
# worldfootballR (R) or a direct FBref scrape. Flagged as next-stage work.
PLAYER_STAT_TYPES = [
    "standard",   # minutes, goals, assists, AND (post-upgrade) xG/npxG/xAG
    "shooting",   # shots, shots on target, G/Sh, xG/shot (post-upgrade)
    "misc",       # crosses, fouls drawn, offsides, interceptions, tackles won,
                  # (post-upgrade) aerial duels won/lost/%
]

# Team-level stat types. read_team_season_stats supports many more types
# than the player equivalent. Team-level passing and possession give us
# contextual features the player tables can't (team total progressive
# passes, team total final-third entries, team possession %).
TEAM_STAT_TYPES = [
    "standard",    # team xG, goals, possession%
    "shooting",    # team total shots, xG per shot
    "passing",     # team progressive passes, key passes
    "possession",  # team touches by zone, progressive carries
]

OUT_DIR = Path("./data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("scraper")


# ------------------------------------------------------------------ helpers

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FBref returns MultiIndex columns like ('Performance', 'Gls').
    Flatten them to single strings like 'Performance_Gls' so downstream
    joins are simpler and the CSV header is readable.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(level) for level in col if str(level) != "").strip("_")
            for col in df.columns
        ]
    return df


def safe_save(df: pd.DataFrame, path: Path) -> None:
    """Save to CSV. index=False so we don't get a stray 'Unnamed: 0' column."""
    df.to_csv(path, index=False)


# ------------------------------------------------------------------ scraping

def build_client() -> sd.FBref:
    """
    One client, reused across stat types. soccerdata will cache HTML on
    disk the first time it hits a page, so re-runs are fast.
    """
    return sd.FBref(leagues=LEAGUES, seasons=SEASONS)


def scrape_players(fbref: sd.FBref) -> None:
    for stat_type in PLAYER_STAT_TYPES:
        out_path = OUT_DIR / f"players_{stat_type}.csv"
        if out_path.exists():
            log.info(f"SKIP players/{stat_type}: {out_path} exists")
            continue

        log.info(f"Scraping players/{stat_type} ...")
        t0 = time.time()
        try:
            df = fbref.read_player_season_stats(stat_type=stat_type)
            df = flatten_columns(df.reset_index())
            safe_save(df, out_path)
            log.info(
                f"  OK players/{stat_type}: {len(df)} rows, "
                f"{len(df.columns)} cols, {time.time()-t0:.1f}s"
            )
        except Exception as e:
            log.error(f"  FAIL players/{stat_type}: {e}")


def scrape_teams(fbref: sd.FBref) -> None:
    for stat_type in TEAM_STAT_TYPES:
        out_path = OUT_DIR / f"teams_{stat_type}.csv"
        if out_path.exists():
            log.info(f"SKIP teams/{stat_type}: {out_path} exists")
            continue

        log.info(f"Scraping teams/{stat_type} ...")
        t0 = time.time()
        try:
            df = fbref.read_team_season_stats(stat_type=stat_type)
            df = flatten_columns(df.reset_index())
            safe_save(df, out_path)
            log.info(
                f"  OK teams/{stat_type}: {len(df)} rows, "
                f"{time.time()-t0:.1f}s"
            )
        except Exception as e:
            log.error(f"  FAIL teams/{stat_type}: {e}")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    start = time.time()
    log.info("=" * 60)
    log.info(f"Stage 1 scrape starting")
    log.info(f"  leagues: {LEAGUES}")
    log.info(f"  seasons: {SEASONS}")
    log.info(f"  output:  {OUT_DIR.resolve()}")
    log.info("=" * 60)

    client = build_client()
    scrape_players(client)
    scrape_teams(client)

    log.info("=" * 60)
    log.info(f"Done in {(time.time() - start) / 60:.1f} min")
    log.info("=" * 60)