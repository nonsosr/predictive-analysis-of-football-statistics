"""
Stage 1b: Understat scraper.

soccerdata's FBref scraper drops the 'Expected' column group from player
tables, so we can't get xG/npxG/xA from FBref through that library.
Understat is the alternative: it's the original public xG data source
and soccerdata wraps it cleanly via sd.Understat.

This script pulls per-player per-season stats for the same 5 leagues
and 8 seasons as the FBref scrape. Understat coverage starts in 2014-15
so every season we want is covered.

Run from project root:
    python stage1b_understat.py

Output: ./data/raw/understat_players.csv
"""

import logging
import time
from pathlib import Path

import pandas as pd
import soccerdata as sd


# ------------------------------------------------------------------ config

# Understat uses the same league IDs as FBref in soccerdata, so we can
# match them up cleanly downstream.
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "ITA-Serie A",
]

# Understat does NOT have a "Big 5 Combined" endpoint — each league is
# scraped separately. Still fast because it's a smaller site than FBref.
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]

OUT_DIR = Path("./data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "understat_players.csv"


# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("understat")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    if OUT_PATH.exists():
        log.info(f"SKIP: {OUT_PATH} already exists. Delete it to re-scrape.")
        raise SystemExit(0)

    t0 = time.time()
    log.info("=" * 60)
    log.info("Understat scrape starting")
    log.info(f"  leagues: {LEAGUES}")
    log.info(f"  seasons: {SEASONS}")
    log.info("=" * 60)

    # One client handles all leagues/seasons. Understat's rate limits are
    # gentler than FBref's, so we don't need special delay handling.
    understat = sd.Understat(leagues=LEAGUES, seasons=SEASONS)

    log.info("Reading player season stats ...")
    df = understat.read_player_season_stats()

    # Understat's output uses (league, season, team, player) as a row
    # MultiIndex. Reset to proper columns so CSV is self-describing
    # and downstream merges with FBref data are straightforward.
    df = df.reset_index()

    df.to_csv(OUT_PATH, index=False)
    log.info(
        f"OK: {len(df)} rows, {len(df.columns)} cols, "
        f"{time.time() - t0:.1f}s -> {OUT_PATH}"
    )
    log.info(f"columns: {list(df.columns)}")