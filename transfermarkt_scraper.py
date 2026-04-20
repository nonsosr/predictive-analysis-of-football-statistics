"""
Transfermarkt Premier League Multi-Season Stats Scraper (2021/22 - 2024/25)
Positions: CF (14), LW (11), RW (12), CM (10), AM (7)
Filter: 900+ minutes played
Outputs: premier_league_yyyy_yy.csv files
"""

import time
import random
import logging
import argparse
import re

import pandas as pd
from bs4 import BeautifulSoup
import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Detail view URL formatted to accept season and position
BASE_URL = (
    "https://www.transfermarkt.com/premier-league/torschuetzenliste"
    "/wettbewerb/GB1/saison_id/{season}/altersklasse/alle"
    "/detailpos/{pos}/plus/1/page/{page}"
)

# Added URL for dynamically fetching league tables per season
TABLE_URL = "https://www.transfermarkt.com/premier-league/tabelle/wettbewerb/GB1/saison_id/{season}"

SEASONS = [2021, 2022, 2023, 2024]

POSITIONS = {
    "CF": 14,
    "RW": 12,
    "LW": 11,
    "CM": 10,
    "AM":  7,
}

MIN_MINUTES = 900

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
}

CLUB_ALIASES = {
    "Man City":         "Manchester City",
    "Man Utd":          "Manchester United",
    "Man. United":      "Manchester United",
    "West Ham":         "West Ham United",
    "Brighton":         "Brighton & Hove Albion",
    "Wolves":           "Wolverhampton Wanderers",
    "Newcastle":        "Newcastle United",
    "Newcastle Utd":    "Newcastle United",
    "Spurs":            "Tottenham Hotspur",
    "Tottenham":        "Tottenham Hotspur",
    "Leicester":        "Leicester City",
    "Leeds":            "Leeds United",
    "Aston Villa":      "Aston Villa",
    "Brentford FC":     "Brentford",
    "Nottm Forest":     "Nottingham Forest",
    "Sheff Utd":        "Sheffield United",
    "Luton Town":       "Luton",
    "Bournemouth":      "AFC Bournemouth",
    "Ipswich":          "Ipswich Town",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_page(session: requests.Session, url: str, retries: int = 4):
    """Fetch URL with exponential back-off. Returns BeautifulSoup or None."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=25)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            if resp.status_code == 429:
                wait = 30 * attempt
                log.warning("Rate-limited (429). Waiting %ds ...", wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %d on attempt %d — %s", resp.status_code, attempt, url)
                time.sleep(5 * attempt)
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(5 * attempt)
    log.error("Failed after %d attempts: %s", retries, url)
    return None

def normalize_club_name(club: str) -> str:
    cleaned = club.strip()
    return CLUB_ALIASES.get(cleaned, cleaned)

def fetch_league_table(session: requests.Session, season: int) -> dict:
    """Dynamically fetches the final league table for a given season."""
    log.info("Fetching league standings for %d...", season)
    url = TABLE_URL.format(season=season)
    soup = fetch_page(session, url)
    standings = {}
    
    if not soup:
        log.warning("Failed to fetch league table for %d", season)
        return standings

    table = soup.select_one("table.items tbody")
    if not table:
        log.warning("No standings table found for %d", season)
        return standings

    for tr in table.select("tr"):
        cells = tr.select("td")
        if len(cells) < 3:
            continue
        
        pos_text = cells[0].get_text(strip=True)
        club_link = tr.select_one("td.hauptlink a")
        
        if pos_text.isdigit() and club_link:
            club_name = normalize_club_name(club_link.get_text(strip=True))
            standings[club_name] = int(pos_text)
            
    return standings


def parse_minutes(raw: str) -> int | None:
    cleaned = raw.strip().rstrip("'").replace(",", "").replace(".", "").replace(" ", "")
    return int(cleaned) if cleaned.isdigit() else None


def count_pages(soup) -> int:
    """Read the TM pagination widget to find the last page number."""
    pages = []
    for a in soup.select("li.tm-pagination__list-item a, ul.tm-pagination a"):
        txt = a.get_text(strip=True)
        if txt.isdigit():
            pages.append(int(txt))
    return max(pages) if pages else 1


def parse_table(soup, position_label: str, pl_table: dict) -> list[dict]:
    """Parse one page of the detailed scorer list."""
    table = soup.select_one("table.items")
    if not table:
        log.warning("No table.items found on page.")
        return []

    rows_data: list[dict] = []

    for tr in table.select("tbody tr.odd, tbody tr.even"):
        cells = tr.select("td")
        if len(cells) < 8:
            continue

        try:
            raw_texts = [c.get_text(strip=True) for c in cells]

            # ── Player name + id ──────────────────────────────────────────
            name_cell = cells[1]
            link = (
                name_cell.select_one("a.spielprofil_tooltip")
                or name_cell.select_one("td.hauptlink a")
                or name_cell.select_one("a")
            )
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link.get("href", "")
            player_id = href.rstrip("/").split("/")[-1]
            if not player_id.isdigit():
                player_id = None

            # ── Age / club extraction ─────────────────────────────────────
            age = None
            club = ""
            if len(cells) > 6:
                raw_age = cells[6].get_text(strip=True)
                age_match = re.search(r"(\d+)", raw_age)
                if age_match:
                    age = int(age_match.group(1))
            if len(cells) > 7:
                club_cell = cells[7]
                club_link = club_cell.select_one("a")
                if club_link:
                    club = club_link.get("title", club_link.get_text(strip=True)).strip()
                else:
                    club = club_cell.get_text(strip=True)

            # ── Numeric stat cells ─────────────────────────────────────────
            stat_cells = cells[-7:]

            def parse_int(raw: str) -> int | None:
                value = raw.strip().replace("'", "").replace(",", "").replace(".", "")
                return int(value) if value.isdigit() else None

            appearances = parse_int(stat_cells[0].get_text(strip=True))
            assists     = parse_int(stat_cells[1].get_text(strip=True))
            pens_taken  = parse_int(stat_cells[2].get_text(strip=True))
            minutes     = parse_minutes(stat_cells[3].get_text(strip=True))
            goals       = parse_int(stat_cells[6].get_text(strip=True))
            subs_on     = None

            pens_scored = None
            # TM shows penalty count as a single numeric cell in this view, not x/y.
            for txt in raw_texts:
                m = re.match(r'^(\d+)/(\d+)$', txt)
                if m:
                    pens_scored = int(m.group(1))
                    pens_taken  = int(m.group(2))
                    break

            # Derived
            starts = (appearances - subs_on) if (appearances is not None and subs_on is not None) else None
            ga_per90 = None
            if goals is not None and assists is not None and minutes and minutes > 0:
                ga_per90 = round((goals + assists) / minutes * 90, 2)

            normalized_club = normalize_club_name(club)
            club_pos = pl_table.get(normalized_club)

            rows_data.append({
                "player_id":    player_id,
                "name":         name,
                "age":          age,
                "position":     position_label,
                "minutes":      minutes,
                "appearances":  appearances,
                "starts":       starts,
                "subs":         subs_on,
                "goals":        goals,
                "assists":      assists,
                "g+a per 90":   ga_per90,
                "shots/90":                          None,
                "shots on target/90":                None,
                "pens taken":   pens_taken,
                "pens scored":  pens_scored,
                "matches missed due to injury":      None,
                "club position at the end of the league": club_pos,
            })

        except Exception as exc:
            log.debug("Skipping row due to error: %s", exc, exc_info=True)
            continue

    return rows_data


# ── Position scraper ──────────────────────────────────────────────────────────

def scrape_position(session, season: int, pos_label: str, pos_id: int, pl_table: dict) -> list[dict]:
    log.info("=== Scraping %s (detailpos=%d) for Season %d ===", pos_label, pos_id, season)

    first_url = BASE_URL.format(season=season, pos=pos_id, page=1)
    soup = fetch_page(session, first_url)
    if not soup:
        return []

    total_pages = count_pages(soup)
    log.info("  %d page(s) found for %s", total_pages, pos_label)

    all_rows: list[dict] = []

    for page in range(1, total_pages + 1):
        log.info("  Page %d / %d ...", page, total_pages)
        page_soup = soup if page == 1 else fetch_page(
            session, BASE_URL.format(season=season, pos=pos_id, page=page)
        )
        if not page_soup:
            log.warning("  Skipping page %d (fetch failed)", page)
            continue

        rows = parse_table(page_soup, pos_label, pl_table)
        log.info("    -> %d players parsed on this page", len(rows))
        all_rows.extend(rows)

        if page < total_pages:
            time.sleep(random.uniform(2.5, 5.0))

    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_all(seasons: list[int], min_minutes: int = MIN_MINUTES):
    session = get_session()

    for season in seasons:
        # Format the year for the CSV (e.g., 2021_22)
        season_str = f"{season}_{str(season + 1)[-2:]}"
        output_file = f"premier_league_{season_str}.csv"
        
        log.info("=" * 50)
        log.info("STARTING SCRAPE FOR SEASON %s", season_str)
        log.info("=" * 50)

        # 1. Fetch dynamic league table for this season
        pl_table = fetch_league_table(session, season)
        log.info("Fetched %d clubs for the %s season standings.", len(pl_table), season_str)

        all_players: list[dict] = []

        # 2. Iterate through positions
        for pos_label, pos_id in POSITIONS.items():
            rows = scrape_position(session, season, pos_label, pos_id, pl_table)
            all_players.extend(rows)
            log.info("Running total: %d players scraped so far for %s", len(all_players), season_str)
            time.sleep(random.uniform(4.0, 8.0))

        if not all_players:
            log.error("No data scraped for %s. Check connection or TM blocking.", season_str)
            continue

        df = pd.DataFrame(all_players)

        # 3. Filter and Deduplicate
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
        before = len(df)
        df = df[df["minutes"] >= min_minutes].copy()
        log.info("After %d+ min filter: %d / %d players kept for %s", min_minutes, len(df), before, season_str)

        df = df.drop_duplicates(subset=["player_id", "position"])
        df = df.sort_values(["position", "goals"], ascending=[True, False])

        # 4. Final column order and export
        col_order = [
            "player_id", "name", "age", "position", "minutes",
            "appearances", "starts", "subs", "goals", "assists",
            "g+a per 90", "shots/90", "shots on target/90",
            "pens taken", "pens scored",
            "matches missed due to injury",
            "club position at the end of the league",
        ]
        df = df.reindex(columns=col_order)

        df.to_csv(output_file, index=False)
        log.info("Saved %d players -> %s", len(df), output_file)
        
    log.info("All selected seasons completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Transfermarkt PL detailed stats across multiple seasons.")
    parser.add_argument("--seasons",     nargs="+", type=int, default=SEASONS, help="List of start years (e.g., 2021 2022 2023)")
    parser.add_argument("--min-minutes", type=int, default=MIN_MINUTES)
    parser.add_argument("--debug",       action="store_true", help="Print cell-level debug info")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    scrape_all(seasons=args.seasons, min_minutes=args.min_minutes)