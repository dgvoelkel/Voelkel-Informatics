import time
import random
from pathlib import Path
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamelog

LEAGUE_ID = "00"
SLEEP_BETWEEN_CALLS = (0.8, 1.6)


def get_active_player_ids() -> set[int]:
    active = players.get_active_players()
    return {p["id"] for p in active}


def fetch_league_player_gamelog(season: str, season_type: str) -> pd.DataFrame:
    max_tries = 6
    last_err = None

    for attempt in range(1, max_tries + 1):
        try:
            time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))

            lg = leaguegamelog.LeagueGameLog(
                league_id=LEAGUE_ID,
                season=season,
                season_type_all_star=season_type,   # "Regular Season" or "Playoffs"
                player_or_team_abbreviation="P"     # player rows
            )
            return lg.get_data_frames()[0]

        except Exception as e:
            last_err = e
            wait = min((2 ** attempt) + random.random(), 30)
            print(
                f"[WARN] {season} {season_type}: attempt {attempt}/{max_tries} "
                f"failed ({type(e).__name__}). Sleeping {wait:.1f}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed LeagueGameLog for {season} {season_type}: {last_err}")


def build_active_player_gamelog_csv(
    season: str,
    out_dir: str = "nba_player_gamelogs_by_season"
) -> Path:
    """
    Build one CSV for a season containing all regular season + playoff
    game logs for active players only.

    Returns:
        Path to the saved CSV file.
    """
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    active_ids = get_active_player_ids()
    print(f"Active players found: {len(active_ids)}")

    reg = fetch_league_player_gamelog(season, "Regular Season")
    reg = reg[reg["PLAYER_ID"].isin(active_ids)].copy()
    reg["SEASON"] = season
    reg["SEASON_TYPE"] = "Regular Season"

    po = fetch_league_player_gamelog(season, "Playoffs")
    po = po[po["PLAYER_ID"].isin(active_ids)].copy()
    po["SEASON"] = season
    po["SEASON_TYPE"] = "Playoffs"

    combined = pd.concat([reg, po], ignore_index=True)

    out_path = out_path_dir / f"game_stats_{season}.csv"
    combined.to_csv(out_path, index=False)

    print(
        f"[OK] Saved {season}: {out_path} | rows={len(combined):,} "
        f"(reg={len(reg):,}, po={len(po):,})"
    )

    return out_path


if __name__ == "__main__":
    build_active_player_gamelog_csv("2025-26")