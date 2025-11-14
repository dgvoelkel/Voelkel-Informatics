import sys
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, shotchartdetail, commonplayerinfo
from nba_api.stats.library.parameters import SeasonType
import unicodedata
from datetime import datetime

def normalize_string(s):
    if not isinstance(s, str):
        return s
    normalized = ''.join(
        c for c in unicodedata.normalize('NFKD', s)
        if not unicodedata.combining(c)
    ).lower()
    return normalized

def get_player_id(player_name):
    player_dict = players.get_players()
    normalized_input = normalize_string(player_name)
    matched_players = []
    for player in player_dict:
        normalized_player_name = normalize_string(player['full_name'])
        if normalized_player_name == normalized_input:
            matched_players.append(player)
    
    if len(matched_players) == 1:
        return matched_players[0]['id']
    elif len(matched_players) > 1:
        print(f"Multiple players found with the name '{player_name}'. Please specify:")
        for p in matched_players:
            print(f" - {p['full_name']} (ID: {p['id']})")
        sys.exit(1)
    else:
        print(f"Player '{player_name}' not found. Please check the name and try again.")
        sys.exit(1)

def get_player_position(player_id):
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    info = player_info.get_data_frames()[0]
    position = info['POSITION'].values[0]
    return position

def get_all_seasons_for_player(player_id):
    """
    Returns a sorted list of all seasons in which the player has data.
    Example return: ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20']
    """
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    stats = career.get_data_frames()[0]
    stats = stats[stats['LEAGUE_ID'] == '00']  # Filter out playoff or G-League stats
    
    # Convert 'SEASON_ID' from '1996-97' format, ensuring it’s sorted chronologically
    seasons = sorted(stats['SEASON_ID'].unique())
    return seasons

def get_last_season(player_id):
    """
    Retrieves the last season the player played with available data.
    """
    seasons = get_all_seasons_for_player(player_id)
    if not seasons:
        print("No available seasons with data.")
        sys.exit(1)
    return seasons[-1]  # Return the last (most recent) season

def get_player_stats(player_id, season_id):
    """
    Retrieves the player's per-game stats for a specific season.
    """
    career = playercareerstats.PlayerCareerStats(player_id=player_id, per_mode36='PerGame')
    df = career.get_data_frames()[0]
    df = df[(df['LEAGUE_ID'] == '00') & (df['SEASON_ID'] == season_id)]
    if df.empty:
        print(f"No data found for season {season_id}.")
        sys.exit(1)

    # If multiple rows exist (rare for trades mid-season), pick the sum or last row
    # Here we pick the last row
    last_season = df.iloc[-1]
    shooting_stats = {
        'Points': last_season['PTS'],
        'FGA': last_season['FGA'],
        'FGM': last_season['FGM'],
        'FG%': last_season['FG_PCT'],
        '3PA': last_season['FG3A'],
        '3PM': last_season['FG3M'],
        '3P%': last_season['FG3_PCT'],
        'FTA': last_season['FTA'],
        'FTM': last_season['FTM'],
        'FT%': last_season['FT_PCT']
    }
    return shooting_stats

def get_shot_zone_fg_percentages(player_id, season_id):
    """
    Calculates FG% for 10-16 ft and 16-24 ft for a given season.
    """
    shotchart = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_type_all_star=SeasonType.regular,
        season_nullable=season_id,
        context_measure_simple='FGA'
    )
    shot_data = shotchart.get_data_frames()[0]
    if shot_data.empty:
        print(f"No shot data available for player ID {player_id} in season {season_id}.")
        return {
            '10-16 FG%': 'N/A',
            '16-24 FG%': 'N/A'
        }

    ranges = {
        '10-16 FG%': (10, 16),
        '16-24 FG%': (16, 24)
    }

    fg_percentages = {}
    for stat_label, (min_dist, max_dist) in ranges.items():
        zone_shots = shot_data[(shot_data['SHOT_DISTANCE'] >= min_dist) & (shot_data['SHOT_DISTANCE'] < max_dist)]
        attempts = len(zone_shots)
        made = zone_shots['SHOT_MADE_FLAG'].sum()
        if attempts > 0:
            fg_pct = made / attempts
            fg_percentages[stat_label] = fg_pct
        else:
            fg_percentages[stat_label] = 'N/A'

    return fg_percentages

def true_shooting_percentage(points_scored, fga, fta):
    denominator = 2 * (fga + 0.44 * fta)
    if denominator == 0:
        return 0.0
    ts_percentage = points_scored / denominator
    return ts_percentage

def effective_field_goal_percentage(fgm, fga, fg3m):
    if fga == 0:
        return 0.0
    efg_percentage = (fgm + 0.5 * fg3m) / fga
    return efg_percentage

def points_per_shot_attempt(points_scored, fga):
    if fga == 0:
        return 0.0
    pps = points_scored / fga
    return pps

def three_point_attempt_rate(fg3a, fga):
    if fga == 0:
        return 0.0
    three_par = fg3a / fga
    return three_par

def main():
    """
    Usage:
        python basic_stat_grab.py "Player Name" [Season]
    """
    if len(sys.argv) < 2:
        print("Usage: python basic_stat_grab.py 'Player Name' [Season]")
        sys.exit(1)
    
    player_name = sys.argv[1]

    # If a season was provided, use it; otherwise get the last season
    if len(sys.argv) == 3:
        season_id = sys.argv[2]
    else:
        player_id = get_player_id(player_name)
        season_id = get_last_season(player_id)

    player_id = get_player_id(player_name)
    position = get_player_position(player_id)
    
    # Now get stats for the chosen season
    stats = get_player_stats(player_id, season_id)

    ts_pct = true_shooting_percentage(stats['Points'], stats['FGA'], stats['FTA'])
    efg_pct = effective_field_goal_percentage(stats['FGM'], stats['FGA'], stats['3PM'])
    pps = points_per_shot_attempt(stats['Points'], stats['FGA'])
    three_par = three_point_attempt_rate(stats['3PA'], stats['FGA'])
    
    zone_fg_percentages = get_shot_zone_fg_percentages(player_id, season_id)
    stats.update(zone_fg_percentages)

    if stats['10-16 FG%'] == 'N/A':
        fg_pct_10_16 = 'N/A'
    else:
        fg_pct_10_16 = f"{stats['10-16 FG%'] * 100:.1f}%"

    if stats['16-24 FG%'] == 'N/A':
        fg_pct_16_24 = 'N/A'
    else:
        fg_pct_16_24 = f"{stats['16-24 FG%'] * 100:.1f}%"

    print(f"Shooting stats for {player_name} in season {season_id}:")
    print(f"Position: {position}")
    print(f"Points per Game: {stats['Points']:.1f}")
    print(f"FGA per Game: {stats['FGA']:.1f}")
    print(f"FG%: {stats['FG%'] * 100:.1f}%")
    print(f"3PA per Game: {stats['3PA']:.1f}")
    print(f"3P%: {stats['3P%'] * 100:.1f}%")
    print(f"FTA per Game: {stats['FTA']:.1f}")
    print(f"FT%: {stats['FT%'] * 100:.1f}%")
    print(f"10-16 FG%: {fg_pct_10_16}")
    print(f"16-24 FG%: {fg_pct_16_24}\n")

    print("Advanced Metrics:")
    print(f"True Shooting Percentage (TS%): {ts_pct * 100:.1f}%")
    print(f"Effective Field Goal Percentage (eFG%): {efg_pct * 100:.1f}%")
    print(f"Points Per Shot Attempt (PPS): {pps:.3f}")
    print(f"Three-Point Attempt Rate (3PAr): {three_par * 100:.1f}%")

if __name__ == "__main__":
    main()
