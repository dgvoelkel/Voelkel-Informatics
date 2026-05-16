import sys
import time
import pandas as pd
import basic_stat_grab
from nba_api.stats.endpoints import leaguedashplayerstats, shotchartdetail
from nba_api.stats.library.parameters import SeasonType

def get_players_by_position_and_season(position, season_id):
    """
    Retrieves a DataFrame of player stats for the specified season and position.
    """
    position_mapping = {
        'Guard': 'G',
        'Forward': 'F',
        'Center': 'C'
    }

    player_position_code = None
    for key, value in position_mapping.items():
        if key in position:
            player_position_code = value
            break

    if not player_position_code:
        print(f"Could not determine position code for position: {position}")
        player_position_code = ''  # Empty string returns all positions

    try:
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_id,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            player_position_abbreviation_nullable=player_position_code
        ).get_data_frames()[0]
    except Exception as e:
        print(f"Error retrieving data for season {season_id}: {e}")
        return None

    if player_stats.empty:
        print(f"No data returned for season {season_id}.")
        return None

    return player_stats

def calculate_advanced_metrics(stats_df):
    stats_df['TS%'] = stats_df.apply(
        lambda row: basic_stat_grab.true_shooting_percentage(
            points_scored=row['PTS'], fga=row['FGA'], fta=row['FTA']
        ), axis=1
    )
    stats_df['eFG%'] = stats_df.apply(
        lambda row: basic_stat_grab.effective_field_goal_percentage(
            fgm=row['FGM'], fga=row['FGA'], fg3m=row['FG3M']
        ), axis=1
    )
    stats_df['PPS'] = stats_df.apply(
        lambda row: basic_stat_grab.points_per_shot_attempt(
            points_scored=row['PTS'], fga=row['FGA']
        ), axis=1
    )
    stats_df['3PAr'] = stats_df.apply(
        lambda row: basic_stat_grab.three_point_attempt_rate(
            fg3a=row['FG3A'], fga=row['FGA']
        ), axis=1
    )
    return stats_df

def get_shot_zone_fg_percentages(player_id, season_id):
    try:
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_type_all_star=SeasonType.regular,
            season_nullable=season_id,
            context_measure_simple='FGA'
        )
        shot_data = shotchart.get_data_frames()[0]
        if shot_data.empty:
            return {'10-16 FG%': None, '16-24 FG%': None}

        ranges = {
            '10-16 FG%': (10, 16),
            '16-24 FG%': (16, 24)
        }

        fg_percentages = {}
        for stat_label, (min_dist, max_dist) in ranges.items():
            zone_shots = shot_data[
                (shot_data['SHOT_DISTANCE'] >= min_dist) & (shot_data['SHOT_DISTANCE'] < max_dist)
            ]
            attempts = len(zone_shots)
            made = zone_shots['SHOT_MADE_FLAG'].sum()
            if attempts > 0:
                fg_pct = made / attempts
                fg_percentages[stat_label] = fg_pct
            else:
                fg_percentages[stat_label] = None

        return fg_percentages
    except Exception as e:
        print(f"Error fetching shot data for player ID {player_id}: {e}")
        return {'10-16 FG%': None, '16-24 FG%': None}

def collect_shot_zone_stats(stats_df, season_id):
    stats_df = stats_df.copy()
    stats_df['10-16 FG%'] = None
    stats_df['16-24 FG%'] = None

    total_players = len(stats_df)
    for idx, row in stats_df.iterrows():
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER_NAME']
        print(f"Processing player {idx + 1}/{total_players}: {player_name}")
        try:
            fg_percentages = get_shot_zone_fg_percentages(player_id, season_id)
            stats_df.at[idx, '10-16 FG%'] = fg_percentages['10-16 FG%']
            stats_df.at[idx, '16-24 FG%'] = fg_percentages['16-24 FG%']
        except Exception as e:
            print(f"Error processing player {player_name} (ID {player_id}): {e}")
            stats_df.at[idx, '10-16 FG%'] = None
            stats_df.at[idx, '16-24 FG%'] = None
        time.sleep(0.6)

    return stats_df

def main():
    """
    Usage:
        python zcalc.py "Player Name" [Season]
    """
    if len(sys.argv) < 2:
        print("Usage: python zcalc.py 'Player Name' [Season]")
        sys.exit(1)

    player_name = sys.argv[1]
    # If season is provided, use it, otherwise fallback to last season
    if len(sys.argv) == 3:
        season_id = sys.argv[2]
    else:
        player_id = basic_stat_grab.get_player_id(player_name)
        season_id = basic_stat_grab.get_last_season(player_id)

    player_id = basic_stat_grab.get_player_id(player_name)
    position = basic_stat_grab.get_player_position(player_id)

    print(f"Analyzing stats for players in position: {position}, season: {season_id}")
    players_df = get_players_by_position_and_season(position, season_id)

    if players_df is None or players_df.empty:
        print(f"No players found in position {position} for season {season_id}.")
        sys.exit(1)

    print(f"Found {len(players_df)} players in position {position} who played in season {season_id}")
    if 'MIN' in players_df.columns:
        players_df = players_df[players_df['MIN'] > 15]
        print(f"Filtered to {len(players_df)} players with >15 MIN per game.")

    stats_columns = [
        'PLAYER_ID', 'PLAYER_NAME', 'PTS', 'FGA', 'FGM', 'FG_PCT',
        'FG3A', 'FG3M', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT'
    ]
    stats_df = players_df[stats_columns].copy()

    stats_df = calculate_advanced_metrics(stats_df)
    stats_df = collect_shot_zone_stats(stats_df, season_id)

    stats_df_numeric = stats_df.drop(columns=['PLAYER_ID', 'PLAYER_NAME'])
    stats_df_numeric = stats_df_numeric.dropna()
    stats_mean = stats_df_numeric.mean()
    stats_std = stats_df_numeric.std()

    print(f"\nMean stats for position {position} in season {season_id}:")
    print(stats_mean)
    print("\nStandard deviation of stats:")
    print(stats_std)

if __name__ == "__main__":
    main()
