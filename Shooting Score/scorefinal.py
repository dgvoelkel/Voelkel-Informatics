import sys
import basic_stat_grab
import zcalc
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def calculate_composite_z_score(player_name, season_id=None):
    """
    Calculates the standardized composite z-score for a player in a given season.
    """
    player_id = basic_stat_grab.get_player_id(player_name)
    player_position = basic_stat_grab.get_player_position(player_id)

    # If season_id not provided, use last season
    if not season_id:
        season_id = basic_stat_grab.get_last_season(player_id)
    
    # Step 2: Get player stats for that season
    player_stats = basic_stat_grab.get_player_stats(player_id, season_id)
    
    # Step 3: Get stats for players of the same position in the same season
    position_stats_df = zcalc.get_players_by_position_and_season(player_position, season_id)
    if position_stats_df is None or position_stats_df.empty:
        print(f"No stats found for players in position {player_position} for season {season_id}.")
        return None

    # Step 4: Calculate advanced metrics for all players
    position_stats_df = zcalc.calculate_advanced_metrics(position_stats_df)

    # Ensure the target player's stats are in DataFrame
    player_stats_df = pd.DataFrame([player_stats])
    full_stats_df = pd.concat([position_stats_df, player_stats_df], ignore_index=True, sort=False)

    # Step 5: Select numeric columns and handle missing
    numeric_columns = full_stats_df.select_dtypes(include=[np.number]).columns
    stats_for_z = full_stats_df[numeric_columns].dropna(axis=1)  # drop columns with NaN

    stds = stats_for_z.std()
    stats_with_variance = stds[stds != 0].index.tolist()

    stats_for_z = stats_for_z[stats_with_variance]
    means = stats_for_z.mean()
    stds = stats_for_z.std()

    # Step 8: Calculate z-scores
    z_scores_df = (stats_for_z - means) / stds

    # Step 9: Composite z-score
    z_scores_df['CompositeZ'] = z_scores_df.mean(axis=1)

    # The target player's row is the last one (because of concat)
    target_player_z = z_scores_df.iloc[-1]['CompositeZ']

    composite_mean = z_scores_df['CompositeZ'].mean()
    composite_std = z_scores_df['CompositeZ'].std()
    z_scores_df['StandardizedCompositeZ'] = (z_scores_df['CompositeZ'] - composite_mean) / composite_std

    standardized_composite_z_score = (target_player_z - composite_mean) / composite_std

    all_standardized_scores = z_scores_df['StandardizedCompositeZ']
    percentile = percentileofscore(all_standardized_scores, standardized_composite_z_score, kind='mean')
    rank = all_standardized_scores.rank(ascending=False, method='min').iloc[-1]

    return {
        "Season": season_id,
        "Position": player_position,
        "Composite Z-Score": standardized_composite_z_score,
        "Percentile": percentile,
        "Rank": int(rank),
        "Total Players": len(all_standardized_scores)
    }

def main():
    """
    Usage:
        python scorefinal.py "Player Name" [Season]
    """
    if len(sys.argv) < 2:
        print("Usage: python scorefinal.py 'Player Name' [Season]")
        sys.exit(1)

    player_name = sys.argv[1]
    season_id = sys.argv[2] if len(sys.argv) == 3 else None

    result = calculate_composite_z_score(player_name, season_id)
    if result is not None:
        print(f"Shooting Score for {player_name} in {result['Season']}: {result['Composite Z-Score']:.4f}")
        print(f"Position: {result['Position']}")
        print(f"Percentile: {result['Percentile']:.2f}%")
        print(f"Rank: {result['Rank']} out of {result['Total Players']} {result['Position']}s")

if __name__ == "__main__":
    main()

