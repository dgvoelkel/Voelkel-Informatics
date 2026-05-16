def update():
    def get_last_6_points(): #This function gets the average points per game in up to the last 6 games.
        all_years_game_stats[f"Avg_PPG_last_6"] = (
            all_years_game_stats
            .sort_values(["PLAYER_ID", "GAME_DATE"])
            .groupby("PLAYER_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

    def get_matchup_average(): #Gets the points per game average of the 5 previous matchups against the same team.
        all_years_game_stats["Avg_PPG_Matchup"] = (
            all_years_game_stats
            .sort_values(["PLAYER_ID", "GAME_DATE"])
            .groupby(["PLAYER_ID","MATCHUP"])["PTS"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    def get_usage_rate():
        # Team totals for each team in each game
        team_totals = (
            all_years_game_stats
            .groupby(["GAME_ID", "TEAM_ID"])[["FGA", "FTA", "TOV", "MIN"]]
            .sum()
            .rename(columns={
                "FGA": "TEAM_FGA",
                "FTA": "TEAM_FTA",
                "TOV": "TEAM_TOV",
                "MIN": "TEAM_MIN"
            })
            .reset_index()
        )

    # Sort only for calculation, but keep original index for realignment
    sorted_df = all_years_game_stats.sort_values(
        ["SEASON", "PLAYER_ID", "GAME_DATE"]
    ).copy()

    sorted_df["ORIGINAL_INDEX"] = sorted_df.index

    # Attach team totals to each player-game row
    temp = sorted_df.merge(
        team_totals,
        on=["GAME_ID", "TEAM_ID"],
        how="left"
    )

    # Single-game usage rate
    temp["GAME_USG_PCT"] = (
        100
        * (temp["FGA"] + 0.44 * temp["FTA"] + temp["TOV"])
        * (temp["TEAM_MIN"] / 5)
        / (
            temp["MIN"]
            * (temp["TEAM_FGA"] + 0.44 * temp["TEAM_FTA"] + temp["TEAM_TOV"])
        )
    )

    # Season-to-date usage rate before the current game
    temp["USG_PCT"] = (
        temp
        .groupby(["SEASON", "PLAYER_ID"])["GAME_USG_PCT"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Put the values back into the original dataframe order
    all_years_game_stats["USG_PCT"] = (
        temp
        .set_index("ORIGINAL_INDEX")["USG_PCT"]
        .reindex(all_years_game_stats.index)
    )

    def get_last_6_shot_rates(): #This function gets the last 6 games shot tendencies.
        sorted_df = all_years_game_stats.sort_values(
            ["PLAYER_ID", "GAME_DATE"]
        )

        prior_3par = (
            sorted_df
            .groupby("PLAYER_ID")["3PAr"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        prior_2par = (
            sorted_df
            .groupby("PLAYER_ID")["2PAr"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        prior_ftar = (
            sorted_df
            .groupby("PLAYER_ID")["FTAr"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

        all_years_game_stats[f"3PAr_last_6"] = prior_3par.reindex(
            all_years_game_stats.index
        )

        all_years_game_stats[f"2PAr_last_6"] = prior_2par.reindex(
            all_years_game_stats.index
        )

        all_years_game_stats[f"FTAr_last_6"] = prior_ftar.reindex(
            all_years_game_stats.index
            )


    def get_home_away_season_avg(stat_col="PTS", new_col=None):
        """
        Creates a player's season average for a stat, separated by home/away games,
        using only games before the current game.

        Example:
            get_home_away_season_avg("PTS", "HOME_AWAY_PPG")
        """

        if new_col is None:
            new_col = f"SEASON_{stat_col}_AVG_BY_HOME"

        # Sort so cumulative averages are calculated in chronological order
        sorted_df = all_years_game_stats.sort_values(
            ["SEASON", "PLAYER_ID", "HOME", "GAME_DATE"]
        ).copy()

        # Group by player, season, and HOME status
        group_cols = ["SEASON", "PLAYER_ID", "HOME"]

         #Cumulative sum before current game
        prior_sum = (
            sorted_df
            .groupby(group_cols)[stat_col]
            .cumsum()
            - sorted_df[stat_col]
        )

        # Number of prior home/away games
        prior_count = (
            sorted_df
            .groupby(group_cols)
            .cumcount()
        )

        # Prior season average at home or away
        sorted_df[new_col] = prior_sum / prior_count

        # Put the new column back into the original dataframe order
        all_years_game_stats[new_col] = sorted_df.sort_index()[new_col]

        return all_years_game_stats

    get_last_6_points()
    get_matchup_average()
    get_usage_rate()
    get_last_6_shot_rates()
    get_home_away_season_avg("PTS", "HOME_AWAY_PPG")
