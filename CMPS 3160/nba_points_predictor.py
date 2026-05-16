

import os
import glob
import re
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import scheduleleaguev2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
import math

class NBAPointsPredictor:
    

    def round_to_betting_half(self,value):
        """
        Rounds to the nearest .5 betting line.
        Whole numbers are not allowed.

        Examples:
            23.38 -> 23.5
            23.89 -> 23.5
            24.10 -> 24.5
            24.70 -> 24.5
        """
        lower_half = math.floor(value) + 0.5
        upper_half = math.ceil(value) + 0.5

        if abs(value - lower_half) <= abs(value - upper_half):
            return float(lower_half)
        else:
            return float(upper_half)
    def __init__(
        self,
        stats_df=None,
        stats_path=None,
        cv=8,
        ridge_alpha=1.0,
        auto_build_future_schedule=True
    ):
        """
        NBA player points prediction engine.

        Initialize with either:

        1. stats_df:
            Existing all_years_game_stats dataframe.

        2. stats_path:
            Folder containing game_stats_*.csv files.

        For your AI Engineering project, prefer:

            predictor = NBAPointsPredictor(stats_df=all_years_game_stats)
        """

        self.cv = cv
        self.ridge_alpha = ridge_alpha

        if stats_df is not None:
            self.all_years_game_stats = stats_df.copy()
        elif stats_path is not None:
            self.all_years_game_stats = self.load_all_game_stats(stats_path)
        else:
            raise ValueError("You must provide either stats_df or stats_path.")

        self.feature_cols = [
            "Avg_PPG_last_6",
            "Avg_PPG_Matchup",
            "USG_PCT",
            "3PAr_last_6",
            "2PAr_last_6",
            "FTAr_last_6",
            "HOME_AWAY_PPG"
        ]

        self.model_cache = {}
        self.results_cache = {}

        self.prepare_base_features()

        if auto_build_future_schedule:
            self.future_games_df = self.build_future_schedule_df()
        else:
            self.future_games_df = None

    # ============================================================
    # Data loading
    # ============================================================

    def load_all_game_stats(self, stats_path):
        csv_files = glob.glob(os.path.join(stats_path, "game_stats_*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No files matching game_stats_*.csv found in {stats_path}"
            )

        frames = []

        for path in csv_files:
            df = pd.read_csv(path)
            frames.append(df)

        all_stats = pd.concat(frames, ignore_index=True)

        return all_stats

    # ============================================================
    # Basic cleaning / feature engineering
    # ============================================================

    def normalize_player_name(self, name):
        """
        Internal replacement for normalize_string from basic_stat_grab.
        Keeps the predictor file self-contained.
        """
        if pd.isna(name):
            return name

        name = str(name).lower().strip()
        name = name.replace(".", "")
        name = name.replace("'", "")
        name = name.replace("-", " ")
        name = re.sub(r"\s+", " ", name)

        return name

    def prepare_base_features(self):
        """
        Takes the baseline all_years_game_stats dataframe and creates
        every derived column needed for model training and prediction.
        """

        df = self.all_years_game_stats.copy()

        # -------------------------
        # Date / season setup
        # -------------------------
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        if "SEASON_START" not in df.columns:
            df["SEASON_START"] = (
                df["SEASON"]
                .astype(str)
                .str.split("-")
                .str[0]
                .astype(int)
            )

        # -------------------------
        # Drop unnecessary columns if they exist
        # -------------------------
        drop_cols = [
            "FANTASY_PTS",
            "VIDEO_AVAILABLE",
            "SEASON_TYPE",
            "SEASON_ID",
            "TEAM_NAME",
            "WL"
        ]

        existing_drop_cols = [col for col in drop_cols if col in df.columns]

        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)

        # -------------------------
        # Normalize names
        # -------------------------
        df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(self.normalize_player_name)

        # -------------------------
        # Remove rows where the player logged no minutes
        # -------------------------
        if "MIN" in df.columns:
            df = df[df["MIN"] > 0].copy()

        # -------------------------
        # Scoring efficiency / tendency stats
        # -------------------------
        shot_denominator = df["FGA"] + 0.44 * df["FTA"]

        df["TS_PCT"] = np.where(
            shot_denominator > 0,
            df["PTS"] / (2 * shot_denominator),
            0.0
        )

        df["EFG_PCT"] = np.where(
            df["FGA"] > 0,
            (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"],
            0.0
        )

        df["PPS"] = np.where(
            df["FGA"] > 0,
            df["PTS"] / df["FGA"],
            0.0
        )

        df["3PAr"] = np.where(
            df["FGA"] > 0,
            df["FG3A"] / df["FGA"],
            0.0
        )

        df["2PAr"] = np.where(
            df["FGA"] > 0,
            (df["FGA"] - df["FG3A"]) / df["FGA"],
            0.0
        )

        df["FTAr"] = np.where(
            df["FGA"] > 0,
            df["FTA"] / df["FGA"],
            0.0
        )

        # -------------------------
        # HOME and OPPONENT columns
        # -------------------------
        df["MATCHUP"] = df["MATCHUP"].astype(str)

        df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False, na=False)

        df["OPPONENT"] = np.where(
            df["MATCHUP"].str.contains("vs.", regex=False, na=False),
            df["MATCHUP"].str.split("vs.").str[1].str.strip(),
            df["MATCHUP"].str.split("@").str[1].str.strip()
        )

        # -------------------------
        # Sort once before rolling features
        # -------------------------
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        self.all_years_game_stats = df

        # -------------------------
        # Model features
        # -------------------------
        self.get_last_6_points()
        self.get_matchup_average()
        self.get_usage_rate()
        self.get_last_6_shot_rates()
        self.get_home_away_season_avg("PTS", "HOME_AWAY_PPG")

    def get_last_6_points(self):
        df = self.all_years_game_stats

        df["Avg_PPG_last_6"] = (
            df.sort_values(["PLAYER_ID", "GAME_DATE"])
            .groupby("PLAYER_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )

    def get_matchup_average(self):
        df = self.all_years_game_stats

        df["Avg_PPG_Matchup"] = (
            df.sort_values(["PLAYER_ID", "GAME_DATE"])
            .groupby(["PLAYER_ID", "OPPONENT"])["PTS"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    def get_last_6_shot_rates(self):
        df = self.all_years_game_stats

        for source_col, new_col in [
            ("3PAr", "3PAr_last_6"),
            ("2PAr", "2PAr_last_6"),
            ("FTAr", "FTAr_last_6")
        ]:
            df[new_col] = (
                df.sort_values(["PLAYER_ID", "GAME_DATE"])
                .groupby("PLAYER_ID")[source_col]
                .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
            )

    def get_usage_rate(self):
        """
        Creates USG_PCT as a prior season-to-date average usage rate.
        This avoids leaking the current game's stats into the prediction row.
        """

        df = self.all_years_game_stats.copy()
        df["_ORIGINAL_INDEX"] = df.index

        team_totals = (
            df.groupby(["GAME_ID", "TEAM_ID"])[["FGA", "FTA", "TOV", "MIN"]]
            .sum()
            .rename(columns={
                "FGA": "TEAM_FGA",
                "FTA": "TEAM_FTA",
                "TOV": "TEAM_TOV",
                "MIN": "TEAM_MIN"
            })
            .reset_index()
        )

        temp = df.merge(
            team_totals,
            on=["GAME_ID", "TEAM_ID"],
            how="left"
        )

        denominator = (
            temp["MIN"]
            * (
                temp["TEAM_FGA"]
                + 0.44 * temp["TEAM_FTA"]
                + temp["TEAM_TOV"]
            )
        )

        temp["GAME_USG_PCT"] = np.where(
            denominator > 0,
            100
            * (
                temp["FGA"]
                + 0.44 * temp["FTA"]
                + temp["TOV"]
            )
            * (temp["TEAM_MIN"] / 5)
            / denominator,
            np.nan
        )

        temp = temp.sort_values(["SEASON", "PLAYER_ID", "GAME_DATE"])

        temp["USG_PCT"] = (
            temp.groupby(["SEASON", "PLAYER_ID"])["GAME_USG_PCT"]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

        temp = temp.sort_values("_ORIGINAL_INDEX")

        self.all_years_game_stats["USG_PCT"] = temp["USG_PCT"].values

    def get_home_away_season_avg(self, stat_col, new_col):
        """
        Creates a prior season-to-date home/away average.

        Example:
            HOME_AWAY_PPG = player's prior home PPG if current game is home,
                            prior away PPG if current game is away.
        """

        df = self.all_years_game_stats.copy()
        df["_ORIGINAL_INDEX"] = df.index

        sorted_df = df.sort_values(["SEASON", "PLAYER_ID", "HOME", "GAME_DATE"])

        sorted_df[new_col] = (
            sorted_df
            .groupby(["SEASON", "PLAYER_ID", "HOME"])[stat_col]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

        sorted_df = sorted_df.sort_values("_ORIGINAL_INDEX")

        self.all_years_game_stats[new_col] = sorted_df[new_col].values

    # ============================================================
    # Future NBA schedule
    # ============================================================

    def get_current_nba_season(self, today=None):
        """
        Returns NBA season string like '2025-26'.
        NBA seasons start around October.
        """

        if today is None:
            today = pd.Timestamp.today()

        today = pd.to_datetime(today)

        if today.tzinfo is not None:
            today = today.tz_convert(None)

        if today.month >= 10:
            return f"{today.year}-{str(today.year + 1)[-2:]}"
        else:
            return f"{today.year - 1}-{str(today.year)[-2:]}"

    def build_future_schedule_df(self, season=None, today=None):
        """
        Builds a dataframe of all currently scheduled NBA games
        that have not happened yet.

        Returns one row per team per game:

            TEAM, OPPONENT, GAME_DATE, HOME
        """

        if today is None:
            today = pd.Timestamp.today().normalize()
        else:
            today = pd.to_datetime(today).normalize()

        if today.tzinfo is not None:
            today = today.tz_convert(None)

        if season is None:
            season = self.get_current_nba_season(today)

        schedule = scheduleleaguev2.ScheduleLeagueV2(
            league_id="00",
            season=season
        )

        games = schedule.get_data_frames()[0].copy()

        games["GAME_DATE"] = (
            pd.to_datetime(games["gameDateEst"], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )

        # gameStatus:
        # 1 = scheduled
        # 2 = in progress
        # 3 = final
        future_games = games[
            (games["GAME_DATE"] >= today)
            & (games["gameStatus"] == 1)
        ].copy()

        home_rows = future_games[[
            "homeTeam_teamTricode",
            "awayTeam_teamTricode",
            "GAME_DATE"
        ]].copy()

        home_rows.columns = ["TEAM", "OPPONENT", "GAME_DATE"]
        home_rows["HOME"] = True

        away_rows = future_games[[
            "awayTeam_teamTricode",
            "homeTeam_teamTricode",
            "GAME_DATE"
        ]].copy()

        away_rows.columns = ["TEAM", "OPPONENT", "GAME_DATE"]
        away_rows["HOME"] = False

        future_schedule_df = pd.concat(
            [home_rows, away_rows],
            ignore_index=True
        )

        future_schedule_df = future_schedule_df[
            future_schedule_df["TEAM"].notna()
            & future_schedule_df["OPPONENT"].notna()
            & (future_schedule_df["TEAM"] != "None")
            & (future_schedule_df["OPPONENT"] != "None")
        ].copy()

        future_schedule_df["GAME_DATE"] = pd.to_datetime(
            future_schedule_df["GAME_DATE"]
        ).dt.normalize()

        future_schedule_df = (
            future_schedule_df
            .sort_values(["GAME_DATE", "TEAM"])
            .reset_index(drop=True)
        )

        return future_schedule_df

    # ============================================================
    # Player helpers
    # ============================================================

    def get_player_games(self, player_name):
        player_name = self.normalize_player_name(player_name)

        player_games = (
            self.all_years_game_stats.loc[
                self.all_years_game_stats["PLAYER_NAME"] == player_name
            ]
            .sort_values("GAME_DATE")
            .copy()
        )

        if player_games.empty:
            raise ValueError(f"No games found for player: {player_name}")

        return player_games

    def get_prior_player_games(self, player_name, game_date):
        game_date = pd.to_datetime(game_date).normalize()

        player_games = self.get_player_games(player_name).copy()
        player_games["GAME_DATE"] = pd.to_datetime(
            player_games["GAME_DATE"]
        ).dt.normalize()

        prior_games = player_games.loc[
            player_games["GAME_DATE"] < game_date
        ].sort_values("GAME_DATE")

        if prior_games.empty:
            raise ValueError(
                f"No prior games found for {player_name} before {game_date.date()}."
            )

        return prior_games

    def get_player_game_on_date(self, player_name, game_date):
        """
        Returns either:

        1. The actual historical game row, if the player already played
           on that date.

        2. A synthetic future game row, if the player's latest known team
           has a scheduled future game on that date.

        3. None, if no game exists for that player/team on that date.
        """

        game_date = pd.to_datetime(game_date).normalize()

        player_games = self.get_player_games(player_name).copy()
        player_games["GAME_DATE"] = pd.to_datetime(
            player_games["GAME_DATE"]
        ).dt.normalize()

        # -------------------------
        # Historical game lookup
        # -------------------------
        historical_game = player_games.loc[
            player_games["GAME_DATE"] == game_date
        ]

        if not historical_game.empty:
            game = historical_game.iloc[0].copy()
            game["IS_FUTURE"] = False
            return game

        # -------------------------
        # Future game lookup
        # -------------------------
        if self.future_games_df is None:
            self.future_games_df = self.build_future_schedule_df()

        latest_game = player_games.sort_values("GAME_DATE").iloc[-1]

        if "TEAM_ABBREVIATION" not in latest_game.index:
            raise ValueError(
                "TEAM_ABBREVIATION column is required to match players "
                "to future scheduled games."
            )

        player_team = latest_game["TEAM_ABBREVIATION"]

        future_match = self.future_games_df.loc[
            (
                self.future_games_df["TEAM"].str.upper()
                == str(player_team).upper()
            )
            & (
                self.future_games_df["GAME_DATE"].dt.normalize()
                == game_date
            )
        ]

        if future_match.empty:
            return None

        future_game = future_match.iloc[0]

        return pd.Series({
            "PLAYER_NAME": latest_game["PLAYER_NAME"],
            "PLAYER_ID": latest_game["PLAYER_ID"],
            "TEAM_ABBREVIATION": player_team,
            "GAME_DATE": future_game["GAME_DATE"],
            "OPPONENT": future_game["OPPONENT"],
            "HOME": future_game["HOME"],
            "SEASON": latest_game["SEASON"],
            "PTS": np.nan,
            "IS_FUTURE": True
        })

    # ============================================================
    # Model dataframe / model training
    # ============================================================

    def create_player_model_df(self, player_name):
        player_df = self.get_player_games(player_name)

        keep_cols = [
            "PLAYER_NAME",
            "PLAYER_ID",
            "SEASON",
            "GAME_DATE",
            "OPPONENT",
            "HOME",
            *self.feature_cols,
            "PTS"
        ]

        missing_cols = [
            col for col in keep_cols
            if col not in player_df.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing columns in stats dataframe: {missing_cols}")

        player_df = player_df[keep_cols].copy()

        player_df = player_df.dropna(
            subset=self.feature_cols + ["PTS"]
        ).reset_index(drop=True)

        if player_df.empty:
            raise ValueError(
                f"No usable model rows for {player_name}. "
                "Check missing feature values."
            )

        return player_df

    def build_player_regression_model(self, player_name):
        """
        Builds and caches a Ridge regression model for one player.
        """

        cache_key = self.normalize_player_name(player_name)

        if cache_key in self.model_cache:
            return (
                self.model_cache[cache_key],
                self.results_cache[cache_key],
                self.feature_cols
            )

        player_df = self.create_player_model_df(player_name)

        X = player_df[self.feature_cols]
        y = player_df["PTS"]

        cv = min(self.cv, len(player_df))

        if cv < 2:
            raise ValueError(
                f"Not enough games to cross-validate model for {player_name}."
            )

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.ridge_alpha)
        )

        scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2"
        }

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )

        results = {
            "train_MAE": -cv_results["train_MAE"].mean(),
            "validation_MAE": -cv_results["test_MAE"].mean(),
            "train_MSE": -cv_results["train_MSE"].mean(),
            "validation_MSE": -cv_results["test_MSE"].mean(),
            "train_R2": cv_results["train_R2"].mean(),
            "validation_R2": cv_results["test_R2"].mean(),
            "num_games": len(player_df),
            "cv": cv
        }

        model.fit(X, y)

        self.model_cache[cache_key] = model
        self.results_cache[cache_key] = results

        return model, results, self.feature_cols

    # ============================================================
    # Prediction feature creation
    # ============================================================

    def impute_with_previous_game_value(self, value, prior_games, feature_col):
        """
        If value is missing, use the most recent non-null value of
        that already-created historical feature.
        """

        if not pd.isna(value):
            return value

        if feature_col not in prior_games.columns:
            return np.nan

        previous_values = prior_games[feature_col].dropna()

        if previous_values.empty:
            return np.nan

        return previous_values.iloc[-1]

    def create_prediction_features_for_player_date(self, player_name, game_date):
        """
        Creates one model-ready row for a historical or future game.

        For future games:
            - Last 6 features use real prior games.
            - Usage rate uses previous season-to-date value.
            - Home/away PPG uses prior games with matching home/away status.
            - Matchup average uses prior games against that opponent.
        """

        game_date = pd.to_datetime(game_date).normalize()

        target_game = self.get_player_game_on_date(
            player_name=player_name,
            game_date=game_date
        )

        if target_game is None:
            return None

        opponent = target_game["OPPONENT"]
        home = target_game["HOME"]

        prior_games = self.get_prior_player_games(
            player_name=player_name,
            game_date=game_date
        )

        latest_season = target_game["SEASON"]

        # -------------------------
        # Last 6 points
        # -------------------------
        avg_ppg_last_6 = prior_games["PTS"].tail(6).mean()

        avg_ppg_last_6 = self.impute_with_previous_game_value(
            value=avg_ppg_last_6,
            prior_games=prior_games,
            feature_col="Avg_PPG_last_6"
        )

        # -------------------------
        # Last 6 shot rates
        # -------------------------
        three_par_last_6 = prior_games["3PAr"].tail(6).mean()
        two_par_last_6 = prior_games["2PAr"].tail(6).mean()
        ftar_last_6 = prior_games["FTAr"].tail(6).mean()

        three_par_last_6 = self.impute_with_previous_game_value(
            value=three_par_last_6,
            prior_games=prior_games,
            feature_col="3PAr_last_6"
        )

        two_par_last_6 = self.impute_with_previous_game_value(
            value=two_par_last_6,
            prior_games=prior_games,
            feature_col="2PAr_last_6"
        )

        ftar_last_6 = self.impute_with_previous_game_value(
            value=ftar_last_6,
            prior_games=prior_games,
            feature_col="FTAr_last_6"
        )

        # -------------------------
        # Usage rate
        # -------------------------
        season_prior_games = prior_games.loc[
            prior_games["SEASON"] == latest_season
        ].copy()

        season_usg_values = season_prior_games["USG_PCT"].dropna()

        if season_usg_values.empty:
            usg_pct = np.nan
        else:
            usg_pct = season_usg_values.iloc[-1]

        usg_pct = self.impute_with_previous_game_value(
            value=usg_pct,
            prior_games=prior_games,
            feature_col="USG_PCT"
        )

        # -------------------------
        # Home/away PPG
        # -------------------------
        home_away_games = prior_games.loc[
            (prior_games["SEASON"] == latest_season)
            & (prior_games["HOME"] == home)
        ]

        home_away_ppg = home_away_games["PTS"].mean()

        home_away_ppg = self.impute_with_previous_game_value(
            value=home_away_ppg,
            prior_games=prior_games,
            feature_col="HOME_AWAY_PPG"
        )

        # -------------------------
        # Matchup average
        # -------------------------
        matchup_games = prior_games.loc[
            prior_games["OPPONENT"].str.upper() == str(opponent).upper()
        ]

        matchup_ppg = matchup_games["PTS"].tail(3).mean()

        # For a production AI Engineering app, it is safer to fall back
        # than to crash if there is no matchup history.
        if pd.isna(matchup_ppg):
            matchup_ppg = prior_games["Avg_PPG_Matchup"].dropna()

            if matchup_ppg.empty:
                matchup_ppg = prior_games["PTS"].tail(6).mean()
            else:
                matchup_ppg = matchup_ppg.iloc[-1]

        feature_row = pd.DataFrame([{
            "Avg_PPG_last_6": avg_ppg_last_6,
            "Avg_PPG_Matchup": matchup_ppg,
            "USG_PCT": usg_pct,
            "3PAr_last_6": three_par_last_6,
            "2PAr_last_6": two_par_last_6,
            "FTAr_last_6": ftar_last_6,
            "HOME_AWAY_PPG": home_away_ppg
        }])

        # Final safety check for any remaining missing values.
        for col in self.feature_cols:
            if pd.isna(feature_row.loc[0, col]):
                fallback_values = prior_games[col].dropna()

                if not fallback_values.empty:
                    feature_row.loc[0, col] = fallback_values.iloc[-1]

        if feature_row[self.feature_cols].isna().any().any():
            missing_cols = feature_row.columns[
                feature_row.isna().any()
            ].tolist()

            raise ValueError(
                f"Could not create complete feature row for {player_name} "
                f"on {game_date.date()}. Missing: {missing_cols}"
            )

        metadata = {
            "PLAYER_NAME": target_game["PLAYER_NAME"],
            "GAME_DATE": target_game["GAME_DATE"],
            "OPPONENT": opponent,
            "HOME": home,
            "IS_FUTURE": target_game["IS_FUTURE"],
            "ACTUAL_PTS": target_game["PTS"]
        }

        return feature_row, metadata

    # ============================================================
    # Public prediction methods
    # ============================================================

    def predict_player_points_for_date(
        self,
        player_name,
        game_date,
        round_to_half=True
    ):
        """
        Predicts points for one player on one date.

        Returns:
            {
                "raw_prediction": float,
                "rounded_prediction": float
            }

        Returns None if the player's team does not have a game on that date.
        """

        result = self.create_prediction_features_for_player_date(
            player_name=player_name,
            game_date=game_date
        )

        if result is None:
            return None

        feature_row, metadata = result

        model, results, feature_cols = self.build_player_regression_model(
            player_name=player_name
        )

        raw_prediction = float(model.predict(feature_row[feature_cols])[0])

        if round_to_half:
            rounded_prediction = self.round_to_betting_half(raw_prediction)
        else:
            rounded_prediction = raw_prediction

        return {
            "raw_prediction": raw_prediction,
            "rounded_prediction": rounded_prediction
        }

    def predict_many_players_for_date(
        self,
        player_names,
        game_date,
        round_to_half=True,
        skip_errors=True
    ):
        """
        Predicts points for many players on one date.

        Useful for replacing season PPG lines in your AI Engineering project.
        """

        predictions = {}

        for player_name in player_names:
            try:
                predictions[player_name] = self.predict_player_points_for_date(
                    player_name=player_name,
                    game_date=game_date,
                    round_to_half=round_to_half
                )
            except Exception as e:
                if skip_errors:
                    predictions[player_name] = None
                else:
                    raise e

        return predictions

    def get_model_results(self, player_name):
        """
        Returns cached validation results for a player.
        If the model has not been built yet, this builds it first.
        """

        cache_key = self.normalize_player_name(player_name)

        if cache_key not in self.results_cache:
            self.build_player_regression_model(player_name)

        return self.results_cache[cache_key]

    def clear_model_cache(self):
        self.model_cache = {}
        self.results_cache = {}