import os
import glob
import re
import math
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import scheduleleaguev2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate


class NBAPointsPredictor:

    def round_to_betting_half(self, value):
        """
        Rounds to the nearest .5 betting line. Whole numbers are not allowed.

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
        lasso_alpha=0.1,
        auto_build_future_schedule=True
    ):
        """
        NBA player points prediction engine (Model 1 — game-level).

        Initialize with either:
            stats_df:   Existing all_years_game_stats DataFrame.
            stats_path: Folder containing game_stats_*.csv files.
        """

        self.cv = cv
        self.lasso_alpha = lasso_alpha

        if stats_df is not None:
            self.all_years_game_stats = stats_df.copy()
        elif stats_path is not None:
            self.all_years_game_stats = self.load_all_game_stats(stats_path)
        else:
            raise ValueError("You must provide either stats_df or stats_path.")

        self.feature_cols = [
            "Avg_PPG_last_6",
            "Avg_PPG_Matchup",
            "Avg_MIN_last_6",
            "Avg_FTA_last_6",
            "Avg_FGA_last_6",
            "Avg_FG3A_last_6",
            "HOME_AWAY_PPG",
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

        frames = [pd.read_csv(f) for f in csv_files]
        return pd.concat(frames, ignore_index=True)

    # ============================================================
    # Basic cleaning / feature engineering
    # ============================================================

    def normalize_player_name(self, name):
        if pd.isna(name):
            return name

        name = str(name).lower().strip()
        name = name.replace(".", "")
        name = name.replace("'", "")
        name = name.replace("-", " ")
        name = re.sub(r"\s+", " ", name)

        return name

    def prepare_base_features(self):
        df = self.all_years_game_stats.copy()

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        if "SEASON_START" not in df.columns:
            df["SEASON_START"] = (
                df["SEASON"]
                .astype(str)
                .str.split("-")
                .str[0]
                .astype(int)
            )

        drop_cols = [
            "FANTASY_PTS",
            "VIDEO_AVAILABLE",
            "SEASON_TYPE",
            "SEASON_ID",
            "TEAM_NAME",
            "WL",
        ]
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)

        df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(self.normalize_player_name)

        if "MIN" in df.columns:
            df = df[df["MIN"] > 0].copy()

        shot_denominator = df["FGA"] + 0.44 * df["FTA"]

        df["TS_PCT"] = np.where(
            shot_denominator > 0,
            df["PTS"] / (2 * shot_denominator),
            0.0,
        )
        df["EFG_PCT"] = np.where(
            df["FGA"] > 0,
            (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"],
            0.0,
        )
        df["PPS"] = np.where(df["FGA"] > 0, df["PTS"] / df["FGA"], 0.0)
        df["3PAr"] = np.where(df["FGA"] > 0, df["FG3A"] / df["FGA"], 0.0)
        df["2PAr"] = np.where(
            df["FGA"] > 0, (df["FGA"] - df["FG3A"]) / df["FGA"], 0.0
        )
        df["FTAr"] = np.where(df["FGA"] > 0, df["FTA"] / df["FGA"], 0.0)

        df["MATCHUP"] = df["MATCHUP"].astype(str)
        df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False, na=False)
        df["OPPONENT"] = np.where(
            df["MATCHUP"].str.contains("vs.", regex=False, na=False),
            df["MATCHUP"].str.split("vs.").str[1].str.strip(),
            df["MATCHUP"].str.split("@").str[1].str.strip(),
        )

        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        self.all_years_game_stats = df

        self.get_last_6_points()
        self.get_matchup_average()
        self.get_last_6_volume_stats()
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

    def get_last_6_volume_stats(self):
        df = self.all_years_game_stats
        for source_col, new_col in [
            ("MIN", "Avg_MIN_last_6"),
            ("FTA", "Avg_FTA_last_6"),
            ("FGA", "Avg_FGA_last_6"),
            ("FG3A", "Avg_FG3A_last_6"),
        ]:
            df[new_col] = (
                df.sort_values(["PLAYER_ID", "GAME_DATE"])
                .groupby("PLAYER_ID")[source_col]
                .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
            )

    def get_home_away_season_avg(self, stat_col, new_col):
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
            season=season,
            timeout=30,
        )
        games = schedule.get_data_frames()[0].copy()
        games["GAME_DATE"] = (
            pd.to_datetime(games["gameDateEst"], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )

        # gameStatus: 1=scheduled, 2=in-progress, 3=final.
        # During playoffs the API sometimes returns float 1.0 — use pd.to_numeric so
        # "1", 1, and 1.0 all match correctly.
        numeric_status = pd.to_numeric(games["gameStatus"], errors="coerce")
        future_games = games[
            (games["GAME_DATE"] >= today)
            & (numeric_status == 1)
        ].copy()

        home_rows = future_games[
            ["homeTeam_teamTricode", "awayTeam_teamTricode", "GAME_DATE"]
        ].copy()
        home_rows.columns = ["TEAM", "OPPONENT", "GAME_DATE"]
        home_rows["HOME"] = True

        away_rows = future_games[
            ["awayTeam_teamTricode", "homeTeam_teamTricode", "GAME_DATE"]
        ].copy()
        away_rows.columns = ["TEAM", "OPPONENT", "GAME_DATE"]
        away_rows["HOME"] = False

        future_schedule_df = pd.concat([home_rows, away_rows], ignore_index=True)
        future_schedule_df = future_schedule_df[
            future_schedule_df["TEAM"].notna()
            & future_schedule_df["OPPONENT"].notna()
            & (future_schedule_df["TEAM"] != "None")
            & (future_schedule_df["OPPONENT"] != "None")
        ].copy()

        future_schedule_df["GAME_DATE"] = pd.to_datetime(
            future_schedule_df["GAME_DATE"]
        ).dt.normalize()

        return (
            future_schedule_df
            .sort_values(["GAME_DATE", "TEAM"])
            .reset_index(drop=True)
        )

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
        game_date = pd.to_datetime(game_date).normalize()
        player_games = self.get_player_games(player_name).copy()
        player_games["GAME_DATE"] = pd.to_datetime(
            player_games["GAME_DATE"]
        ).dt.normalize()

        historical_game = player_games.loc[player_games["GAME_DATE"] == game_date]
        if not historical_game.empty:
            game = historical_game.iloc[0].copy()
            game["IS_FUTURE"] = False
            return game

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
            (self.future_games_df["TEAM"].str.upper() == str(player_team).upper())
            & (self.future_games_df["GAME_DATE"].dt.normalize() == game_date)
        ]

        if future_match.empty:
            return None

        future_game = future_match.iloc[0]
        return pd.Series(
            {
                "PLAYER_NAME": latest_game["PLAYER_NAME"],
                "PLAYER_ID": latest_game["PLAYER_ID"],
                "TEAM_ABBREVIATION": player_team,
                "GAME_DATE": future_game["GAME_DATE"],
                "OPPONENT": future_game["OPPONENT"],
                "HOME": future_game["HOME"],
                "SEASON": latest_game["SEASON"],
                "PTS": np.nan,
                "IS_FUTURE": True,
            }
        )

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
            "PTS",
        ]
        missing_cols = [col for col in keep_cols if col not in player_df.columns]
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
        cache_key = self.normalize_player_name(player_name)

        if cache_key in self.model_cache:
            return (
                self.model_cache[cache_key],
                self.results_cache[cache_key],
                self.feature_cols,
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
            Lasso(alpha=self.lasso_alpha, max_iter=10000),
        )

        scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2",
        }

        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=True
        )

        results = {
            "train_MAE": -cv_results["train_MAE"].mean(),
            "validation_MAE": -cv_results["test_MAE"].mean(),
            "train_MSE": -cv_results["train_MSE"].mean(),
            "validation_MSE": -cv_results["test_MSE"].mean(),
            "train_R2": cv_results["train_R2"].mean(),
            "validation_R2": cv_results["test_R2"].mean(),
            "num_games": len(player_df),
            "cv": cv,
        }

        model.fit(X, y)
        self.model_cache[cache_key] = model
        self.results_cache[cache_key] = results

        return model, results, self.feature_cols

    # ============================================================
    # Prediction feature creation
    # ============================================================

    def impute_with_previous_game_value(self, value, prior_games, feature_col):
        if not pd.isna(value):
            return value
        if feature_col not in prior_games.columns:
            return np.nan
        previous_values = prior_games[feature_col].dropna()
        if previous_values.empty:
            return np.nan
        return previous_values.iloc[-1]

    def create_prediction_features_for_player_date(self, player_name, game_date):
        game_date = pd.to_datetime(game_date).normalize()

        target_game = self.get_player_game_on_date(
            player_name=player_name, game_date=game_date
        )
        if target_game is None:
            return None

        opponent = target_game["OPPONENT"]
        home = target_game["HOME"]
        prior_games = self.get_prior_player_games(
            player_name=player_name, game_date=game_date
        )
        latest_season = target_game["SEASON"]

        # Last 6 points
        avg_ppg_last_6 = prior_games["PTS"].tail(6).mean()
        avg_ppg_last_6 = self.impute_with_previous_game_value(
            avg_ppg_last_6, prior_games, "Avg_PPG_last_6"
        )

        # Last 6 volume stats
        avg_min_last_6 = prior_games["MIN"].tail(6).mean()
        avg_min_last_6 = self.impute_with_previous_game_value(
            avg_min_last_6, prior_games, "Avg_MIN_last_6"
        )

        avg_fta_last_6 = prior_games["FTA"].tail(6).mean()
        avg_fta_last_6 = self.impute_with_previous_game_value(
            avg_fta_last_6, prior_games, "Avg_FTA_last_6"
        )

        avg_fga_last_6 = prior_games["FGA"].tail(6).mean()
        avg_fga_last_6 = self.impute_with_previous_game_value(
            avg_fga_last_6, prior_games, "Avg_FGA_last_6"
        )

        avg_fg3a_last_6 = prior_games["FG3A"].tail(6).mean()
        avg_fg3a_last_6 = self.impute_with_previous_game_value(
            avg_fg3a_last_6, prior_games, "Avg_FG3A_last_6"
        )

        # Home/away PPG (current season)
        home_away_games = prior_games.loc[
            (prior_games["SEASON"] == latest_season) & (prior_games["HOME"] == home)
        ]
        home_away_ppg = home_away_games["PTS"].mean()
        home_away_ppg = self.impute_with_previous_game_value(
            home_away_ppg, prior_games, "HOME_AWAY_PPG"
        )

        # Matchup average (last 3 vs this opponent)
        matchup_games = prior_games.loc[
            prior_games["OPPONENT"].str.upper() == str(opponent).upper()
        ]
        matchup_ppg = matchup_games["PTS"].tail(3).mean()
        if pd.isna(matchup_ppg):
            matchup_ppg = prior_games["Avg_PPG_Matchup"].dropna()
            if matchup_ppg.empty:
                matchup_ppg = prior_games["PTS"].tail(6).mean()
            else:
                matchup_ppg = matchup_ppg.iloc[-1]

        feature_row = pd.DataFrame(
            [
                {
                    "Avg_PPG_last_6": avg_ppg_last_6,
                    "Avg_PPG_Matchup": matchup_ppg,
                    "Avg_MIN_last_6": avg_min_last_6,
                    "Avg_FTA_last_6": avg_fta_last_6,
                    "Avg_FGA_last_6": avg_fga_last_6,
                    "Avg_FG3A_last_6": avg_fg3a_last_6,
                    "HOME_AWAY_PPG": home_away_ppg,
                }
            ]
        )

        # Final fallback: use last known historical feature value
        for col in self.feature_cols:
            if pd.isna(feature_row.loc[0, col]):
                fallback_values = prior_games[col].dropna() if col in prior_games.columns else pd.Series([], dtype=float)
                if not fallback_values.empty:
                    feature_row.loc[0, col] = fallback_values.iloc[-1]

        if feature_row[self.feature_cols].isna().any().any():
            missing_cols = feature_row.columns[feature_row.isna().any()].tolist()
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
            "ACTUAL_PTS": target_game["PTS"],
        }

        return feature_row, metadata

    # ============================================================
    # Public prediction methods
    # ============================================================

    def predict_player_points_for_date(self, player_name, game_date, round_to_half=True):
        """
        Predicts points for one player on one date.

        Returns {"raw_prediction": float, "rounded_prediction": float}
        or None if the player's team has no game on that date.
        """
        result = self.create_prediction_features_for_player_date(
            player_name=player_name, game_date=game_date
        )
        if result is None:
            return None

        feature_row, metadata = result
        model, results, feature_cols = self.build_player_regression_model(player_name)
        raw_prediction = float(model.predict(feature_row[feature_cols])[0])

        if round_to_half:
            rounded_prediction = self.round_to_betting_half(raw_prediction)
        else:
            rounded_prediction = raw_prediction

        return {"raw_prediction": raw_prediction, "rounded_prediction": rounded_prediction}

    def predict_many_players_for_date(
        self, player_names, game_date, round_to_half=True, skip_errors=True
    ):
        predictions = {}
        for player_name in player_names:
            try:
                predictions[player_name] = self.predict_player_points_for_date(
                    player_name=player_name,
                    game_date=game_date,
                    round_to_half=round_to_half,
                )
            except Exception as e:
                if skip_errors:
                    predictions[player_name] = None
                else:
                    raise e
        return predictions

    def get_model_results(self, player_name):
        cache_key = self.normalize_player_name(player_name)
        if cache_key not in self.results_cache:
            self.build_player_regression_model(player_name)
        return self.results_cache[cache_key]

    def clear_model_cache(self):
        self.model_cache = {}
        self.results_cache = {}


# ============================================================
# Model 2 — Season-level PPG predictor
# ============================================================

class NBASeasonPredictor:
    """
    Predicts a player's points-per-game average for the next season
    based on their current season statistics.

    Mirrors the season-level Lasso regression from CMPS3160_Project (1).ipynb.
    """

    FEATURE_COLS = [
        "MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
        "REB", "AST", "STL", "BLK", "TOV",
        "TS_PCT", "EFG_PCT", "PPS", "3PAr", "2PAr", "FTAr",
        "PLUS_MINUS", "PTS", "GAMES_PLAYED",
    ]

    def __init__(self, stats_path: str, lasso_alpha: float = 0.1, cv: int = 8):
        self.lasso_alpha = lasso_alpha
        self.cv = cv
        self._load_raw_data(stats_path)
        self._prepare_season_stats()
        self._train_model()

    def _load_raw_data(self, stats_path: str):
        csv_files = glob.glob(os.path.join(stats_path, "game_stats_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No game_stats_*.csv files found in {stats_path}"
            )

        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)

        # Basic cleaning
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        if "SEASON_START" not in df.columns:
            df["SEASON_START"] = (
                df["SEASON"].astype(str).str.split("-").str[0].astype(int)
            )

        drop_cols = ["FANTASY_PTS", "VIDEO_AVAILABLE", "SEASON_TYPE", "SEASON_ID", "TEAM_NAME", "WL"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Normalize names
        def _norm(name):
            if pd.isna(name):
                return name
            name = str(name).lower().strip()
            name = re.sub(r"[.'`]", "", name)
            name = name.replace("-", " ")
            return re.sub(r"\s+", " ", name)

        df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(_norm)

        # Remove DNPs
        if "MIN" in df.columns:
            df = df[df["MIN"] > 0].copy()

        # Derive advanced metrics at the game level before aggregating
        shot_denom = df["FGA"] + 0.44 * df["FTA"]
        df["TS_PCT"] = np.where(shot_denom > 0, df["PTS"] / (2 * shot_denom), 0.0)
        df["EFG_PCT"] = np.where(df["FGA"] > 0, (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"], 0.0)
        df["PPS"] = np.where(df["FGA"] > 0, df["PTS"] / df["FGA"], 0.0)
        df["3PAr"] = np.where(df["FGA"] > 0, df["FG3A"] / df["FGA"], 0.0)
        df["2PAr"] = np.where(df["FGA"] > 0, (df["FGA"] - df["FG3A"]) / df["FGA"], 0.0)
        df["FTAr"] = np.where(df["FGA"] > 0, df["FTA"] / df["FGA"], 0.0)

        self._raw_df = df

    def _prepare_season_stats(self):
        df = self._raw_df

        avg_cols = [
            "MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
            "REB", "AST", "STL", "BLK", "TOV",
            "TS_PCT", "EFG_PCT", "PPS", "3PAr", "2PAr", "FTAr",
            "PLUS_MINUS", "PTS",
        ]
        existing_avg_cols = [c for c in avg_cols if c in df.columns]

        season_stats = (
            df.groupby(["PLAYER_NAME", "PLAYER_ID", "SEASON_START"])[existing_avg_cols]
            .mean()
            .reset_index()
        )

        games_played = (
            df.groupby(["PLAYER_NAME", "PLAYER_ID", "SEASON_START"])
            .size()
            .reset_index(name="GAMES_PLAYED")
        )
        season_stats = season_stats.merge(
            games_played, on=["PLAYER_NAME", "PLAYER_ID", "SEASON_START"], how="left"
        )

        # Only keep players with at least 20 games (matches notebook filter)
        season_stats = season_stats[season_stats["GAMES_PLAYED"] >= 20].copy()

        season_stats = season_stats.sort_values(["PLAYER_ID", "SEASON_START"])

        # Target: next season's PPG
        season_stats["PTS_next"] = (
            season_stats.groupby("PLAYER_ID")["PTS"].shift(-1)
        )

        self.season_stats_full = season_stats
        self.train_season_stats = season_stats.dropna(subset=["PTS_next"]).copy()

    def _train_model(self):
        model_df = self.train_season_stats.dropna(
            subset=[c for c in self.FEATURE_COLS if c in self.train_season_stats.columns]
        ).copy()

        available_features = [c for c in self.FEATURE_COLS if c in model_df.columns]
        X = model_df[available_features]
        y = model_df["PTS_next"]

        self._used_features = available_features

        self.model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=self.lasso_alpha, max_iter=10000),
        )
        self.model.fit(X, y)

    def predict_next_season(self, player_name: str) -> dict:
        """
        Predict a player's PPG for the next season.

        Returns a dict with current season stats and the prediction.
        Raises ValueError if the player is not found.
        """
        norm_name = player_name.lower().strip()
        norm_name = re.sub(r"[.'`]", "", norm_name)
        norm_name = norm_name.replace("-", " ")
        norm_name = re.sub(r"\s+", " ", norm_name).strip()
        player_df = self.season_stats_full[
            self.season_stats_full["PLAYER_NAME"] == norm_name
        ].sort_values("SEASON_START")

        if player_df.empty:
            raise ValueError(f"No seasons found for player: {player_name!r}")

        latest_row = player_df.iloc[[-1]].copy()
        prediction = float(self.model.predict(latest_row[self._used_features])[0])

        return {
            "PLAYER_NAME": player_name,
            "PLAYER_ID": int(latest_row["PLAYER_ID"].iloc[0]),
            "SEASON_START": int(latest_row["SEASON_START"].iloc[0]),
            "current_season_ppg": float(latest_row["PTS"].iloc[0]),
            "predicted_next_season_ppg": prediction,
            "games_played_current_season": int(latest_row["GAMES_PLAYED"].iloc[0]),
        }


# ============================================================
# Interactive CLI
# ============================================================

if __name__ == "__main__":
    import sys

    STATS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nba_player_gamelogs_by_season",
    )

    print("=" * 50)
    print("  NBA POINTS PREDICTOR")
    print("=" * 50)
    print("  1. Game-level prediction (Model 1)")
    print("  2. Season-level prediction (Model 2)")
    print("=" * 50)

    choice = input("Choose a model (1 or 2): ").strip()

    if choice == "1":
        print("\nLoading game stats — this may take ~30 seconds on first run...")
        predictor = NBAPointsPredictor(stats_path=STATS_PATH)
        player = input("Player name: ").strip()
        date = input("Game date (YYYY-MM-DD): ").strip()
        result = predictor.predict_player_points_for_date(player, date)
        if result is None:
            print(f"\nNo game found for {player!r} on {date}.")
        else:
            print(f"\nPrediction for {player} on {date}:")
            print(f"  Raw prediction:     {result['raw_prediction']:.2f} pts")
            print(f"  Rounded prediction: {result['rounded_prediction']} pts")

    elif choice == "2":
        print("\nLoading season stats and training model...")
        sp = NBASeasonPredictor(stats_path=STATS_PATH)
        player = input("Player name: ").strip()
        result = sp.predict_next_season(player)
        print(f"\nPlayer:                    {result['PLAYER_NAME']}")
        print(f"Season:                    {result['SEASON_START']}-{result['SEASON_START']+1}")
        print(f"Current season PPG:        {result['current_season_ppg']:.1f}")
        print(f"Games played this season:  {result['games_played_current_season']}")
        print(f"Predicted next season PPG: {result['predicted_next_season_ppg']:.2f}")

    else:
        print("Invalid choice. Please enter 1 or 2.")
        sys.exit(1)
