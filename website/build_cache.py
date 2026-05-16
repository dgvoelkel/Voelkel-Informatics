#!/usr/bin/env python3
"""
Run during Render's build step to pre-build the predictor joblib cache.
Render build command: pip install -r requirements.txt && python build_cache.py
"""
import os
import sys
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from nba_points_predictor import NBAPointsPredictor, NBASeasonPredictor

NBA_STATS_PATH = os.path.join(_HERE, "nba_player_gamelogs_by_season")
CACHE_DIR = os.path.join(_HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print("Building game predictor cache...")
gp = NBAPointsPredictor(stats_path=NBA_STATS_PATH, auto_build_future_schedule=False)
joblib.dump(gp, os.path.join(CACHE_DIR, "game_predictor.joblib"), compress=3)
print("Game predictor saved.")

print("Building season predictor cache...")
sp = NBASeasonPredictor(stats_path=NBA_STATS_PATH)
joblib.dump(sp, os.path.join(CACHE_DIR, "season_predictor.joblib"), compress=3)
print("Season predictor saved.")
