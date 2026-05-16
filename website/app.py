import os
import glob
import threading
import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_caching import Cache

import basic_stat_grab
import zcalc
from scorefinal import calculate_composite_z_score
from nba_points_predictor import NBAPointsPredictor, NBASeasonPredictor

# ── App + caching setup ──────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, "cache")
NBA_STATS_PATH = os.path.join(_HERE, "nba_player_gamelogs_by_season")

# ── In-memory singletons ─────────────────────────────────────────────────────
_game_predictor = None
_season_predictor = None
_warmup_error = None


def _cache_is_fresh(cache_file: str, stats_path: str) -> bool:
    csv_files = glob.glob(os.path.join(stats_path, "game_stats_*.csv"))
    if not csv_files:
        return False
    newest_csv = max(os.path.getmtime(f) for f in csv_files)
    # Allow 5-minute skew: on Render, git checkout writes files sequentially so
    # CSVs can have a slightly newer mtime than the cache even in the same deploy.
    return os.path.getmtime(cache_file) >= newest_csv - 300


def _get_game_predictor() -> NBAPointsPredictor:
    global _game_predictor
    if _game_predictor is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(_CACHE_DIR, "game_predictor.joblib")
        if os.path.exists(cache_file) and _cache_is_fresh(cache_file, NBA_STATS_PATH):
            _game_predictor = joblib.load(cache_file)
        else:
            # auto_build_future_schedule=False avoids an NBA API call at startup
            # that can fail and crash the warmup thread. Schedule is built lazily
            # on first prediction request via get_player_game_on_date().
            _game_predictor = NBAPointsPredictor(
                stats_path=NBA_STATS_PATH, auto_build_future_schedule=False
            )
            joblib.dump(_game_predictor, cache_file, compress=3)
    return _game_predictor


def _get_season_predictor() -> NBASeasonPredictor:
    global _season_predictor
    if _season_predictor is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(_CACHE_DIR, "season_predictor.joblib")
        if os.path.exists(cache_file) and _cache_is_fresh(cache_file, NBA_STATS_PATH):
            _season_predictor = joblib.load(cache_file)
        else:
            _season_predictor = NBASeasonPredictor(stats_path=NBA_STATS_PATH)
            joblib.dump(_season_predictor, cache_file, compress=3)
    return _season_predictor


def _warmup():
    global _warmup_error
    try:
        _get_game_predictor()
        _get_season_predictor()
    except Exception as e:
        _warmup_error = str(e)


# Start background warmup immediately on server start
threading.Thread(target=_warmup, daemon=True).start()


# ── NBA API cached helpers ────────────────────────────────────────────────────

@cache.memoize(timeout=3600)
def _cached_get_player_id(player_name: str):
    return basic_stat_grab.get_player_id(player_name)


@cache.memoize(timeout=3600)
def _cached_get_seasons(player_name: str):
    player_id = _cached_get_player_id(player_name)
    return basic_stat_grab.get_all_seasons_for_player(player_id)


@cache.memoize(timeout=3600)
def _cached_search(query: str):
    from nba_api.stats.static import players
    all_players = players.get_players()
    q = query.lower()
    return [p["full_name"] for p in all_players if q in p["full_name"].lower()][:20]


def _normalize_name(name: str) -> str:
    """Same logic as NBAPointsPredictor.normalize_player_name."""
    import re as _re
    name = str(name).lower().strip()
    name = name.replace(".", "").replace("'", "").replace("-", " ")
    return _re.sub(r"\s+", " ", name)



def _cached_nba_search(query: str):
    """Autocomplete filtered to players who exist in all_years_game_stats."""
    from nba_api.stats.static import players
    q = query.lower()
    all_players = players.get_players()

    if _game_predictor is None:
        # Predictor not ready yet — return unfiltered search so autocomplete works during warmup
        return [p["full_name"] for p in all_players if q in p["full_name"].lower()][:20]

    csv_names = set(_game_predictor.all_years_game_stats["PLAYER_NAME"].unique())
    results = []
    for p in all_players:
        if _normalize_name(p["full_name"]) in csv_names and q in p["full_name"].lower():
            results.append(p["full_name"])
    return results[:20]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/select")
def select():
    return render_template("select.html")


# ── Shooting Score ────────────────────────────────────────────────────────────

@app.route("/shooting-score")
def shooting_score():
    return render_template("shooting_score_index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    try:
        return jsonify(_cached_search(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/nba-search")
def nba_search():
    """Autocomplete for NBA predictor — only players in the game-log CSVs."""
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    try:
        return jsonify(_cached_nba_search(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/seasons")
def seasons():
    player_name = request.args.get("player_name", "").strip()
    if not player_name:
        return jsonify([])
    try:
        return jsonify(_cached_get_seasons(player_name))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shooting-score/result", methods=["POST"])
def shooting_score_result():
    player_name = request.form.get("player", "").strip()
    season_id = request.form.get("season_dropdown", "").strip()
    if not player_name or not season_id:
        return render_template("error.html", message="Player name and season are required.")
    try:
        result = calculate_composite_z_score(player_name, season_id)
    except Exception as e:
        return render_template("error.html", message=str(e))
    if result is None:
        return render_template(
            "error.html",
            message=f"No shooting data found for {player_name} in {season_id}.",
        )
    return render_template(
        "shooting_score_result.html", player_name=player_name.title(), result=result
    )


@app.route("/player-games")
def player_games():
    """Return upcoming scheduled games for a player's current team."""
    player_name = request.args.get("player_name", "").strip()
    if not player_name:
        return jsonify([])
    try:
        predictor = _get_game_predictor()
        norm_name = predictor.normalize_player_name(player_name)
        player_df = predictor.all_years_game_stats[
            predictor.all_years_game_stats["PLAYER_NAME"] == norm_name
        ].sort_values("GAME_DATE")

        if player_df.empty:
            return jsonify({"error": f"Player not found: {player_name}"})

        latest_team = str(player_df.iloc[-1]["TEAM_ABBREVIATION"])

        # Build future schedule lazily (first call only)
        if predictor.future_games_df is None:
            predictor.future_games_df = predictor.build_future_schedule_df()

        team_games = predictor.future_games_df[
            predictor.future_games_df["TEAM"].str.upper() == latest_team.upper()
        ].sort_values("GAME_DATE")

        games = []
        for _, row in team_games.iterrows():
            date_str = pd.Timestamp(row["GAME_DATE"]).strftime("%Y-%m-%d")
            display_date = pd.Timestamp(row["GAME_DATE"]).strftime("%b %d")
            matchup = f"vs. {row['OPPONENT']}" if bool(row["HOME"]) else f"@ {row['OPPONENT']}"
            games.append({"date": date_str, "label": f"{display_date}  {matchup}"})

        return jsonify(games)
    except Exception as e:
        import traceback
        traceback.print_exc()   # visible in the Flask server console
        return jsonify({"error": str(e)})


# ── NBA Predictor ─────────────────────────────────────────────────────────────

@app.route("/nba-predictor")
def nba_predictor():
    return render_template("nba_predictor_index.html")


@app.route("/loading")
def loading():
    return render_template("loading.html")


@app.route("/ready")
def ready():
    return jsonify(
        {
            "ready": _game_predictor is not None and _season_predictor is not None,
            "error": _warmup_error,
        }
    )


@app.route("/nba-predictor/result", methods=["POST"])
def nba_predictor_result():
    model_choice = request.form.get("model_choice", "1")
    player_name = request.form.get("player_name", "").strip()

    if not player_name:
        return render_template("error.html", message="Player name is required.")

    if model_choice == "1":
        game_date = request.form.get("game_date", "").strip()
        if not game_date:
            return render_template("error.html", message="Game date is required for game-level prediction.")
        try:
            predictor = _get_game_predictor()
            result = predictor.predict_player_points_for_date(player_name, game_date)
        except Exception as e:
            return render_template("error.html", message=str(e))
        if result is None:
            return render_template(
                "error.html",
                message=f"No game found for {player_name!r} on {game_date}. Check the date and try again.",
            )
        return render_template(
            "nba_predictor_result.html",
            model="game",
            player_name=player_name.title(),
            game_date=game_date,
            result=result,
        )

    else:
        try:
            sp = _get_season_predictor()
            result = sp.predict_next_season(player_name)
        except ValueError as e:
            return render_template("error.html", message=str(e))
        except Exception as e:
            return render_template("error.html", message=str(e))
        return render_template(
            "nba_predictor_result.html",
            model="season",
            player_name=player_name.title(),
            result=result,
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
