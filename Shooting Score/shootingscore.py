from flask import Flask, render_template, request, jsonify
import basic_stat_grab
import zcalc
from scorefinal import calculate_composite_z_score
from nba_api.stats.static import players

app = Flask(__name__, template_folder='shooting score html')

# 1. Make '/' route render home.html
@app.route('/')
def home():
    return render_template('home.html')

# 2. Make a new route for your shooting score index page
@app.route('/shooting-score')
def shooting_score():
    return render_template('index.html')

@app.route('/search')
def search():
    # your existing code
    query = request.args.get('q', '')
    player_names = search_player_names(query)
    return jsonify(player_names)

def search_player_names(query):
    # your existing code
    player_dict = players.get_players()
    normalized_query = basic_stat_grab.normalize_string(query)
    matching_players = []
    for player in player_dict:
        normalized_player_name = basic_stat_grab.normalize_string(player['full_name'])
        if normalized_query in normalized_player_name:
            matching_players.append(player['full_name'])
    return matching_players

@app.route('/seasons')
def seasons():
    player_name = request.args.get('player_name', '')
    if not player_name:
        return jsonify([])
    try:
        player_id = basic_stat_grab.get_player_id(player_name)
        all_seasons = basic_stat_grab.get_all_seasons_for_player(player_id)
        return jsonify(all_seasons)
    except Exception as e:
        app.logger.error(f"Error fetching seasons for '{player_name}': {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/result', methods=['POST'])
def result():
    # your existing code
    player_name = request.form['player']
    season_id = request.form['season_dropdown']
    result = calculate_composite_z_score(player_name, season_id)
    if result is None:
        return render_template('error.html', message="Player not found or no data available.")
    else:
        return render_template('result.html', player_name=player_name, result=result)

if __name__ == '__main__':
    app.run(debug=True)
