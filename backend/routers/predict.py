import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.models.schemas import PredictRequest
from backend.services.riot_api import RiotAPIService, RiotAPIError
from backend.services.predictor import predict
from backend.services.analytics import compute_comparison, compute_all_ranks_distance
from backend.services.live_game import analyze_live_game
from backend.services.champion_recommender import recommend_champions
from Training.preprocessing.feature_engineering import aggregate_player_games

router = APIRouter()


@router.post("/predict")
def predict_rank(request: PredictRequest):
    try:
        riot_service = RiotAPIService(region=request.region)
        player_data = riot_service.get_player_matches(request.game_name, request.tag_line)
    except RiotAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))

    matches = player_data['matches']
    matches_df = pd.DataFrame(matches)
    player_agg = aggregate_player_games(matches_df)

    try:
        prediction = predict(player_agg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    comparison = compute_comparison(player_agg, prediction['predicted_rank'])
    all_ranks = compute_all_ranks_distance(player_agg)

    match_summaries = []
    for m in matches:
        match_summaries.append({
            'matchId': m['matchId'],
            'champion': m['championName'],
            'role': m['role'],
            'kills': m['kills'],
            'deaths': m['deaths'],
            'assists': m['assists'],
            'kda': m['kda'],
            'cs_per_min': m['cs_per_min'],
            'damage_per_min': m['damage_per_min'],
            'gold_per_min': m['gold_per_min'],
            'vision_per_min': m['vision_per_min'],
            'win': m['win'],
        })

    wins = sum(1 for m in matches if m['win'] == 1)
    losses = len(matches) - wins

    champion_data = recommend_champions(matches, top_n=3)

    return {
        'summoner': {
            'game_name': player_data['game_name'],
            'tag_line': player_data['tag_line'],
        },
        'prediction': prediction,
        'player_stats': {
            k: round(float(v), 4)
            for k, v in player_agg.items()
            if not isinstance(v, str)
        },
        'comparison': comparison,
        'all_ranks_distance': all_ranks,
        'matches': match_summaries,
        'record': {'wins': wins, 'losses': losses, 'total': len(matches)},
        'champion_recommendations': champion_data,
    }


@router.post("/live-game")
def live_game(request: PredictRequest):
    try:
        result = analyze_live_game(request.game_name, request.tag_line, request.region)
    except RiotAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result
