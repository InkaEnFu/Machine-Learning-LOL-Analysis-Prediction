import pandas as pd
from backend.services.riot_api import RiotAPIService, RiotAPIError
from backend.services.predictor import predict, TIER_ORDER
from Training.preprocessing.feature_engineering import aggregate_player_games

RANK_VALUES = {tier: i + 1 for i, tier in enumerate(TIER_ORDER)}


def analyze_live_game(game_name, tag_line, region='euw'):
    riot = RiotAPIService(region=region)

    account = riot.get_account(game_name, tag_line)
    puuid = account['puuid']

    active_game = riot.get_active_game(puuid)
    if active_game is None:
        return {'in_game': False}

    champion_map = riot.get_champion_id_map()
    version = riot.get_ddragon_version()

    participants = active_game.get('participants', [])

    blue_team = []
    red_team = []
    searched_team = None

    for p in participants:
        p_puuid = p.get('puuid', '')
        champ_id = p.get('championId', 0)
        team_id = p.get('teamId', 100)
        champ_name = champion_map.get(champ_id, 'Unknown')
        champ_img = (
            f"https://ddragon.leagueoflegends.com/cdn/{version}"
            f"/img/champion/{champ_name}.png"
        )

        try:
            acc = riot.get_account_by_puuid(p_puuid)
            name = acc.get('gameName', '???')
            tag = acc.get('tagLine', '')
        except Exception:
            name = '???'
            tag = ''

        matches = riot.get_player_matches_by_puuid(p_puuid, num_matches=5)

        info = {
            'summoner_name': name,
            'summoner_tag': tag,
            'champion_name': champ_name,
            'champion_image': champ_img,
            'is_searched_player': p_puuid == puuid,
        }

        if matches and len(matches) >= 2:
            df = pd.DataFrame(matches)
            agg = aggregate_player_games(df)

            wins = sum(1 for m in matches if m['win'] == 1)
            info['winrate'] = round(wins / len(matches) * 100, 1)
            info['games'] = len(matches)

            try:
                pred = predict(agg)
                info['predicted_rank'] = pred['predicted_rank']
                info['confidence'] = round(pred['confidence'], 3)
            except Exception:
                info['predicted_rank'] = None
                info['confidence'] = 0
        else:
            info['winrate'] = None
            info['games'] = 0
            info['predicted_rank'] = None
            info['confidence'] = 0

        if team_id == 100:
            blue_team.append(info)
        else:
            red_team.append(info)

        if p_puuid == puuid:
            searched_team = 'blue' if team_id == 100 else 'red'

    blue_score = _team_score(blue_team)
    red_score = _team_score(red_team)

    total = blue_score + red_score
    if total > 0:
        blue_pct = round(blue_score / total * 100, 1)
        red_pct = round(100 - blue_pct, 1)
    else:
        blue_pct = 50.0
        red_pct = 50.0

    return {
        'in_game': True,
        'game_mode': active_game.get('gameMode', 'CLASSIC'),
        'queue_id': active_game.get('gameQueueConfigId', 0),
        'blue_team': blue_team,
        'red_team': red_team,
        'searched_player_team': searched_team,
        'prediction': {
            'blue_win_probability': blue_pct,
            'red_win_probability': red_pct,
            'predicted_winner': 'blue' if blue_pct > red_pct else 'red',
        },
    }


def _team_score(team):
    scores = []
    for p in team:
        rank = p.get('predicted_rank')
        val = RANK_VALUES.get(rank, 3.5)

        wr = p.get('winrate')
        if wr is not None:
            wr_factor = wr / 50.0
        else:
            wr_factor = 1.0

        scores.append(val * wr_factor)

    return sum(scores) / max(len(scores), 1)
