import pandas as pd
import numpy as np


PER_GAME_FEATURES = [
    'kda', 'cs_per_min', 'damage_per_min', 'gold_per_min',
    'deaths_per_min', 'vision_per_min', 'damage_taken_per_min',
    'kills', 'deaths', 'assists',
]

MEAN_FEATURES = [
    'kills', 'deaths', 'assists', 'kda',
    'cs_per_min', 'damage_per_min', 'gold_per_min', 'deaths_per_min',
    'vision_per_min', 'damage_taken_per_min',
]

STD_FEATURES = [
    'kda', 'cs_per_min', 'damage_per_min', 'gold_per_min',
    'deaths_per_min', 'vision_per_min', 'damage_taken_per_min',
    'kills', 'deaths', 'assists',
]

MEDIAN_FEATURES = [
    'kda', 'cs_per_min', 'damage_per_min', 'gold_per_min',
    'deaths_per_min', 'vision_per_min',
]

AGGREGATED_FEATURE_NAMES = (
    [f'{f}_mean' for f in MEAN_FEATURES] +
    [f'{f}_std' for f in STD_FEATURES] +
    [f'{f}_median' for f in MEDIAN_FEATURES] +
    ['winrate', 'unique_roles', 'unique_champions', 'avg_game_duration']
)

TIER_ORDER = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND']
TIER_TO_INT = {t: i for i, t in enumerate(TIER_ORDER)}


def compute_derived_features(df):
    df = df.copy()
    game_minutes = df['timePlayed'] / 60.0
    game_minutes = game_minutes.replace(0, np.nan)
    df['vision_per_min'] = df['visionScore'] / game_minutes
    df['damage_taken_per_min'] = df['totalDamageTaken'] / game_minutes
    return df


def aggregate_player_games(games_df):
    result = {}
    for f in MEAN_FEATURES:
        if f in games_df.columns:
            result[f'{f}_mean'] = games_df[f].mean()
        else:
            result[f'{f}_mean'] = 0
    for f in STD_FEATURES:
        if f in games_df.columns:
            val = games_df[f].std()
            result[f'{f}_std'] = val if pd.notna(val) else 0
        else:
            result[f'{f}_std'] = 0
    for f in MEDIAN_FEATURES:
        if f in games_df.columns:
            result[f'{f}_median'] = games_df[f].median()
        else:
            result[f'{f}_median'] = 0
    result['winrate'] = games_df['win'].mean() if 'win' in games_df.columns else 0
    result['unique_roles'] = games_df['role'].nunique() if 'role' in games_df.columns else 1
    result['unique_champions'] = games_df['championName'].nunique() if 'championName' in games_df.columns else 1
    if 'gameDuration' in games_df.columns:
        result['avg_game_duration'] = games_df['gameDuration'].mean()
    elif 'timePlayed' in games_df.columns:
        result['avg_game_duration'] = games_df['timePlayed'].mean()
    else:
        result['avg_game_duration'] = 0
    return result


def build_player_dataset(df):
    df = compute_derived_features(df)
    players = []
    for puuid, group in df.groupby('puuid'):
        agg = aggregate_player_games(group)
        agg['puuid'] = puuid
        agg['tier'] = group['tier'].iloc[0]
        agg['tier_encoded'] = TIER_TO_INT.get(group['tier'].iloc[0], -1)
        players.append(agg)
    result = pd.DataFrame(players)
    result = result.fillna(0)
    return result
