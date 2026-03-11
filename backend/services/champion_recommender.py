import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

_champion_profiles = None
_profile_scaler = None
_dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'Training', 'datasets', 'lol_rank_dataset.csv',
)

PROFILE_FEATURES = [
    'kda', 'cs_per_min', 'damage_per_min', 'gold_per_min',
    'deaths_per_min', 'kills', 'deaths', 'assists',
]

MIN_GAMES_FOR_PROFILE = 20


def _build_champion_profiles():
    global _champion_profiles, _profile_scaler

    if _champion_profiles is not None:
        return

    df = pd.read_csv(_dataset_path, low_memory=False)

    champ_stats = df.groupby('championName').agg(
        games=('win', 'count'),
        avg_winrate=('win', 'mean'),
        **{f'avg_{f}': (f, 'mean') for f in PROFILE_FEATURES},
    ).reset_index()

    champ_stats = champ_stats[champ_stats['games'] >= MIN_GAMES_FOR_PROFILE].copy()

    feature_cols = [f'avg_{f}' for f in PROFILE_FEATURES]
    _profile_scaler = StandardScaler()
    champ_stats[feature_cols] = _profile_scaler.fit_transform(champ_stats[feature_cols])

    _champion_profiles = champ_stats


def recommend_champions(player_matches, top_n=3):
    _build_champion_profiles()

    if not player_matches or len(player_matches) < 2:
        return {'player_champions': [], 'recommendations': []}

    champ_data = {}
    for m in player_matches:
        name = m.get('championName', '')
        if not name:
            continue
        if name not in champ_data:
            champ_data[name] = {'wins': 0, 'games': 0, 'stats': []}
        champ_data[name]['games'] += 1
        champ_data[name]['wins'] += m.get('win', 0)
        champ_data[name]['stats'].append(
            [m.get(f, 0) for f in PROFILE_FEATURES]
        )

    player_champions = []
    for name, data in sorted(champ_data.items(), key=lambda x: -x[1]['games']):
        wr = round(data['wins'] / data['games'] * 100, 1) if data['games'] > 0 else 0
        player_champions.append({
            'champion': name,
            'games': data['games'],
            'wins': data['wins'],
            'winrate': wr,
        })

    all_stats = []
    for data in champ_data.values():
        all_stats.extend(data['stats'])

    player_vec = np.mean(all_stats, axis=0).reshape(1, -1)
    feature_cols = [f'avg_{f}' for f in PROFILE_FEATURES]
    player_df = pd.DataFrame(player_vec, columns=feature_cols)
    player_vec_scaled = _profile_scaler.transform(player_df)

    played_names = set(champ_data.keys())

    candidates = _champion_profiles[
        ~_champion_profiles['championName'].isin(played_names)
    ].copy()

    if candidates.empty:
        return {'player_champions': player_champions, 'recommendations': []}

    candidate_vecs = candidates[feature_cols].values
    similarities = cosine_similarity(player_vec_scaled, candidate_vecs)[0]

    candidates = candidates.copy()
    candidates['similarity'] = similarities

    candidates['score'] = (
        candidates['similarity'] * 0.6
        + candidates['avg_winrate'] * 0.4
    )

    top = candidates.nlargest(top_n, 'score')

    recommendations = []
    for _, row in top.iterrows():
        recommendations.append({
            'champion': row['championName'],
            'match_score': round(float(row['similarity']) * 100, 1),
            'dataset_winrate': round(float(row['avg_winrate']) * 100, 1) if row['avg_winrate'] <= 1 else round(float(row['avg_winrate']), 1),
            'dataset_games': int(row['games']),
        })

    return {
        'player_champions': player_champions,
        'recommendations': recommendations,
    }
