import pandas as pd
import numpy as np


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['puuid', 'tier'])
    df = df[df['tier'].isin(['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND'])]

    numeric_cols = [
        'kills', 'deaths', 'assists', 'totalMinionsKilled', 'neutralMinionsKilled',
        'visionScore', 'totalDamageDealtToChampions', 'totalDamageTaken',
        'goldEarned', 'timePlayed', 'kda', 'cs_per_min', 'damage_per_min',
        'gold_per_min', 'deaths_per_min', 'gameDuration',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['kills', 'deaths', 'assists', 'timePlayed'])
    df = df[df['timePlayed'] > 300]
    return df
