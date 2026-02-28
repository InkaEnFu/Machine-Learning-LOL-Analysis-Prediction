import sys
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.base import clone

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Training.preprocessing.cleaner import load_and_clean
from Training.preprocessing.feature_engineering import (
    build_player_dataset, AGGREGATED_FEATURE_NAMES, TIER_ORDER,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'lol_rank_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def compute_rank_stats(player_df, feature_names):
    stats = {}
    for tier in TIER_ORDER:
        tier_data = player_df[player_df['tier'] == tier][feature_names]
        stats[tier] = {
            'mean': tier_data.mean().to_dict(),
            'std': tier_data.std().to_dict(),
            'median': tier_data.median().to_dict(),
            'count': len(tier_data),
        }
    return stats


def train():
    print("Loading and cleaning data...")
    raw_df = load_and_clean(DATASET_PATH)
    print(f"Raw data: {len(raw_df)} rows")

    print("Building player-level dataset...")
    player_df = build_player_dataset(raw_df)
    print(f"Players: {len(player_df)}")
    print(f"Tier distribution:\n{player_df['tier'].value_counts().sort_index()}")

    feature_names = [f for f in AGGREGATED_FEATURE_NAMES if f in player_df.columns]
    X = player_df[feature_names].values
    y = player_df['tier_encoded'].values
    groups = player_df['puuid'].values

    print(f"Features ({len(feature_names)}): {feature_names}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000, C=1.0)

    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(model, X_scaled, y, cv=gkf, groups=groups, scoring='accuracy')
    print(f"\nCV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("Training final model on all data...")
    model.fit(X_scaled, y)

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    rank_stats = compute_rank_stats(player_df, feature_names)
    with open(os.path.join(MODELS_DIR, 'rank_stats.pkl'), 'wb') as f:
        pickle.dump(rank_stats, f)

    meta = {
        'model_name': 'LogisticRegression',
        'feature_names': feature_names,
        'tier_order': TIER_ORDER,
        'cv_accuracy': float(scores.mean()),
        'num_players': len(player_df),
    }
    with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel artifacts saved to {MODELS_DIR}")

    gkf_split = list(gkf.split(X_scaled, y, groups))
    train_idx, test_idx = gkf_split[0]

    fold_model = clone(model)
    fold_model.fit(X_scaled[train_idx], y[train_idx])
    y_pred = fold_model.predict(X_scaled[test_idx])

    print(f"\nClassification Report (fold 1):")
    print(classification_report(y[test_idx], y_pred, target_names=TIER_ORDER))


if __name__ == '__main__':
    train()
