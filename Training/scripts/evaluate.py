import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Training.preprocessing.cleaner import load_and_clean
from Training.preprocessing.feature_engineering import build_player_dataset, TIER_ORDER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'lol_rank_dataset.csv')


def evaluate():
    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'model_meta.json')) as f:
        meta = json.load(f)

    feature_names = meta['feature_names']
    print(f"Model: {meta['model_name']}")
    print(f"Features: {len(feature_names)}")

    raw_df = load_and_clean(DATASET_PATH)
    player_df = build_player_dataset(raw_df)

    X = player_df[feature_names].values
    y = player_df['tier_encoded'].values
    groups = player_df['puuid'].values

    X_scaled = scaler.transform(X)

    print(f"\n=== Training Data Performance ===")
    y_pred_train = model.predict(X_scaled)
    print(f"Accuracy: {accuracy_score(y, y_pred_train):.4f}")

    print(f"\n=== Cross-Validation Performance ===")
    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups)):
        model_clone = clone(model)
        model_clone.fit(X_scaled[train_idx], y[train_idx])
        all_preds[test_idx] = model_clone.predict(X_scaled[test_idx])
        fold_acc = accuracy_score(y[test_idx], all_preds[test_idx])
        print(f"  Fold {fold + 1}: {fold_acc:.4f}")

    print(f"\nOverall CV Accuracy: {accuracy_score(y, all_preds):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, all_preds, target_names=TIER_ORDER))
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, all_preds)
    print(pd.DataFrame(cm, index=TIER_ORDER, columns=TIER_ORDER))


if __name__ == '__main__':
    evaluate()
