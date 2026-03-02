import os
import pickle
import json
import numpy as np
from backend.config import MODELS_DIR

_model = None
_scaler = None
_meta = None
_rank_stats = None

TIER_ORDER = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND']


def _load_artifacts():
    global _model, _scaler, _meta, _rank_stats

    if _model is not None:
        return

    model_path = os.path.join(MODELS_DIR, 'model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
    stats_path = os.path.join(MODELS_DIR, 'rank_stats.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not trained yet. Run: python Training/scripts/train.py"
        )

    with open(model_path, 'rb') as f:
        _model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        _scaler = pickle.load(f)
    with open(meta_path) as f:
        _meta = json.load(f)
    with open(stats_path, 'rb') as f:
        _rank_stats = pickle.load(f)


def get_feature_names():
    _load_artifacts()
    return _meta['feature_names']


def get_rank_stats():
    _load_artifacts()
    return _rank_stats


def get_model_meta():
    _load_artifacts()
    return _meta


def predict(features_dict):
    _load_artifacts()

    feature_names = _meta['feature_names']
    X = np.array([[features_dict.get(f, 0) for f in feature_names]])
    X_scaled = _scaler.transform(X)

    prediction = _model.predict(X_scaled)[0]
    predicted_tier = TIER_ORDER[int(prediction)]

    probabilities = {}
    if hasattr(_model, 'predict_proba'):
        proba = _model.predict_proba(X_scaled)[0]
        for i, tier in enumerate(TIER_ORDER):
            if i < len(proba):
                probabilities[tier] = round(float(proba[i]), 4)

    confidence = probabilities.get(predicted_tier, 0)

    feature_importance = {}
    if hasattr(_model, 'feature_importances_'):
        importances = _model.feature_importances_
        for name, imp in zip(feature_names, importances):
            feature_importance[name] = round(float(imp), 4)
    elif hasattr(_model, 'coef_'):
        avg_coef = np.mean(np.abs(_model.coef_), axis=0)
        for name, imp in zip(feature_names, avg_coef):
            feature_importance[name] = round(float(imp), 4)

    return {
        'predicted_rank': predicted_tier,
        'confidence': round(confidence, 4),
        'rank_probabilities': probabilities,
        'feature_importance': feature_importance,
    }
