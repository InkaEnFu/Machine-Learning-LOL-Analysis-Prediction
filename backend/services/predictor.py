"""
Model predictor service for LoL rank prediction.
Supports loading the best model or any specific trained model.
"""

import os
import pickle
import json
from typing import Dict, Any, List, Optional
import numpy as np
from backend.config import MODELS_DIR

# Global cache for loaded artifacts
_model = None
_scaler = None
_meta = None
_rank_stats = None
_comparison = None
_loaded_model_name = None

TIER_ORDER = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND']


def _load_artifacts(model_name: Optional[str] = None):
    """
    Load model artifacts from disk.
    
    Args:
        model_name: Optional specific model to load (e.g., 'xgboost', 'random_forest').
                   If None, loads the best model (model.pkl).
    """
    global _model, _scaler, _meta, _rank_stats, _loaded_model_name

    # Skip if already loaded the same model
    if _model is not None and _loaded_model_name == model_name:
        return

    # Determine which model file to load
    if model_name:
        model_path = os.path.join(MODELS_DIR, f'model_{model_name}.pkl')
        meta_path = os.path.join(MODELS_DIR, f'model_{model_name}_meta.json')
    else:
        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    stats_path = os.path.join(MODELS_DIR, 'rank_stats.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run: python Training/scripts/train.py"
        )

    with open(model_path, 'rb') as f:
        _model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        _scaler = pickle.load(f)
    with open(meta_path) as f:
        _meta = json.load(f)
    with open(stats_path, 'rb') as f:
        _rank_stats = pickle.load(f)
    
    _loaded_model_name = model_name


def _load_comparison():
    """Load model comparison data."""
    global _comparison
    
    if _comparison is not None:
        return
    
    comparison_path = os.path.join(MODELS_DIR, 'models_comparison.json')
    if os.path.exists(comparison_path):
        with open(comparison_path, encoding='utf-8') as f:
            _comparison = json.load(f)
    else:
        _comparison = {}


def get_feature_names() -> List[str]:
    """Get list of feature names used by the model."""
    _load_artifacts()
    return _meta['feature_names']


def get_rank_stats() -> Dict:
    """Get rank statistics for each tier."""
    _load_artifacts()
    return _rank_stats


def get_model_meta() -> Dict:
    """Get metadata about the current model."""
    _load_artifacts()
    return _meta


def get_models_comparison() -> Dict:
    """
    Get comparison data for all trained models.
    Returns ranking, accuracy metrics, and best model info.
    """
    _load_comparison()
    return _comparison


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of all available trained models with their metrics.
    """
    _load_comparison()
    if _comparison and 'ranking' in _comparison:
        return _comparison['ranking']
    return []


def switch_model(model_name: str):
    """
    Switch to a different trained model.
    
    Args:
        model_name: Short name of the model (e.g., 'xgboost', 'random_forest', 'lightgbm')
    """
    global _model, _scaler, _meta, _rank_stats, _loaded_model_name
    
    # Reset cache to force reload
    _model = None
    _loaded_model_name = None
    
    # Load the specified model
    _load_artifacts(model_name)


def predict(features_dict: Dict[str, float], model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Predict rank for given features.
    
    Args:
        features_dict: Dictionary of feature values
        model_name: Optional specific model to use for prediction
        
    Returns:
        Dictionary with prediction results including:
        - predicted_rank: The predicted tier
        - confidence: Confidence score
        - rank_probabilities: Probabilities for each tier
        - feature_importance: Importance of each feature
        - model_used: Name of the model used
    """
    _load_artifacts(model_name)

    feature_names = _meta['feature_names']
    X = np.array([[features_dict.get(f, 0) for f in feature_names]])
    X_scaled = _scaler.transform(X)

    prediction = _model.predict(X_scaled)[0]
    predicted_tier = TIER_ORDER[int(prediction)]

    # Get probabilities if available
    probabilities = {}
    if hasattr(_model, 'predict_proba'):
        proba = _model.predict_proba(X_scaled)[0]
        for i, tier in enumerate(TIER_ORDER):
            if i < len(proba):
                probabilities[tier] = round(float(proba[i]), 4)

    confidence = probabilities.get(predicted_tier, 0)

    # Get feature importance
    feature_importance = {}
    if hasattr(_model, 'feature_importances_'):
        # Tree-based models (Random Forest, XGBoost, LightGBM, etc.)
        importances = _model.feature_importances_
        for name, imp in zip(feature_names, importances):
            feature_importance[name] = round(float(imp), 4)
    elif hasattr(_model, 'coef_'):
        # Linear models (Logistic Regression, SVM)
        avg_coef = np.mean(np.abs(_model.coef_), axis=0)
        for name, imp in zip(feature_names, avg_coef):
            feature_importance[name] = round(float(imp), 4)

    return {
        'predicted_rank': predicted_tier,
        'confidence': round(confidence, 4),
        'rank_probabilities': probabilities,
        'feature_importance': feature_importance,
        'model_used': _meta.get('model_name', 'Unknown'),
        'model_accuracy': _meta.get('cv_accuracy', 0),
    }
