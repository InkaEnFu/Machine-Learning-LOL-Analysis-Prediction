"""
Unified training script for all LoL rank prediction models.
Trains 10 different ML models, compares them, and selects the best one.
"""

import sys
import os
import json
import pickle
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import clone

warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Training.preprocessing.cleaner import load_and_clean
from Training.preprocessing.feature_engineering import (
    build_player_dataset, AGGREGATED_FEATURE_NAMES, TIER_ORDER,
)

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed, skipping XGBoost model")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed, skipping LightGBM model")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'lol_rank_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def get_models():
    """Returns dictionary of all models to train."""
    models = {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'model': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
            'description': 'Lineární klasifikátor používající logistickou funkci'
        },
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'description': 'Ensemble stromů s náhodným výběrem features'
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            ),
            'description': 'Sklearn implementace gradient boostingu'
        },
        'extra_trees': {
            'name': 'Extra Trees',
            'model': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'description': 'Extremely Randomized Trees'
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'model': KNeighborsClassifier(
                n_neighbors=15, weights='distance', metric='minkowski', n_jobs=-1
            ),
            'description': 'Klasifikace podle nejbližších sousedů'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'model': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
            'description': 'Support Vector Classifier s RBF kernelem'
        },
        'mlp': {
            'name': 'Neural Network (MLP)',
            'model': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                alpha=0.001, max_iter=500, random_state=42, early_stopping=True
            ),
            'description': 'Multi-Layer Perceptron neuronová síť'
        },
    }
    
    if HAS_XGBOOST:
        models['xgboost'] = {
            'name': 'XGBoost',
            'model': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='mlogloss', verbosity=0
            ),
            'description': 'Gradient boosting s optimalizovanou implementací'
        }
    
    if HAS_LIGHTGBM:
        models['lightgbm'] = {
            'name': 'LightGBM',
            'model': LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                num_leaves=31, random_state=42, verbose=-1, force_col_wise=True
            ),
            'description': 'Rychlý gradient boosting od Microsoftu'
        }
    
    return models


def compute_rank_stats(player_df, feature_names):
    """Computes statistics for each rank tier."""
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


def train_all_models():
    """Main training function that trains all models."""
    print("=" * 70)
    print("LOL RANK PREDICTION - MULTI-MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Načítám data...")
    raw_df = load_and_clean(DATASET_PATH)
    print(f"      Surová data: {len(raw_df)} řádků")

    print("\n[2/5] Vytvářím dataset na úrovni hráčů...")
    player_df = build_player_dataset(raw_df)
    print(f"      Počet hráčů: {len(player_df)}")
    print(f"      Distribuce tierů:")
    for tier, count in player_df['tier'].value_counts().sort_index().items():
        print(f"        {tier}: {count}")

    # Prepare features
    feature_names = [f for f in AGGREGATED_FEATURE_NAMES if f in player_df.columns]
    X = player_df[feature_names].values
    y = player_df['tier_encoded'].values
    groups = player_df['puuid'].values

    print(f"\n      Počet features: {len(feature_names)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Setup cross-validation
    gkf = GroupKFold(n_splits=5)
    
    # Get models
    models = get_models()
    print(f"\n[3/5] Trénuji {len(models)} modelů...")
    
    results = []
    
    for short_name, config in models.items():
        print(f"\n      Training: {config['name']}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            # Cross-validation
            scores = cross_val_score(
                config['model'], X_scaled, y,
                cv=gkf, groups=groups, scoring='accuracy'
            )
            cv_accuracy = scores.mean()
            cv_std = scores.std()
            
            # Train final model
            model = clone(config['model'])
            model.fit(X_scaled, y)
            
            # F1 score on first fold
            gkf_split = list(gkf.split(X_scaled, y, groups))
            train_idx, test_idx = gkf_split[0]
            fold_model = clone(config['model'])
            fold_model.fit(X_scaled[train_idx], y[train_idx])
            y_pred = fold_model.predict(X_scaled[test_idx])
            f1 = f1_score(y[test_idx], y_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            print(f"CV Accuracy: {cv_accuracy:.4f} (+/- {cv_std:.4f}) [F1: {f1:.4f}] [{training_time:.1f}s]")
            
            results.append({
                'short_name': short_name,
                'name': config['name'],
                'description': config['description'],
                'cv_accuracy': cv_accuracy,
                'cv_std': cv_std,
                'f1_score': f1,
                'training_time': training_time,
                'model': model
            })
            
        except Exception as e:
            print(f"CHYBA: {e}")
    
    # Sort by accuracy
    results.sort(key=lambda x: x['cv_accuracy'], reverse=True)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("POROVNÁNÍ MODELŮ - SEŘAZENO PODLE CV ACCURACY")
    print("=" * 70)
    print(f"{'Rank':<5} {'Model':<25} {'CV Accuracy':<15} {'F1 Score':<12} {'Time':<8}")
    print("-" * 70)
    
    for i, r in enumerate(results, 1):
        marker = " ★ BEST" if i == 1 else ""
        print(f"{i:<5} {r['name']:<25} {r['cv_accuracy']:.4f} (+/- {r['cv_std']:.4f})  {r['f1_score']:.4f}       {r['training_time']:.1f}s{marker}")
    
    print("=" * 70)
    
    # Save all models
    print("\n[4/5] Ukládám modely...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Compute rank stats
    rank_stats = compute_rank_stats(player_df, feature_names)
    
    # Save scaler and rank_stats (once)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'rank_stats.pkl'), 'wb') as f:
        pickle.dump(rank_stats, f)
    
    for i, r in enumerate(results):
        is_best = (i == 0)
        short_name = r['short_name']
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'model_{short_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(r['model'], f)
        
        # Save metadata
        meta = {
            'model_name': r['name'],
            'short_name': short_name,
            'description': r['description'],
            'feature_names': feature_names,
            'tier_order': TIER_ORDER,
            'cv_accuracy': float(r['cv_accuracy']),
            'cv_std': float(r['cv_std']),
            'f1_score': float(r['f1_score']),
            'training_time': float(r['training_time']),
            'num_players': len(player_df),
            'timestamp': datetime.now().isoformat()
        }
        
        meta_path = os.path.join(MODELS_DIR, f'model_{short_name}_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        # If best, also save as primary model
        if is_best:
            with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
                pickle.dump(r['model'], f)
            with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"      ✓ model_{short_name}.pkl" + (" [HLAVNÍ MODEL]" if is_best else ""))
    
    # Save comparison
    comparison = {
        'ranking': [
            {
                'rank': i + 1,
                'model_name': r['name'],
                'short_name': r['short_name'],
                'cv_accuracy': float(r['cv_accuracy']),
                'cv_std': float(r['cv_std']),
                'f1_score': float(r['f1_score']),
                'description': r['description']
            }
            for i, r in enumerate(results)
        ],
        'best_model': results[0]['short_name'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(MODELS_DIR, 'models_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print("      ✓ models_comparison.json")
    
    # Print best model details
    best = results[0]
    print(f"\n[5/5] HOTOVO!")
    print("=" * 70)
    print(f"NEJLEPŠÍ MODEL: {best['name']}")
    print(f"CV Accuracy: {best['cv_accuracy']:.4f} (+/- {best['cv_std']:.4f})")
    print(f"F1 Score: {best['f1_score']:.4f}")
    print(f"\nHlavní model uložen jako: model.pkl")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    train_all_models()
