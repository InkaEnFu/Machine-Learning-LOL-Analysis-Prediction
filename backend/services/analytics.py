import numpy as np
from scipy import stats as scipy_stats
from backend.services.predictor import get_rank_stats

METRIC_CATEGORIES = {
    'combat': {
        'kda_mean': 'KDA',
        'kills_mean': 'Kills',
        'deaths_mean': 'Deaths',
        'assists_mean': 'Assists',
        'damage_per_min_mean': 'Damage/min',
    },
    'farming': {
        'cs_per_min_mean': 'CS/min',
    },
    'vision': {
        'vision_per_min_mean': 'Vision/min',
    },
    'survivability': {
        'deaths_per_min_mean': 'Deaths/min',
        'damage_taken_per_min_mean': 'Dmg Taken/min',
    },
    'economy': {
        'gold_per_min_mean': 'Gold/min',
    },
}

INVERTED_METRICS = {'deaths_mean', 'deaths_per_min_mean', 'damage_taken_per_min_mean'}


def compute_comparison(player_stats, predicted_rank):
    rank_stats = get_rank_stats()

    if predicted_rank not in rank_stats:
        return {}

    tier_stats = rank_stats[predicted_rank]
    tier_mean = tier_stats['mean']
    tier_std = tier_stats['std']

    comparison = {}

    for category, metrics in METRIC_CATEGORIES.items():
        category_data = []
        for metric_key, metric_label in metrics.items():
            if metric_key not in tier_mean:
                continue

            player_val = player_stats.get(metric_key, 0)
            avg_val = tier_mean.get(metric_key, 0)
            std_val = tier_std.get(metric_key, 1)

            diff_abs = player_val - avg_val
            diff_pct = (diff_abs / avg_val * 100) if avg_val != 0 else 0
            z_score = (player_val - avg_val) / std_val if std_val > 0 else 0
            percentile = float(scipy_stats.norm.cdf(z_score) * 100)

            is_inverted = metric_key in INVERTED_METRICS
            if is_inverted:
                above_average = bool(player_val < avg_val)
            else:
                above_average = bool(player_val > avg_val)

            category_data.append({
                'metric': metric_key,
                'label': metric_label,
                'player_value': round(float(player_val), 4),
                'rank_average': round(float(avg_val), 4),
                'diff_absolute': round(float(diff_abs), 4),
                'diff_percent': round(float(diff_pct), 2),
                'z_score': round(float(z_score), 4),
                'percentile': round(percentile, 2),
                'above_average': above_average,
                'inverted': is_inverted,
            })

        comparison[category] = category_data

    return comparison


def compute_all_ranks_distance(player_stats):
    rank_stats = get_rank_stats()
    result = {}

    for tier, tier_data in rank_stats.items():
        tier_mean = tier_data['mean']
        distances = []
        for key, val in player_stats.items():
            if key in tier_mean and tier_data['std'].get(key, 0) > 0:
                z = (val - tier_mean[key]) / tier_data['std'][key]
                distances.append(z ** 2)
        result[tier] = round(float(np.sqrt(np.mean(distances))) if distances else 0, 4)

    return result
