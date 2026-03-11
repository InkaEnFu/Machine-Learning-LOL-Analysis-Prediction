"""
Rank Progression Prediction Service

Estimates how many games a player needs to reach the next tier,
based on their current rank, LP, winrate, and performance strength
relative to each tier.
"""

import math
from backend.services.predictor import get_rank_stats

TIER_ORDER = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND']
DIVISIONS = ['IV', 'III', 'II', 'I']
LP_PER_DIVISION = 100

AVG_LP_GAIN = 22
AVG_LP_LOSS = 18


def _tier_index(tier):
    try:
        return TIER_ORDER.index(tier.upper())
    except ValueError:
        return -1


def _division_index(div):
    try:
        return DIVISIONS.index(div.upper())
    except ValueError:
        return 0


def _lp_to_next_tier(current_tier, current_division, current_lp):
    """Calculate total LP needed to reach the next tier."""
    div_idx = _division_index(current_division)
    # LP remaining in current division
    lp_remaining = LP_PER_DIVISION - current_lp
    # Full divisions above current
    divisions_above = (len(DIVISIONS) - 1) - div_idx
    return lp_remaining + divisions_above * LP_PER_DIVISION


def _estimate_games(lp_needed, winrate):
    """Estimate games needed given LP to gain and a winrate."""
    if winrate < 0.45:
        return None  # Actively losing LP, can't climb
    net_lp_per_game = winrate * AVG_LP_GAIN - (1 - winrate) * AVG_LP_LOSS
    if net_lp_per_game <= 0:
        return None
    return math.ceil(lp_needed / net_lp_per_game)


def _compute_tier_strength(player_stats, tier, rank_stats):
    """
    Compute how well a player's stats match a given tier.
    Returns a 0-100 score (higher = stronger relative to that tier).
    """
    if tier not in rank_stats:
        return 50.0

    tier_data = rank_stats[tier]
    tier_mean = tier_data['mean']
    tier_std = tier_data['std']

    z_scores = []
    inverted = {'deaths_mean', 'deaths_per_min_mean', 'damage_taken_per_min_mean'}

    for key, val in player_stats.items():
        if key not in tier_mean or key not in tier_std:
            continue
        std = tier_std[key]
        if std <= 0:
            continue
        z = (val - tier_mean[key]) / std
        if key in inverted:
            z = -z  # lower is better for these
        z_scores.append(z)

    if not z_scores:
        return 50.0

    avg_z = sum(z_scores) / len(z_scores)
    # Convert z-score to 0-100 scale (sigmoid-like mapping)
    strength = 1 / (1 + math.exp(-avg_z * 0.8))
    return round(strength * 100, 1)


def compute_rank_progression(real_rank, player_stats, winrate):
    """
    Compute rank progression prediction.

    Args:
        real_rank: Dict with 'tier', 'rank' (division), 'lp', 'wins', 'losses'
        player_stats: Aggregated player stats dict
        winrate: Player's overall winrate (0-1)

    Returns:
        Dict with progression data for each tier above current.
    """
    if not real_rank:
        return None

    current_tier = real_rank['tier'].upper()
    current_div = real_rank.get('rank', 'IV')
    current_lp = real_rank.get('lp', 0)
    current_tier_idx = _tier_index(current_tier)

    if current_tier_idx < 0:
        return None

    rank_stats = get_rank_stats()

    # Compute strength against each tier
    tier_strengths = {}
    for tier in TIER_ORDER:
        tier_strengths[tier] = _compute_tier_strength(player_stats, tier, rank_stats)

    # Compute progression to each tier above current
    tiers_progression = []

    # LP to next tier from current position
    lp_to_next = _lp_to_next_tier(current_tier, current_div, current_lp)

    cumulative_lp = 0
    for i in range(current_tier_idx + 1, len(TIER_ORDER)):
        target_tier = TIER_ORDER[i]

        if i == current_tier_idx + 1:
            # First tier above: use actual LP distance
            cumulative_lp += lp_to_next
        else:
            # Each additional tier = 4 full divisions
            cumulative_lp += LP_PER_DIVISION * len(DIVISIONS)

        est_games = _estimate_games(cumulative_lp, winrate)

        tiers_progression.append({
            'tier': target_tier,
            'lp_needed': cumulative_lp,
            'estimated_games': est_games,
            'strength': tier_strengths.get(target_tier, 50.0),
        })

    return {
        'current_tier': current_tier,
        'current_division': current_div,
        'current_lp': current_lp,
        'winrate_percent': round(winrate * 100, 1),
        'tier_strengths': tier_strengths,
        'progression': tiers_progression,
    }
