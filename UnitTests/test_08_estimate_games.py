import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.rank_progression import _estimate_games


class TestEstimateGames(unittest.TestCase):

    def test_estimate_games_valid_winrate(self):
        lp_needed = 200
        winrate = 0.60
        net = winrate * 22 - (1 - winrate) * 18
        expected = math.ceil(lp_needed / net)
        result = _estimate_games(lp_needed, winrate)
        self.assertEqual(result, expected)

    def test_estimate_games_low_winrate_returns_none(self):
        result = _estimate_games(300, 0.40)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
