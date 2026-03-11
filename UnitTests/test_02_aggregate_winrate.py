import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Training.preprocessing.feature_engineering import aggregate_player_games


class TestAggregateWinrate(unittest.TestCase):

    def test_winrate_computed_correctly(self):
        df = pd.DataFrame({
            'kda': [2.0, 3.0, 4.0, 5.0],
            'kills': [1, 2, 3, 4],
            'deaths': [1, 1, 1, 1],
            'assists': [1, 1, 1, 1],
            'cs_per_min': [6.0, 7.0, 8.0, 9.0],
            'damage_per_min': [500.0, 600.0, 700.0, 800.0],
            'gold_per_min': [300.0, 350.0, 400.0, 450.0],
            'deaths_per_min': [0.05, 0.05, 0.05, 0.05],
            'vision_per_min': [1.0, 1.2, 1.4, 1.6],
            'damage_taken_per_min': [200.0, 220.0, 240.0, 260.0],
            'win': [1, 1, 0, 0],
            'role': ['MID', 'MID', 'MID', 'MID'],
            'championName': ['Zed', 'Zed', 'Zed', 'Zed'],
            'gameDuration': [1800, 1800, 1800, 1800],
        })
        result = aggregate_player_games(df)
        self.assertAlmostEqual(result['winrate'], 0.5)


if __name__ == '__main__':
    unittest.main()
