import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Training.preprocessing.feature_engineering import aggregate_player_games


class TestAggregateUniqueStats(unittest.TestCase):

    def test_unique_champions_and_roles(self):
        df = pd.DataFrame({
            'kda': [2.0, 3.0],
            'kills': [3, 5],
            'deaths': [2, 2],
            'assists': [6, 8],
            'cs_per_min': [6.0, 8.0],
            'damage_per_min': [500.0, 700.0],
            'gold_per_min': [300.0, 400.0],
            'deaths_per_min': [0.1, 0.1],
            'vision_per_min': [1.0, 1.5],
            'damage_taken_per_min': [200.0, 250.0],
            'win': [1, 0],
            'role': ['TOP', 'MID'],
            'championName': ['Garen', 'Zed'],
            'gameDuration': [1800, 1800],
        })
        result = aggregate_player_games(df)
        self.assertEqual(result['unique_roles'], 2)
        self.assertEqual(result['unique_champions'], 2)


if __name__ == '__main__':
    unittest.main()
