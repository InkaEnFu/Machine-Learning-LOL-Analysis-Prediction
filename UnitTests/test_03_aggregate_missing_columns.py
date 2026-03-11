import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Training.preprocessing.feature_engineering import aggregate_player_games


class TestAggregateMissingColumns(unittest.TestCase):

    def test_missing_columns_default_to_zero(self):
        df = pd.DataFrame({
            'kda': [3.0],
            'win': [1],
            'role': ['TOP'],
            'championName': ['Garen'],
            'gameDuration': [1800],
        })
        result = aggregate_player_games(df)
        self.assertEqual(result['kills_mean'], 0)
        self.assertEqual(result['cs_per_min_mean'], 0)
        self.assertEqual(result['vision_per_min_mean'], 0)


if __name__ == '__main__':
    unittest.main()
