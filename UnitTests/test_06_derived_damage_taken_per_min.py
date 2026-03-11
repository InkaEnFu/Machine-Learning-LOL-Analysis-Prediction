import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Training.preprocessing.feature_engineering import compute_derived_features


class TestDerivedDamageTakenPerMin(unittest.TestCase):

    def test_damage_taken_per_min_computed(self):
        df = pd.DataFrame({
            'visionScore': [30.0],
            'totalDamageTaken': [9000.0],
            'timePlayed': [1800.0],
        })
        result = compute_derived_features(df)
        self.assertAlmostEqual(result['damage_taken_per_min'].iloc[0], 300.0)


if __name__ == '__main__':
    unittest.main()
