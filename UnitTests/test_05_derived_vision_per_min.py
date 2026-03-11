import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Training.preprocessing.feature_engineering import compute_derived_features


class TestDerivedVisionPerMin(unittest.TestCase):

    def test_vision_per_min_computed(self):
        df = pd.DataFrame({
            'visionScore': [60.0],
            'totalDamageTaken': [12000.0],
            'timePlayed': [1800.0],
        })
        result = compute_derived_features(df)
        self.assertAlmostEqual(result['vision_per_min'].iloc[0], 2.0)


if __name__ == '__main__':
    unittest.main()
