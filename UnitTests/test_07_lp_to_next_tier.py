import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.rank_progression import _lp_to_next_tier


class TestLpToNextTier(unittest.TestCase):

    def test_lp_to_next_tier_from_div4(self):
        result = _lp_to_next_tier('GOLD', 'IV', 50)
        self.assertEqual(result, 350)

    def test_lp_to_next_tier_from_div1(self):
        result = _lp_to_next_tier('GOLD', 'I', 75)
        self.assertEqual(result, 25)


if __name__ == '__main__':
    unittest.main()
