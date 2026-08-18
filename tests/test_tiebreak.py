"""Unit tests for Stage F AMAT deterministic center tie-breaker rule."""

import unittest
from src.candidate import Candidate
from src.tiebreak import compute_distance_to_center, apply_amat_tiebreaker


class TestTiebreak(unittest.TestCase):

    def test_distance_to_center(self):
        dist = compute_distance_to_center(500.0, 500.0, center=(500.0, 500.0))
        self.assertEqual(dist, 0.0)

        dist_345 = compute_distance_to_center(503.0, 504.0, center=(500.0, 500.0))
        self.assertEqual(dist_345, 5.0)

    def test_single_winner_no_tie(self):
        c1 = Candidate(x=100.0, y=100.0, zncc_score=0.90, scale=10.0, rotation=0.0, rank=1, composite_score=0.90)
        c2 = Candidate(x=500.0, y=500.0, zncc_score=0.80, scale=10.0, rotation=0.0, rank=2, composite_score=0.80)

        winner, tie_occurred = apply_amat_tiebreaker([c1, c2], tie_tolerance=1e-4)
        self.assertFalse(tie_occurred)
        self.assertEqual(winner.x, 100.0)
        self.assertEqual(winner.y, 100.0)

    def test_amat_tiebreaker_selects_closest_to_center(self):
        # Candidate 1: at (100, 100) -> distance to (500, 500) = sqrt(400^2 + 400^2) = 565.68 px
        c1 = Candidate(x=100.0, y=100.0, zncc_score=0.88, scale=10.0, rotation=0.0, rank=1, composite_score=0.88)
        # Candidate 2: at (450, 450) -> distance to (500, 500) = sqrt(50^2 + 50^2) = 70.71 px
        c2 = Candidate(x=450.0, y=450.0, zncc_score=0.88, scale=10.0, rotation=0.0, rank=2, composite_score=0.88)

        winner, tie_occurred = apply_amat_tiebreaker([c1, c2], tie_tolerance=1e-3)
        self.assertTrue(tie_occurred)
        # AMAT rule MUST select c2 because (450, 450) is closer to (500, 500) than (100, 100)
        self.assertEqual(winner.x, 450.0)
        self.assertEqual(winner.y, 450.0)


if __name__ == "__main__":
    unittest.main()
