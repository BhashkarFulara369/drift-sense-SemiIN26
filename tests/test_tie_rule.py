import unittest
from src.candidate import Candidate
from src.forensics import estimate_ambiguity
from src.tiebreak import apply_amat_tiebreaker

class TestTieRule(unittest.TestCase):
    def test_candidate_clearly_better(self):
        c1 = Candidate(x=10, y=10, scale=1.0, rotation=0.0, zncc_score=0.90, rank=1)
        c1.composite_score = 0.90
        c2 = Candidate(x=20, y=20, scale=1.0, rotation=0.0, zncc_score=0.80, rank=2)
        c2.composite_score = 0.80
        
        # Relative difference: (0.90 - 0.80) / 0.90 = 11.1% > 3%
        self.assertFalse(estimate_ambiguity([c1, c2], threshold_pct=3.0))

    def test_candidate_within_3_percent(self):
        c1 = Candidate(x=10, y=10, scale=1.0, rotation=0.0, zncc_score=0.90, rank=1)
        c1.composite_score = 0.90
        c2 = Candidate(x=20, y=20, scale=1.0, rotation=0.0, zncc_score=0.89, rank=2)
        c2.composite_score = 0.89
        
        # Relative difference: (0.90 - 0.89) / 0.90 = 1.1% <= 3%
        self.assertTrue(estimate_ambiguity([c1, c2], threshold_pct=3.0))

    def test_candidate_exactly_at_3_percent(self):
        c1 = Candidate(x=10, y=10, scale=1.0, rotation=0.0, zncc_score=1.00, rank=1)
        c1.composite_score = 1.00
        c2 = Candidate(x=20, y=20, scale=1.0, rotation=0.0, zncc_score=0.97, rank=2)
        c2.composite_score = 0.97
        
        # Relative difference: (1.00 - 0.97) / 1.00 = 3.0% <= 3.0%
        self.assertTrue(estimate_ambiguity([c1, c2], threshold_pct=3.0))

    def test_candidate_outside_3_percent(self):
        c1 = Candidate(x=10, y=10, scale=1.0, rotation=0.0, zncc_score=1.00, rank=1)
        c1.composite_score = 1.00
        c2 = Candidate(x=20, y=20, scale=1.0, rotation=0.0, zncc_score=0.9699, rank=2)
        c2.composite_score = 0.9699
        
        # Relative difference: (1.00 - 0.9699) / 1.00 = 3.01% > 3.0%
        self.assertFalse(estimate_ambiguity([c1, c2], threshold_pct=3.0))

    def test_amat_tiebreaker_closest_to_center(self):
        # Center is (500, 500)
        c1 = Candidate(x=200, y=200, scale=1.0, rotation=0.0, zncc_score=0.99, rank=1) # Distance: ~424
        c1.composite_score = 0.99
        c2 = Candidate(x=450, y=450, scale=1.0, rotation=0.0, zncc_score=1.00, rank=2) # Distance: ~70
        c2.composite_score = 1.00
        c3 = Candidate(x=800, y=800, scale=1.0, rotation=0.0, zncc_score=0.98, rank=3) # Distance: ~424
        c3.composite_score = 0.98

        # c2 is both the top score and the closest to center
        winner, tied = apply_amat_tiebreaker([c2, c1, c3], tie_tolerance=0.03, disable=False)
        self.assertTrue(tied)
        self.assertEqual(winner, c2)

        # Now make c1 the top score, but c2 is still within 3% and closer to center
        c1.composite_score = 1.00
        c2.composite_score = 0.98
        c3.composite_score = 0.97
        
        # Tie-breaker should pick c2 because it's closest to (500,500), despite c1 being the top score
        winner, tied = apply_amat_tiebreaker([c1, c2, c3], tie_tolerance=0.03, disable=False)
        self.assertTrue(tied)
        self.assertEqual(winner, c2)
        
        # If disable=True, it should just return the top score (c1)
        winner, tied = apply_amat_tiebreaker([c1, c2, c3], tie_tolerance=0.03, disable=True)
        self.assertTrue(tied)
        self.assertEqual(winner, c1)

if __name__ == '__main__':
    unittest.main()
