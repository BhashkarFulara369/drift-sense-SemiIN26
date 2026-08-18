"""Unit tests for Stage D candidate generation and ZNCC matching."""

import unittest
import numpy as np
from src.candidate import prepare_reference_template, match_zncc_single, generate_top_candidates, Candidate


class TestCandidates(unittest.TestCase):

    def setUp(self):
        self.search_img = np.random.uniform(0.2, 0.8, (500, 500)).astype(np.float32)
        # Create a distinct square target in search image at (200, 200)
        self.search_img[180:220, 180:220] = 0.95
        self.ref_img = np.random.uniform(0.2, 0.8, (400, 400)).astype(np.float32)
        self.ref_img[160:240, 160:240] = 0.95

    def test_prepare_reference_template_shape(self):
        template, (th, tw) = prepare_reference_template(self.ref_img, scale_factor=10.0, rotation_deg=0.0)
        self.assertEqual(tw, 40)
        self.assertEqual(th, 40)
        self.assertEqual(template.shape, (40, 40))

    def test_zncc_matching_peak(self):
        template, _ = prepare_reference_template(self.ref_img, scale_factor=10.0)
        zmap = match_zncc_single(self.search_img, template)
        self.assertEqual(zmap.shape, (461, 461))

    def test_top_candidates_count(self):
        cands = generate_top_candidates(self.search_img, self.ref_img, init_scale=10.0, top_k=20)
        self.assertLessEqual(len(cands), 20)
        if cands:
            self.assertIsInstance(cands[0], Candidate)
            self.assertEqual(cands[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
