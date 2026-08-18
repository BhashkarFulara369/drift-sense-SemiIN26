"""Unit tests for Stage G sub-pixel 2D quadratic peak refinement."""

import unittest
import numpy as np
from src.candidate import Candidate
from src.refinement import parabolic_2d_subpixel, refine_subpixel_position


class TestSubpixel(unittest.TestCase):

    def test_parabolic_2d_subpixel_exact_peak(self):
        # Synthetic symmetric quadratic peak at exact center (0,0)
        grid = np.array([
            [0.5, 0.8, 0.5],
            [0.8, 1.0, 0.8],
            [0.5, 0.8, 0.5]
        ], dtype=np.float64)

        dx, dy, peak_val = parabolic_2d_subpixel(grid)
        self.assertAlmostEqual(dx, 0.0, places=4)
        self.assertAlmostEqual(dy, 0.0, places=4)
        self.assertAlmostEqual(peak_val, 1.0, places=4)

    def test_parabolic_2d_subpixel_shifted_peak(self):
        # Peak shifted slightly right (+0.2 px) and down (+0.1 px)
        dx_true = 0.20
        dy_true = 0.10

        grid = np.zeros((3, 3), dtype=np.float64)
        for y in range(-1, 2):
            for x in range(-1, 2):
                grid[y+1, x+1] = 1.0 - 0.5 * ((x - dx_true)**2 + (y - dy_true)**2)

        dx, dy, peak_val = parabolic_2d_subpixel(grid)
        self.assertAlmostEqual(dx, dx_true, places=3)
        self.assertAlmostEqual(dy, dy_true, places=3)
        self.assertLess(abs(dx - dx_true), 0.1)
        self.assertLess(abs(dy - dy_true), 0.1)


if __name__ == "__main__":
    unittest.main()
