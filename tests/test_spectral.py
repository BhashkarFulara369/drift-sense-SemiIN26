"""Unit tests for Stage B spectral reciprocal-space estimation."""

import unittest
import numpy as np
from src.spectral import compute_fft_spectrum, find_reciprocal_peaks, compute_spectral_confidence
from src.lattice import estimate_lattice_parameters, synchronize_spectral_pose


class TestSpectral(unittest.TestCase):

    def setUp(self):
        # Create synthetic 2D sine grid image with known period (30 px) and orientation (0 deg)
        self.img_size = 512
        y, x = np.ogrid[:self.img_size, :self.img_size]
        period = 32.0
        grid = np.sin(2 * np.pi * x / period) + np.sin(2 * np.pi * y / period)
        self.synthetic_grid = (grid - grid.min()) / (grid.max() - grid.min())

    def test_fft_spectrum_shape(self):
        fft_shift, spectrum = compute_fft_spectrum(self.synthetic_grid)
        self.assertEqual(fft_shift.shape, (512, 512))
        self.assertEqual(spectrum.shape, (512, 512))

    def test_reciprocal_peaks_detection(self):
        _, spectrum = compute_fft_spectrum(self.synthetic_grid)
        peaks = find_reciprocal_peaks(spectrum, min_distance=10, dc_radius=5)
        self.assertGreater(len(peaks), 0)
        conf = compute_spectral_confidence(peaks, spectrum)
        self.assertGreater(conf, 0.0)

    def test_lattice_parameter_estimation(self):
        lattice_info = estimate_lattice_parameters(self.synthetic_grid)
        self.assertTrue(lattice_info['is_periodic'])
        self.assertGreater(lattice_info['confidence'], 0.1)

    def test_spectral_pose_synchronization(self):
        # Test pose sync on identical images
        sync_info = synchronize_spectral_pose(self.synthetic_grid, self.synthetic_grid)
        self.assertAlmostEqual(sync_info['rotation_deg'], 0.0, delta=2.0)
        self.assertAlmostEqual(sync_info['scale_factor'], 10.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
