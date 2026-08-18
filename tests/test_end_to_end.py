"""End-to-end integration test for Drift-Sense Localization Pipeline."""

import unittest
from pathlib import Path
import cv2
import numpy as np
from localize import DriftSenseEngine


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = Path("dataset/synthetic_sem_dataset")
        self.ref_path = self.dataset_dir / "reference" / "sample_001.png"
        self.search_path = self.dataset_dir / "search" / "sample_001.png"

    def test_end_to_end_localization_pipeline(self):
        if not self.ref_path.exists() or not self.search_path.exists():
            self.skipTest("Synthetic dataset sample_001 not found.")

        ref_img = cv2.imread(str(self.ref_path), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(str(self.search_path), cv2.IMREAD_GRAYSCALE)

        engine = DriftSenseEngine(verbose=False)
        x, y, diagnostics = engine.localize(ref_img, search_img)

        # Basic Sanity Assertions
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertGreaterEqual(x, 0.0)
        self.assertLessEqual(x, 1000.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y, 1000.0)
        self.assertIn('forensics', diagnostics)


if __name__ == "__main__":
    unittest.main()
