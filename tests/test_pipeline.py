"""
Automated sanity tests for the UTTHAAN traffic intelligence prototype.

Run with:  python -m pytest tests/ -v
       or:  python -m unittest discover tests
"""
import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, reid
from src.association import link_confidence, plate_similarity
from src.pipeline import run
from src.synthetic_data import generate_dataset

TEST_DATA_DIR = os.path.join(config.BASE_DIR, "tests", "_tmp_data")
TEST_OUTPUT_DIR = os.path.join(config.BASE_DIR, "tests", "_tmp_outputs")


class TestReIDAndAssociation(unittest.TestCase):
    def test_plate_similarity_exact(self):
        self.assertEqual(plate_similarity("JK12AB3456", "JK12AB3456"), 1.0)

    def test_plate_similarity_missing(self):
        self.assertIsNone(plate_similarity(None, "JK12AB3456"))

    def test_plate_similarity_one_char_off(self):
        score = plate_similarity("JK12AB3456", "JK12AB3457")
        self.assertGreater(score, 0.85)

    def test_reid_similarity_identical(self):
        import numpy as np
        vec = np.random.rand(64)
        self.assertAlmostEqual(reid.similarity(vec, vec), 1.0, places=5)

    def test_reid_similarity_zero_vector(self):
        import numpy as np
        self.assertEqual(reid.similarity(np.zeros(64), np.random.rand(64)), 0.0)


class TestEndToEndPipeline(unittest.TestCase):
    """Generates a fresh synthetic dataset and runs the full pipeline against
    it, asserting the system meets a minimum reconstruction accuracy bar and
    produces every expected output artifact. This is the closest thing to a
    'does the whole prototype actually work' integration test."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        generate_dataset(TEST_DATA_DIR)
        cls.result = run(TEST_DATA_DIR, TEST_OUTPUT_DIR, verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
        shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)

    def test_output_files_exist(self):
        for fname in ["city_traffic_map.html", "traffic_analytics.json",
                      "trajectories.json", "validation_report.json"]:
            self.assertTrue(os.path.exists(os.path.join(TEST_OUTPUT_DIR, fname)),
                             f"missing output: {fname}")

    def test_minimum_reconstruction_accuracy(self):
        accuracy = self.result["validation"]["accuracy_pct"]
        self.assertGreaterEqual(accuracy, 50.0,
                                 "Cross-camera trajectory reconstruction accuracy dropped "
                                 "below the expected minimum for the synthetic benchmark.")

    def test_at_least_one_multi_camera_trajectory(self):
        multi_cam = [t for t in self.result["trajectories"] if len(t.camera_sequence) > 1]
        self.assertGreater(len(multi_cam), 0)

    def test_gis_map_is_valid_html(self):
        with open(os.path.join(TEST_OUTPUT_DIR, "city_traffic_map.html")) as fh:
            content = fh.read()
        self.assertIn("<html", content.lower())
        self.assertIn("leaflet", content.lower())


if __name__ == "__main__":
    unittest.main()
