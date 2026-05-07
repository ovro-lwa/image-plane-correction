"""Unit tests for catalog-based QC helpers (separations, sky matching)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402

from image_plane_correction import source_detection as sd  # noqa: E402


class TestSummarizeSeparations(unittest.TestCase):
    def test_outlier_does_not_dominate_median(self):
        d = np.array([1.0, 1.1, 1.0, 100.0], dtype=float)
        stats = sd.summarize_separations(d)
        self.assertLess(stats["median"], 10.0)
        self.assertGreater(stats["rms"], 40.0)
        self.assertLess(stats["p90"], 100.0)

    def test_angular_summary_deg_arcsec_scale(self):
        sep_deg = np.array([1 / 3600.0, 2 / 3600.0], dtype=float)
        ang = sd.summarize_angular_separations_deg(sep_deg)
        self.assertAlmostEqual(ang["median_arcsec"], 1.5, places=6)


class TestSkyMatchingSynthetic(unittest.TestCase):
    def test_match_nearest_expected_pairing(self):
        cat = SkyCoord([10.0, 20.0] * u.deg, [20.0, 30.0] * u.deg)
        meas = SkyCoord([10.001, 20.001] * u.deg, [20.001, 30.001] * u.deg)
        idx, sep_deg = sd.match_sky_nearest(meas, cat)
        self.assertTrue(np.array_equal(idx, np.array([0, 1])))


if __name__ == "__main__":
    unittest.main()
