from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_plane_correction.catalogs import (  # noqa: E402
    reference_sources_nvss,
    reference_sources_vlssr,
)


class TestReferenceSourcesVlssr(unittest.TestCase):
    def test_min_flux_in_jy_filters_peak_int(self):
        catalog_text = (
            'RA DEC "PEAK INT"\n'
            "180.0 45.0 0.005\n"
            "180.1 45.1 0.05\n"
            "180.2 45.2 0.5\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(catalog_text)
            path = fh.name

        reference_sources_vlssr.cache_clear()
        try:
            positions, fluxes = reference_sources_vlssr(min_flux=0.01, path=path)
        finally:
            reference_sources_vlssr.cache_clear()

        self.assertEqual(len(positions), 2)
        self.assertAlmostEqual(float(fluxes[0]), 0.05)
        self.assertAlmostEqual(float(fluxes[1]), 0.5)

    def test_max_flux_in_jy_filters_peak_int(self):
        catalog_text = (
            'RA DEC "PEAK INT"\n'
            "180.0 45.0 0.005\n"
            "180.1 45.1 0.05\n"
            "180.2 45.2 0.5\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(catalog_text)
            path = fh.name

        reference_sources_vlssr.cache_clear()
        try:
            positions, fluxes = reference_sources_vlssr(
                min_flux=0.0, max_flux=0.1, path=path
            )
        finally:
            reference_sources_vlssr.cache_clear()

        self.assertEqual(len(positions), 2)
        self.assertAlmostEqual(float(fluxes[0]), 0.005)
        self.assertAlmostEqual(float(fluxes[1]), 0.05)


class TestReferenceSourcesNvss(unittest.TestCase):
    def test_min_flux_in_jy_converts_mjy_catalog(self):
        catalog_text = (
            "rah ram ras dd dm ds f\n"
            "12 0 0 45 0 0 500\n"
            "12 0 1 45 0 0 5000\n"
            "12 0 2 45 0 0 50000\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as fh:
            fh.write(catalog_text)
            path = fh.name

        positions, fluxes = reference_sources_nvss(min_flux=1.0, path=path)

        self.assertEqual(len(positions), 2)
        self.assertAlmostEqual(float(fluxes[0]), 5.0)
        self.assertAlmostEqual(float(fluxes[1]), 50.0)

    def test_max_flux_in_jy_converts_mjy_catalog(self):
        catalog_text = (
            "rah ram ras dd dm ds f\n"
            "12 0 0 45 0 0 500\n"
            "12 0 1 45 0 0 5000\n"
            "12 0 2 45 0 0 50000\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as fh:
            fh.write(catalog_text)
            path = fh.name

        positions, fluxes = reference_sources_nvss(
            min_flux=0.0, max_flux=10.0, path=path
        )

        self.assertEqual(len(positions), 2)
        self.assertAlmostEqual(float(fluxes[0]), 0.5)
        self.assertAlmostEqual(float(fluxes[1]), 5.0)


if __name__ == "__main__":
    unittest.main()
