import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from image_plane_correction import source_detection as sd  # noqa: E402


def _wcs_tan(n: int, cdelt_deg: float = 0.05) -> WCS:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = n
    h["NAXIS2"] = n
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = 180.0
    h["CRVAL2"] = 45.0
    h["CRPIX1"] = (n + 1) / 2.0
    h["CRPIX2"] = (n + 1) / 2.0
    h["CDELT1"] = -cdelt_deg
    h["CDELT2"] = cdelt_deg
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    return WCS(h)


class TestSkyMatching(unittest.TestCase):
    def test_pixels_to_skycoord_and_match(self):
        n = 256
        wcs = _wcs_tan(n, cdelt_deg=0.1)

        # Build a tiny catalog around the field center.
        cat_xy = np.array(
            [
                [n / 2 + 10.0, n / 2 - 5.0],
                [n / 2 - 12.0, n / 2 + 8.0],
                [n / 2 + 30.0, n / 2 + 20.0],
            ],
            dtype=float,
        )
        cat_sky = sd.pixels_to_skycoord(cat_xy, wcs)

        # Measured points are catalog points plus sub-pixel noise.
        rng = np.random.default_rng(0)
        meas_xy = cat_xy + rng.normal(scale=0.2, size=cat_xy.shape)
        meas_sky = sd.pixels_to_skycoord(meas_xy, wcs)

        idx, sep_deg = sd.match_sky_nearest(meas_sky, cat_sky)
        self.assertTrue(np.array_equal(idx, np.array([0, 1, 2], dtype=int)))

        stats = sd.summarize_angular_separations_deg(sep_deg)
        # Small sub-pixel offsets at 0.1 deg/pix -> < ~1 arcmin separations.
        self.assertLess(stats["p90_arcsec"], 120.0)

    def test_catalog_astrometry_qc_smoke(self):
        n = 256
        wcs = _wcs_tan(n, cdelt_deg=0.1)

        # Build a toy image with three bright point sources at known pixels.
        image = np.zeros((n, n), dtype=float)
        cat_xy = np.array(
            [
                [n / 2 + 10.0, n / 2 - 5.0],
                [n / 2 - 12.0, n / 2 + 8.0],
                [n / 2 + 30.0, n / 2 + 20.0],
            ],
            dtype=float,
        )
        for x, y in cat_xy:
            image[int(round(y)), int(round(x))] = 10.0

        # Use centroid refinement (no beam needed) for a lightweight test.
        out = sd.catalog_astrometry_qc(
            image,
            imwcs=wcs,
            catalog=cat_xy,
            n_catalog_sources=3,
            n_measured_sources=3,
            min_separation_px=5,
            centroid_method="centroid",
            max_sep_arcsec=300.0,
            min_matches=3,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["n_matched"], 3)
        self.assertLess(out["p90_arcsec"], 300.0)


if __name__ == "__main__":
    unittest.main()

