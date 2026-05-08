from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from image_plane_correction import catalogs  # noqa: E402


def _wcs_tan(n: int, cdelt_arcsec: float = 60.0) -> WCS:
    cdelt_deg = float(cdelt_arcsec) / 3600.0
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


class TestTheoreticalSkyNanInputs(unittest.TestCase):
    def test_nan_flux_does_not_poison_output(self):
        n = 128
        wcs = _wcs_tan(n)
        psf = np.zeros((31, 31), dtype=np.float32)
        psf[15, 15] = 1.0

        positions = SkyCoord([180.0, 180.1] * u.deg, [45.0, 45.05] * u.deg)
        fluxes = np.array([1.0, np.nan], dtype=np.float32)

        def fake_reference_sources(_catalog, min_flux=0, path=None):
            _ = (min_flux, path)
            return positions, fluxes

        with mock.patch.object(catalogs, "reference_sources", side_effect=fake_reference_sources):
            out = catalogs.theoretical_sky_beam_function(wcs, psf, img_size=n)

        self.assertTrue(np.isfinite(np.asarray(out)).all())
        self.assertGreater(float(np.nanmax(np.asarray(out))), 0.0)


if __name__ == "__main__":
    unittest.main()

