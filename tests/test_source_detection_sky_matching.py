from __future__ import annotations

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


def _wcs_tan(n: int, cdelt_arcsec: float = 1.0) -> WCS:
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


def _gaussian_sources_image(
    n: int,
    centers_xy: np.ndarray,
    *,
    sigma_px: float = 5.0,
    peak_amp: float = 80.0,
    noise_sigma: float = 0.08,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    yy, xx = np.indices((n, n), dtype=np.float64)
    img = rng.normal(0.0, noise_sigma, (n, n)).astype(np.float64)
    for cx, cy in np.asarray(centers_xy, dtype=float):
        img += peak_amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px**2))
    return img.astype(np.float32, copy=False)


class TestSkyMatching(unittest.TestCase):
    def test_pixels_to_skycoord_and_match(self):
        n = 256
        wcs = _wcs_tan(n, cdelt_arcsec=120.0)

        cat_xy = np.array(
            [
                [n / 2 + 10.0, n / 2 - 5.0],
                [n / 2 - 12.0, n / 2 + 8.0],
                [n / 2 + 30.0, n / 2 + 20.0],
            ],
            dtype=float,
        )
        cat_sky = sd.pixels_to_skycoord(cat_xy, wcs)

        rng = np.random.default_rng(0)
        meas_xy = cat_xy + rng.normal(scale=0.2, size=cat_xy.shape)
        meas_sky = sd.pixels_to_skycoord(meas_xy, wcs)

        idx, sep_deg = sd.match_sky_nearest(meas_sky, cat_sky)
        self.assertTrue(np.array_equal(idx, np.array([0, 1, 2], dtype=int)))

        stats = sd.summarize_angular_separations_deg(sep_deg)
        self.assertLess(stats["p90_arcsec"], 120.0)

    def test_catalog_astrometry_qc_smoke(self):
        n = 256
        wcs = _wcs_tan(n, cdelt_arcsec=4.0)

        cat_xy = np.array(
            [
                [n / 2 + 10.0, n / 2 - 5.0],
                [n / 2 - 12.0, n / 2 + 8.0],
                [n / 2 + 30.0, n / 2 + 20.0],
            ],
            dtype=float,
        )
        image = _gaussian_sources_image(n, cat_xy, sigma_px=6.0, peak_amp=100.0)

        beam_deg = (18.0 / 3600.0, 15.0 / 3600.0)
        out = sd.catalog_astrometry_qc(
            image,
            imwcs=wcs,
            catalog=cat_xy,
            n_catalog_sources=3,
            n_measured_sources=3,
            min_separation_px=5,
            beam_fwhm_deg=(beam_deg[0], beam_deg[1]),
            max_sep_arcsec=300.0,
            min_matches=3,
            bdsf_thresh="hard",
            bdsf_thresh_isl=3.0,
            bdsf_thresh_pix=4.0,
            bdsf_minpix_isl=5,
            bdsf_quiet=True,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["n_matched"], 3)
        self.assertLess(out["p90_arcsec"], 300.0)

    def test_catalog_astrometry_qc_ignores_nan_pixels(self):
        n = 256
        wcs = _wcs_tan(n, cdelt_arcsec=4.0)

        cat_xy = np.array(
            [
                [n / 2 + 10.0, n / 2 - 5.0],
                [n / 2 - 12.0, n / 2 + 8.0],
                [n / 2 + 30.0, n / 2 + 20.0],
            ],
            dtype=float,
        )
        image = _gaussian_sources_image(n, cat_xy, sigma_px=6.0, peak_amp=100.0)
        image = np.asarray(image, dtype=np.float32)
        image[0:40, 0:40] = np.nan
        image[-30:, -30:] = np.nan
        image[100:105, 200:210] = np.nan

        beam_deg = (18.0 / 3600.0, 15.0 / 3600.0)
        out = sd.catalog_astrometry_qc(
            image,
            imwcs=wcs,
            catalog=cat_xy,
            n_catalog_sources=3,
            n_measured_sources=3,
            min_separation_px=5,
            beam_fwhm_deg=(beam_deg[0], beam_deg[1]),
            max_sep_arcsec=300.0,
            min_matches=3,
            bdsf_thresh="hard",
            bdsf_thresh_isl=3.0,
            bdsf_thresh_pix=4.0,
            bdsf_minpix_isl=5,
            bdsf_quiet=True,
        )
        self.assertTrue(out["ok"])
        self.assertTrue(np.isfinite(out["median_arcsec"]))
        self.assertTrue(np.isfinite(out["rms_arcsec"]))
        self.assertTrue(np.isfinite(out["p90_arcsec"]))


if __name__ == "__main__":
    unittest.main()
