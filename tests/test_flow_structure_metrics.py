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

from image_plane_correction import flow_metrics  # noqa: E402


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


class TestFlowDivCurl(unittest.TestCase):
    def test_constant_offsets_near_zero_div_curl(self):
        n = 64
        offsets = np.zeros((n, n, 2), dtype=np.float64)
        offsets[..., 0] = 2.0
        offsets[..., 1] = -1.5
        div, curl = flow_metrics.flow_div_curl(offsets)
        self.assertLess(np.nanmax(np.abs(div)), 1e-10)
        self.assertLess(np.nanmax(np.abs(curl)), 1e-10)

    def test_gradient_field_low_curl(self):
        n = 128
        cdelt = 0.05
        k_deg = 1.0 / 40.0  # 40° wavelength
        yy, xx = np.indices((n, n), dtype=np.float64)
        x_deg = (xx - (n - 1) / 2.0) * cdelt
        phi = np.cos(2.0 * np.pi * k_deg * x_deg)
        gy, gx = np.gradient(phi)
        offsets = np.stack([gx, gy], axis=-1)

        div, curl = flow_metrics.flow_div_curl(offsets)
        rms_curl = float(np.sqrt(np.mean(curl**2)))
        rms_div = float(np.sqrt(np.mean(div**2)))
        self.assertGreater(rms_div, 1e-6)
        self.assertLess(rms_curl, 0.2 * rms_div)

    def test_solid_rotation_high_curl(self):
        n = 64
        yy, xx = np.indices((n, n), dtype=np.float64)
        cy = cx = (n - 1) / 2.0
        u = -(yy - cy)
        v = xx - cx
        offsets = np.stack([u, v], axis=-1)
        div, curl = flow_metrics.flow_div_curl(offsets)
        self.assertLess(np.nanmax(np.abs(div)), 1e-10)
        self.assertGreater(np.nanmin(curl), 1.9)
        self.assertLess(np.nanmax(curl), 2.1)


class TestBandPower(unittest.TestCase):
    def test_band_concentrates_power_for_single_scale(self):
        # Resolve ~25° scales on the grid: λ_pix = λ_deg / cdelt must fit in the array.
        n = 512
        cdelt = 0.05
        imwcs = _wcs_tan(n, cdelt_deg=cdelt)
        dpp = flow_metrics.deg_per_pix_from_wcs(imwcs)
        self.assertAlmostEqual(dpp, cdelt, places=6)

        lam_target = 25.0
        k_deg = 1.0 / lam_target
        yy, xx = np.indices((n, n), dtype=np.float64)
        x_deg = (xx - (n - 1) / 2.0) * cdelt
        field = np.sin(2.0 * np.pi * k_deg * x_deg)

        band = (20.0, 100.0)
        p_in, p_tot = flow_metrics.band_power_from_field(
            field, dpp, band, taper=False, exclude_dc=True
        )
        self.assertGreater(p_tot, 0.0)
        frac = p_in / p_tot
        self.assertGreater(frac, 0.85)


class TestStructureScore(unittest.TestCase):
    def test_structure_score_potential_vs_rotation(self):
        n = 512
        cdelt = 0.05
        imwcs = _wcs_tan(n, cdelt_deg=cdelt)
        yy, xx = np.indices((n, n), dtype=np.float64)
        x_deg = (xx - (n - 1) / 2.0) * cdelt
        k_deg = 1.0 / 25.0
        phi = np.cos(2.0 * np.pi * k_deg * x_deg)
        gy, gx = np.gradient(phi)
        offsets_grad = np.stack([gx, gy], axis=-1)

        cy = cx = (n - 1) / 2.0
        offsets_rot = np.stack([-(yy - cy), xx - cx], axis=-1)

        s_grad = flow_metrics.structure_score(offsets_grad, imwcs, band_deg=(20.0, 100.0))
        s_rot = flow_metrics.structure_score(offsets_rot, imwcs, band_deg=(20.0, 100.0))

        self.assertGreater(s_grad["structure_score"], s_rot["structure_score"])

    def test_zero_field_returns_finite_score(self):
        n = 32
        imwcs = _wcs_tan(n)
        offsets = np.zeros((n, n, 2), dtype=np.float64)
        out = flow_metrics.structure_score(offsets, imwcs)
        self.assertEqual(out["structure_score"], 0.0)
        self.assertFalse(np.isnan(out["band_power_frac_div"]))


if __name__ == "__main__":
    unittest.main()
