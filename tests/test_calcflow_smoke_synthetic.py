import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from image_plane_correction.flow import Flow, calcflow  # noqa: E402
from image_plane_correction.flow_metrics import structure_score  # noqa: E402


def _wcs_tan(n: int, cdelt_deg: float = 0.1) -> WCS:
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


def _gaussian_blob(n: int, *, sigma_px: float = 6.0) -> np.ndarray:
    yy, xx = np.indices((n, n), dtype=np.float64)
    cy = cx = (n - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-0.5 * rr2 / float(sigma_px) ** 2)


class TestCalcflowSyntheticSmoke(unittest.TestCase):
    def test_calcflow_dewarps_smooth_potential_warp(self):
        n = 64
        imwcs = _wcs_tan(n, cdelt_deg=0.1)
        hdr = imwcs.to_header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = n
        hdr["NAXIS2"] = n

        reference = _gaussian_blob(n, sigma_px=6.0)

        # Smooth, mostly curl-free displacement as grad(phi).
        yy, xx = np.indices((n, n), dtype=np.float64)
        x = (xx - (n - 1) / 2.0) / float(n)
        y = (yy - (n - 1) / 2.0) / float(n)
        phi = 0.5 * np.cos(2.0 * np.pi * x) + 0.3 * np.sin(2.0 * np.pi * y)
        gy, gx = np.gradient(phi)
        amp = 2.0
        offsets = np.stack([amp * gx, amp * gy], axis=-1)

        image = np.asarray(Flow(offsets).apply(reference))

        with tempfile.TemporaryDirectory() as td:
            image_fn = str(Path(td) / "synthetic-image.fits")
            fits.writeto(image_fn, image.astype(np.float32), hdr, overwrite=True)

            img_out, ref_out, flow, dewarped, qa_ok = calcflow(
                image_fn,
                reference_sky=reference,
                cleaned=False,
                qa=False,
                alpha=1.3,
                gamma=150.0,
            )

        mse_before = float(np.mean((np.asarray(img_out) - np.asarray(ref_out)) ** 2))
        mse_after = float(np.mean((np.asarray(dewarped) - np.asarray(ref_out)) ** 2))
        self.assertLess(mse_after, mse_before)
        self.assertTrue(qa_ok)

        s = structure_score(np.asarray(flow.offsets), imwcs, band_deg=(20.0, 100.0))
        self.assertLess(s["curl_to_div_ratio_band"], 2.0)


if __name__ == "__main__":
    unittest.main()

