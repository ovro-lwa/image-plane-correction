"""calcflow(dewarp=False) skips apply and returns None for the dewarped image."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from image_plane_correction.flow import Flow, calcflow  # noqa: E402


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


class TestCalcflowDewarpFlag(unittest.TestCase):
    def test_dewarp_false_returns_none_and_skips_apply(self) -> None:
        n = 32
        imwcs = _wcs_tan(n)
        hdr = imwcs.to_header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = n
        hdr["NAXIS2"] = n
        image = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
        reference = np.zeros((n, n), dtype=np.float32)
        reference[n // 2, n // 2] = 1.0

        with tempfile.TemporaryDirectory() as td:
            image_fn = str(Path(td) / "img.fits")
            fits.writeto(image_fn, image, hdr, overwrite=True)

            real_apply = Flow.apply_broadcast
            with mock.patch.object(
                Flow, "apply_broadcast", autospec=True, side_effect=real_apply
            ) as apply_broadcast:
                _img, _ref, flow, dewarped, qa_ok = calcflow(
                    image_fn,
                    reference_sky=reference,
                    cleaned=False,
                    qa=False,
                    write=False,
                    dewarp=False,
                )

        self.assertIsNone(dewarped)
        self.assertTrue(qa_ok)
        self.assertEqual(tuple(np.asarray(flow.offsets).shape), (n, n, 2))
        apply_broadcast.assert_not_called()

    def test_fits_image_return_header_single_open(self) -> None:
        from image_plane_correction import data

        n = 8
        imwcs = _wcs_tan(n)
        hdr = imwcs.to_header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = n
        hdr["NAXIS2"] = n
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "tiny.fits")
            fits.writeto(path, np.ones((n, n), dtype=np.float32), hdr, overwrite=True)
            arr, w, header = data.fits_image(path, return_header=True)
        self.assertEqual(tuple(np.asarray(arr).shape), (n, n))
        self.assertEqual(int(header["NAXIS1"]), n)
        self.assertEqual(w.naxis, 2)


if __name__ == "__main__":
    unittest.main()
