"""Smoke tests for the ``target_size`` / reproject resampling path in ``calcflow``."""

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


def _wcs_tan_header(n: int, cdelt_deg: float) -> fits.Header:
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
    return h


def _gaussian_blob(n: int, sigma_px: float) -> np.ndarray:
    yy, xx = np.indices((n, n), dtype=np.float64)
    cy = cx = (n - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-0.5 * rr2 / float(sigma_px) ** 2)


class TestCalcflowResample(unittest.TestCase):
    def test_returns_hdu_with_target_wcs(self):
        """``calcflow`` with ``target_size`` downsamples both inputs onto a common grid."""
        # Image lives on a 96x96 grid (larger), reference on 48x48 (smaller).
        n_img = 96
        n_ref = 48
        # Same FOV (cdelt_deg * n_img == 2 * cdelt_deg/2 * n_ref, i.e. half pixel scale on the
        # smaller image keeps the same sky coverage).
        cdelt = 0.1
        ref_cdelt = cdelt * (n_img / n_ref)

        ref_hdr = _wcs_tan_header(n_ref, ref_cdelt)
        reference_small = _gaussian_blob(n_ref, sigma_px=4.0)
        ref_hdu = fits.PrimaryHDU(
            data=reference_small.astype(np.float32),
            header=ref_hdr,
        )

        img_hdr = _wcs_tan_header(n_img, cdelt)
        big_ref = _gaussian_blob(n_img, sigma_px=8.0)
        yy, xx = np.indices((n_img, n_img), dtype=np.float64)
        x = (xx - (n_img - 1) / 2.0) / float(n_img)
        y = (yy - (n_img - 1) / 2.0) / float(n_img)
        phi = 0.5 * np.cos(2.0 * np.pi * x) + 0.3 * np.sin(2.0 * np.pi * y)
        gy, gx = np.gradient(phi)
        offsets = np.stack([2.0 * gx, 2.0 * gy], axis=-1)
        image = np.asarray(Flow(offsets).apply(big_ref))

        target_size = 48  # smaller of the two

        with tempfile.TemporaryDirectory() as td:
            image_fn = str(Path(td) / "image.fits")
            fits.writeto(image_fn, image.astype(np.float32), img_hdr, overwrite=True)

            img_out, ref_out, flow, dewarped, _ = calcflow(
                image_fn,
                reference_sky=ref_hdu,
                cleaned=False,
                qa=False,
                alpha=1.3,
                gamma=150.0,
                target_size=target_size,
            )

        # Both arrays should now live on the target grid.
        self.assertEqual(np.asarray(img_out).shape, (target_size, target_size))
        self.assertEqual(np.asarray(ref_out.data).shape, (target_size, target_size))
        self.assertEqual(np.asarray(dewarped).shape, (target_size, target_size))
        self.assertEqual(np.asarray(flow.offsets).shape, (target_size, target_size, 2))

        # Reference HDU header should describe the target grid (same FOV).
        out_wcs = WCS(ref_out.header).celestial
        self.assertEqual(int(out_wcs.pixel_shape[0]), target_size)
        # Pixel scale at the target grid should match the image's downsampled scale.
        target_cdelt = abs(float(out_wcs.wcs.cdelt[0]))
        expected_cdelt = cdelt * (n_img / target_size)
        self.assertAlmostEqual(target_cdelt, expected_cdelt, places=6)

    def test_write_strips_stale_naxis_cards(self):
        """``write=True`` must drop NAXIS3/NAXIS4 inherited from radio FITS headers."""
        n = 64
        hdr = _wcs_tan_header(n, cdelt_deg=0.1)
        # Stamp a 4-axis WCS as on-disk radio FITS often does (STOKES + FREQ).
        hdr["NAXIS"] = 4
        hdr["NAXIS3"] = 1
        hdr["NAXIS4"] = 1
        hdr["CTYPE3"] = "FREQ"
        hdr["CRVAL3"] = 73e6
        hdr["CRPIX3"] = 1.0
        hdr["CDELT3"] = 1e6
        hdr["CUNIT3"] = "Hz"
        hdr["CTYPE4"] = "STOKES"
        hdr["CRVAL4"] = 1.0
        hdr["CRPIX4"] = 1.0
        hdr["CDELT4"] = 1.0

        reference = _gaussian_blob(n, sigma_px=6.0)
        image = reference

        with tempfile.TemporaryDirectory() as td:
            image_fn = str(Path(td) / "image.fits")
            fits.writeto(image_fn, image.astype(np.float32), hdr, overwrite=True)

            outroot = Path(td) / "out"
            outroot.mkdir()

            calcflow(
                image_fn,
                reference_sky=reference,
                cleaned=False,
                qa=False,
                alpha=1.3,
                gamma=150.0,
                write=True,
                write_reference_sky=True,
                outroot=str(outroot),
            )

            dewarp_path = outroot / "image_dewarp.fits"
            ref_path = outroot / "image_reference.fits"
            self.assertTrue(dewarp_path.is_file())
            self.assertTrue(ref_path.is_file())

            with fits.open(dewarp_path) as hdul:
                written_hdr = hdul[0].header
                self.assertEqual(int(written_hdr["NAXIS"]), 2)
                self.assertNotIn("NAXIS3", written_hdr)
                self.assertNotIn("NAXIS4", written_hdr)
                self.assertNotIn("CTYPE3", written_hdr)
                self.assertNotIn("CTYPE4", written_hdr)

    def test_array_reference_assumed_on_source_wcs(self):
        """An array-form ``reference_sky`` is treated as living on the source image's WCS."""
        n = 64
        hdr = _wcs_tan_header(n, cdelt_deg=0.1)
        reference = _gaussian_blob(n, sigma_px=6.0)
        image = reference  # no warp; just test the resample plumbing

        target_size = 32

        with tempfile.TemporaryDirectory() as td:
            image_fn = str(Path(td) / "image.fits")
            fits.writeto(image_fn, image.astype(np.float32), hdr, overwrite=True)

            _, ref_out, _, dewarped, _ = calcflow(
                image_fn,
                reference_sky=reference,
                cleaned=False,
                qa=False,
                alpha=1.3,
                gamma=150.0,
                target_size=target_size,
            )

        self.assertEqual(np.asarray(ref_out.data).shape, (target_size, target_size))
        self.assertEqual(np.asarray(dewarped).shape, (target_size, target_size))

    def test_write_preserves_3d_datacube(self):
        """True 3D arrays keep ``NAXIS3`` and frequency-like WCS cards on write."""
        n = 32
        nz = 2
        hdr = _wcs_tan_header(n, cdelt_deg=0.1)
        hdr["NAXIS"] = 3
        hdr["NAXIS3"] = nz
        hdr["CTYPE3"] = "FREQ"
        hdr["CRVAL3"] = 74e6
        hdr["CRPIX3"] = 1.0
        hdr["CDELT3"] = 1e6
        hdr["CUNIT3"] = "Hz"

        ref = _gaussian_blob(n, sigma_px=4.0)
        cube = np.stack([ref, ref * 0.95], axis=0).astype(np.float32)

        with tempfile.TemporaryDirectory() as td:
            image_fn = str(Path(td) / "cube.fits")
            fits.writeto(image_fn, cube, hdr, overwrite=True)

            outroot = Path(td) / "out"
            outroot.mkdir()

            calcflow(
                image_fn,
                reference_sky=ref.astype(np.float32),
                cleaned=False,
                qa=False,
                alpha=1.3,
                gamma=150.0,
                write=True,
                write_reference_sky=True,
                outroot=str(outroot),
            )

            dewarp_path = outroot / "cube_dewarp.fits"
            self.assertTrue(dewarp_path.is_file())
            with fits.open(dewarp_path) as hdul:
                self.assertEqual(hdul[0].data.shape, (nz, n, n))
                wh = hdul[0].header
                self.assertEqual(int(wh["NAXIS"]), 3)
                self.assertEqual(int(wh["NAXIS3"]), nz)
                self.assertEqual(wh["CTYPE3"], "FREQ")
                self.assertAlmostEqual(float(wh["CRVAL3"]), 74e6)


if __name__ == "__main__":
    unittest.main()
